"""Audit probes use generated records only; counterexamples intentionally demonstrate gaps."""
import importlib.util
import logging
from pathlib import Path
import subprocess
import socket
from unittest.mock import patch
import pandas as pd
import pytest
from src.local_processing import aggregate_candidates, run_paths, synthetic_inventory, pseudonym_key, process_inventory
from src.restricted_inventory import validate_inventory, prepare_messages, participant_id, InventoryValidationError
from src.embedding_model import LocalEmbedder, MODEL_DIR

def audit_questions(size):
    return pd.DataFrame({'message_id':[f'FAKE-M{i}' for i in range(size)],
        'participant_id':[f'FAKE-P{i}' for i in range(size)], 'cluster_id':['C1']*size,
        'theme':['Tool calling']*size, 'topic':['Agentic AI']*size, 'needs_review':[False]*size})

def test_differencing_requires_a_second_release_which_is_rejected(tmp_path):
    from src.public_release import release_request,publish_fixed_snapshot,export_snapshot_view
    q=audit_questions(6)
    approval=release_request(q,'synthetic')
    approval.update(approved=True,identity_scope_verified=True,privacy_review_completed=True,review_reference='synthetic-test')
    publish_fixed_snapshot(tmp_path,'synthetic',q,approval)
    assert export_snapshot_view(tmp_path,'synthetic')['themes'][0]['participant_count']==6
    smaller=audit_questions(5)
    newer=release_request(smaller,'synthetic')
    newer.update(approved=True,identity_scope_verified=True,privacy_review_completed=True,review_reference='synthetic-test')
    with pytest.raises(InventoryValidationError,match='already'):
        publish_fixed_snapshot(tmp_path,'synthetic',smaller,newer)
    with pytest.raises(InventoryValidationError):
        export_snapshot_view(tmp_path,'synthetic',view='exclude-one-person')

def test_counterexample_aliases_can_inflate_distinct_people():
    key=b'A'*32
    five_aliases=['Fictional learner','Fictional learner (mobile)','Fictional learner 2026','Fictional learner guest','Fictional learner laptop']
    assert len({participant_id(name,key) for name in five_aliases})==5
    # Input has no account ID, so these cannot be validated as five actual people.

def test_vault_traceability_is_reversible_and_must_stay_restricted():
    inventory,_=validate_inventory(synthetic_inventory())
    prepared=prepare_messages(inventory,b'A'*32,'synthetic')
    # Re-identification remains possible to authorized vault holders. This is
    # intentionally not a claim that HMAC pseudonyms are irreversible.
    reconstructed=prepared[['message_id','participant_id']].merge(inventory[['message_id','display_name']],on='message_id')
    assert reconstructed.display_name.notna().all()
    assert reconstructed.participant_id.nunique()==6

def test_pseudonyms_remain_stable_when_private_scope_key_is_reused(tmp_path):
    first=pseudonym_key(tmp_path,'synthetic')
    second=pseudonym_key(tmp_path,'synthetic')
    assert first==second and len(first)==32
    assert participant_id(' Fictional learner ',first)==participant_id('fictional learner',second)

def test_counterexample_nested_mode_redirect_can_alias_storage(tmp_path,monkeypatch):
    # Simulate directory-junction resolution without creating Windows links,
    # reading CSVs, or running real processing. Pure path planning only.
    root=tmp_path.resolve()
    real_base=root/'data/private/real'
    synthetic_base=root/'data/private/synthetic'
    original=Path.resolve
    def redirect(path,*args,**kwargs):
        resolved=original(path,*args,**kwargs)
        if resolved.is_relative_to(real_base):
            return synthetic_base/resolved.relative_to(real_base)
        return resolved
    monkeypatch.setattr(Path,'resolve',redirect)
    with pytest.raises(InventoryValidationError,match='alias'):
        run_paths(root,'real','audit')

def test_gitignore_rules_cover_restricted_paths_without_reading_them(tmp_path):
    ignore=Path(__file__).parent/'.gitignore'
    subprocess.run(['git','init','--quiet',str(tmp_path)],check=True,capture_output=True)
    (tmp_path/'.gitignore').write_text(ignore.read_text(encoding='utf-8'),encoding='utf-8')
    paths=['data/private/real/input.csv','data/private/real/pseudonym.key',
        'data/private/real/runs/example/review_queue.csv','data/private/real/runs/example/cluster_membership.csv',
        'data/staging/real/aggregate_candidates/example/topic_counts.csv','data/staging/synthetic/fixtures/demo.csv']
    result=subprocess.run(['git','-C',str(tmp_path),'check-ignore','--no-index','--stdin','-z'],
                          input=('\0'.join(paths)+'\0').encode(),capture_output=True,check=True)
    assert {value.decode() for value in result.stdout.split(b'\0') if value}==set(paths)

def test_invalid_inventory_is_rejected_before_any_embedding(tmp_path):
    path=tmp_path/'data/staging/synthetic/fixtures/bad.csv'
    path.parent.mkdir(parents=True)
    invalid=synthetic_inventory()
    invalid.loc[0,'timestamp']='BAD_SYNTHETIC_SENTINEL'
    invalid.to_csv(path,index=False)
    class MustNotInfer:
        def encode(self,*args,**kwargs):
            raise AssertionError('Invalid input must not reach inference')
    with pytest.raises(InventoryValidationError):
        process_inventory(path,mode='synthetic',run_id='bad',project_root=tmp_path,_embedder=MustNotInfer())
    assert not (tmp_path/'data/private').exists()

def test_cli_success_and_unexpected_error_logs_do_not_emit_sensitive_values(tmp_path,capsys,caplog):
    spec=importlib.util.spec_from_file_location('preflight_cli',Path(__file__).parent/'scripts/process_inventory.py')
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch=pytest.MonkeyPatch()
    frame=synthetic_inventory()
    frame.loc[0,'message']+=' SYNTHETIC_TEXT_SENTINEL'
    frame.loc[0,'display_name']='SYNTHETIC_NAME_SENTINEL'
    frame.loc[0,'message_id']='SYNTHETIC_ID_SENTINEL'
    caplog.set_level(logging.INFO)
    with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline only')):
        model=LocalEmbedder()
        monkeypatch.setattr(cli,'ROOT',tmp_path)
        monkeypatch.setattr(cli,'synthetic_inventory',lambda:frame)
        monkeypatch.setattr(cli,'process_inventory',lambda path,**kwargs:process_inventory(path,project_root=tmp_path,_embedder=model,**kwargs))
        assert cli.main(['--mode','synthetic','--run-id','log-probe','--model-dir',str(MODEL_DIR)])==0
    captured=capsys.readouterr()
    for sentinel in ['SYNTHETIC_TEXT_SENTINEL','SYNTHETIC_NAME_SENTINEL','SYNTHETIC_ID_SENTINEL']:
        assert sentinel not in captured.out+captured.err+caplog.text
    def failing(*args,**kwargs):
        raise RuntimeError('SYNTHETIC_ERROR_SENTINEL')
    monkeypatch.setattr(cli,'process_inventory',failing)
    assert cli.main(['--mode','synthetic','--run-id','error-probe'])==2
    captured=capsys.readouterr()
    assert 'SYNTHETIC_ERROR_SENTINEL' not in captured.out+captured.err
    monkeypatch.undo()
