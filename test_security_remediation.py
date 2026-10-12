"""Synthetic-only regression tests for fixed releases, identity separation and paths."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pandas as pd
import pytest
from src.public_release import release_request,publish_fixed_snapshot,export_snapshot_view
from src.storage_security import canonical_path,verify_mode_isolation,verify_no_overlap,restrict_permissions
from src.local_processing import run_paths
from src.restricted_inventory import InventoryValidationError,validate_inventory
from test_preflight_audit import audit_questions

def approval_for(q):
    approval=release_request(q,'synthetic')
    approval.update(approved=True,identity_scope_verified=True,privacy_review_completed=True,review_reference='synthetic-audit-review')
    return approval

def test_unapproved_release_creates_no_public_artifact(tmp_path):
    q=audit_questions(6)
    with pytest.raises(InventoryValidationError,match='approval'):
        publish_fixed_snapshot(tmp_path,'synthetic',q,release_request(q,'synthetic'))
    assert not (tmp_path/'data').exists()

def test_approval_is_bound_to_exact_membership_and_policy(tmp_path):
    q=audit_questions(6)
    approval=approval_for(q)
    changed=q.copy();changed.loc[0,'participant_id']='DIFFERENT-FAKE-PERSON'
    with pytest.raises(InventoryValidationError,match='exact'):
        publish_fixed_snapshot(tmp_path,'synthetic',changed,approval)
    approval['policy_sha256']='unapproved'
    with pytest.raises(InventoryValidationError):
        publish_fixed_snapshot(tmp_path,'synthetic',q,approval)

def test_alias_scope_requires_explicit_identity_review(tmp_path):
    q=audit_questions(5)
    approval=approval_for(q);approval['identity_scope_verified']=False
    with pytest.raises(InventoryValidationError):
        publish_fixed_snapshot(tmp_path,'synthetic',q,approval)

def test_fixed_views_never_recompute_cohorts_or_parent_person_counts(tmp_path,monkeypatch):
    q=audit_questions(6)
    second=audit_questions(6).assign(cluster_id='C2',theme='Retrieval quality',topic='Models & Architecture')
    q=pd.concat([q,second],ignore_index=True)
    publish_fixed_snapshot(tmp_path,'synthetic',q,approval_for(q))
    monkeypatch.setattr(pd,'read_csv',lambda *a,**k:pytest.fail('Export must not read restricted row-level CSVs'))
    whole=export_snapshot_view(tmp_path,'synthetic')
    agent=export_snapshot_view(tmp_path,'synthetic',view='agentic')
    pair=export_snapshot_view(tmp_path,'synthetic',view='agentic-and-models')
    assert pair==whole
    assert agent['themes']==[row for row in whole['themes'] if row['topic']=='Agentic AI']
    assert all('participant_count' not in row for row in whole['topics'])
    assert agent==export_snapshot_view(tmp_path,'synthetic',view='agentic')

@pytest.mark.parametrize('view,geography',[('minus-one','global'),('all','India'),('date-range','global'),('arbitrary-topic-pair','global')])
def test_unapproved_filter_combinations_fail_closed(tmp_path,view,geography):
    q=audit_questions(6);publish_fixed_snapshot(tmp_path,'synthetic',q,approval_for(q))
    with pytest.raises(InventoryValidationError,match='covered'):
        export_snapshot_view(tmp_path,'synthetic',view=view,geography=geography)

def test_second_snapshot_and_changed_version_are_blocked(tmp_path,monkeypatch):
    q=audit_questions(6);publish_fixed_snapshot(tmp_path,'synthetic',q,approval_for(q))
    with pytest.raises(InventoryValidationError,match='already'):
        publish_fixed_snapshot(tmp_path,'synthetic',q,approval_for(q))
    import src.public_release as module
    original=module.policy()
    monkeypatch.setattr(module,'policy',lambda:{**original,'version':'week1-v2'})
    with pytest.raises(InventoryValidationError,match='already'):
        publish_fixed_snapshot(tmp_path,'synthetic',audit_questions(5),approval_for(audit_questions(5)))

def test_small_groups_and_complementary_totals_not_released(tmp_path):
    large=audit_questions(6)
    small=audit_questions(2).assign(cluster_id='C2',theme='Agent evaluation')
    q=pd.concat([large,small],ignore_index=True)
    path=publish_fixed_snapshot(tmp_path,'synthetic',q,approval_for(q))
    payload=json.loads(path.read_text())
    assert len(payload['tables']['themes'])==1
    assert payload['tables']['topics'][0]['question_count']==6
    assert 'total_questions' not in payload and 'suppressed_count' not in payload
    assert 'participant_count' not in payload['tables']['topics'][0]

def test_same_theme_overlapping_cluster_rows_are_collapsed(tmp_path):
    a=audit_questions(6)
    b=audit_questions(5).assign(cluster_id='C2')
    q=pd.concat([a,b],ignore_index=True)
    publish_fixed_snapshot(tmp_path,'synthetic',q,approval_for(q))
    rows=export_snapshot_view(tmp_path,'synthetic')['themes']
    assert len(rows)==1
    assert rows[0]['participant_count']==6 and rows[0]['question_count']==11

def test_permission_failure_stops_before_sensitive_writes(tmp_path,monkeypatch):
    from src.storage_security import secure_directory
    monkeypatch.setattr('src.storage_security.restrict_permissions',lambda *a:(_ for _ in ()).throw(InventoryValidationError('Permissions unavailable')))
    with pytest.raises(InventoryValidationError):
        secure_directory(tmp_path/'data/private/synthetic/identity_vault',tmp_path)
    assert not (tmp_path/'data/private/synthetic/identity_vault/identity_mapping.csv').exists()

def test_path_validation_precedes_model_loading_and_input_read(tmp_path,monkeypatch):
    from src.local_processing import process_inventory
    monkeypatch.setattr('src.local_processing.LocalEmbedder',lambda *a:pytest.fail('Model must not load'))
    monkeypatch.setattr(pd,'read_csv',lambda *a,**k:pytest.fail('Input must not be read'))
    with pytest.raises(InventoryValidationError):
        process_inventory(tmp_path/'public/unsafe.csv',mode='synthetic',run_id='blocked',project_root=tmp_path)
    assert not (tmp_path/'data').exists()

def test_empty_fixed_release_requires_approval_and_has_no_totals(tmp_path):
    q=audit_questions(0)
    publish_fixed_snapshot(tmp_path,'synthetic',q,approval_for(q))
    assert export_snapshot_view(tmp_path,'synthetic')=={'themes':[],'topics':[]}

def test_unmarked_or_restricted_data_cannot_use_legacy_demo_export():
    from src.learning_analytics import privacy_safe_export
    q=audit_questions(6)
    with pytest.raises(ValueError,match='snapshot'):
        privacy_safe_export(q)
    q.attrs.update(synthetic=False,data_mode='real')
    with pytest.raises(ValueError,match='snapshot'):
        privacy_safe_export(q)

def test_public_snapshot_has_no_row_identifiers_and_tampering_is_rejected(tmp_path):
    q=audit_questions(6);q['question_ref']=['PRIVATE-REFERENCE'+str(i) for i in range(6)]
    path=publish_fixed_snapshot(tmp_path,'synthetic',q,approval_for(q))
    text=path.read_text()
    for field in ['message_id','participant_id','question_ref','cluster_id','dataset_sha256','review_reference','display_name']:
        assert field not in text
    for value in list(q.participant_id)+list(q.question_ref):
        assert value not in text
    path.write_text(text.replace('"question_count":6','"question_count":5'))
    with pytest.raises(InventoryValidationError,match='integrity'):
        export_snapshot_view(tmp_path,'synthetic')

def test_physical_verification_failure_is_closed(tmp_path,monkeypatch):
    original=Path.lstat
    def denied(path,*args,**kwargs):
        if path==tmp_path/'data':
            raise PermissionError('SENSITIVE_PATH_SENTINEL')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',denied)
    with pytest.raises(InventoryValidationError,match='verify') as error:
        run_paths(tmp_path,'synthetic','blocked')
    assert 'SENSITIVE' not in str(error.value)

def test_reparse_attributes_rejected_even_if_not_a_symlink(tmp_path,monkeypatch):
    original=Path.lstat
    def reparse(path,*args,**kwargs):
        if path==tmp_path/'data':
            return SimpleNamespace(st_mode=0o40700,st_file_attributes=0x400)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',reparse)
    with pytest.raises(InventoryValidationError,match='reparse'):
        run_paths(tmp_path,'synthetic','blocked')

def test_input_output_overlap_rejected(tmp_path):
    with pytest.raises(InventoryValidationError,match='overlap'):
        verify_no_overlap(tmp_path/'data/private/synthetic/runs/x/input.csv',[tmp_path/'data/private/synthetic/runs/x'])

@pytest.mark.skipif(os.name!='nt',reason='Native Windows junction test')
def test_native_windows_junction_rejected(tmp_path):
    # Junction exists only inside this generated temporary fixture; no real CSV.
    target=tmp_path/'target';target.mkdir()
    link=tmp_path/'data/private/synthetic';link.parent.mkdir(parents=True)
    command=f'New-Item -ItemType Junction -Path "{link}" -Target "{target}" | Out-Null'
    subprocess.run(['powershell','-NoProfile','-Command',command],check=True,capture_output=True)
    try:
        with pytest.raises(InventoryValidationError,match='reparse'):
            run_paths(tmp_path,'synthetic','blocked')
    finally:
        # rmdir removes the junction entry only, never traverses its target.
        link.rmdir()

@pytest.mark.skipif(os.name!='nt',reason='Native Windows DACL test')
def test_windows_protected_dacl_has_owner_and_system_only(tmp_path):
    file=tmp_path/'synthetic-acl-probe.txt';file.write_text('synthetic')
    restrict_permissions(file)
    command=f'$a=Get-Acl -LiteralPath "{file}"; $a.AreAccessRulesProtected; $a.Access.Count'
    result=subprocess.run(['powershell','-NoProfile','-Command',command],check=True,capture_output=True,text=True)
    assert result.stdout.splitlines()==['True','2']

@pytest.mark.parametrize('timezone',['EST','not-a-zone','+05:30',''])
def test_naive_timezone_must_be_explicit_iana_region(timezone):
    from src.local_processing import synthetic_inventory
    frame=synthetic_inventory();frame.loc[0,'timestamp']='2026-01-15T09:00:00'
    with pytest.raises(InventoryValidationError):
        validate_inventory(frame,assume_timezone=timezone)

def test_offsets_and_fall_dst_ambiguity():
    from src.local_processing import synthetic_inventory
    frame=synthetic_inventory();frame.loc[0,'timestamp']='2026-01-15T09:00:00-05:00'
    validated,_=validate_inventory(frame)
    assert validated.iloc[0].timestamp_utc=='2026-01-15T14:00:00+00:00'
    frame.loc[0,'timestamp']='2026-11-01T01:30:00'
    with pytest.raises(InventoryValidationError):
        validate_inventory(frame,assume_timezone='America/Toronto')
