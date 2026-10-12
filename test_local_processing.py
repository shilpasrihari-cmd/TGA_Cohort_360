"""All inputs in this test module are generated synthetic records, never real CSVs."""
import importlib.util
import io
import json
import socket
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest
from src.embedding_model import LocalEmbedder, ModelUnavailableError
from src.local_processing import (synthetic_inventory, process_inventory, aggregate_candidates,
                                  run_paths, restricted_path)
from src.restricted_inventory import (InventoryValidationError, validate_inventory,
                                      participant_id, prepare_messages, REQUIRED_COLUMNS)
from src.pipeline import analyze_messages, analyze_authorized_messages

@pytest.fixture(scope='module')
def local_model():
    with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline inference only')), \
         patch.object(socket,'create_connection',side_effect=AssertionError('Offline inference only')):
        yield LocalEmbedder()

def fixture_csv(root, run_id, frame):
    path = root / 'data' / 'staging' / 'synthetic' / 'fixtures' / f'{run_id}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path,index=False)
    return path

@pytest.fixture
def processed(tmp_path, local_model):
    path=fixture_csv(tmp_path,'test-run',synthetic_inventory())
    manifest=process_inventory(path,mode='synthetic',run_id='test-run',project_root=tmp_path,_embedder=local_model)
    return (tmp_path/'data/private/synthetic/runs/test-run',
            tmp_path/'data/staging/synthetic/aggregate_candidates/test-run',manifest)

def test_count_reconciliation_and_traceability(processed):
    private, candidate, manifest=processed
    assert manifest['input_records']==17
    assert manifest['unique_messages']==17
    assert manifest['detected_requests']==14
    assert manifest['nonrequests']==3
    assert manifest['question_counts_by_topic']==14
    assert manifest['question_counts_by_cluster']==14
    assert manifest['preliminary_labels_used_as_ground_truth'] is False
    questions=pd.read_csv(private/'analysis/classified_questions.csv')
    inventory=pd.read_csv(private/'identity_vault/validated_inventory.csv')
    membership=pd.read_csv(private/'identity_vault/cluster_membership.csv')
    trace=pd.read_csv(private/'identity_vault/source_trace.csv')
    assert set(questions.question_ref)<=set(trace.question_ref)
    assert set(membership.message_id)<=set(inventory.message_id)
    assert 'message_id' not in questions and 'text' not in questions
    assert not inventory.message_id.duplicated().any()
    assert questions.groupby('cluster_id').participant_id.nunique().max()==6
    assert manifest['detected_requests']+manifest['nonrequests']==manifest['unique_messages']
    assert not {'display_name','text','message','message_id','timestamp'} & set(questions.columns)
    identity=pd.read_csv(private/'identity_vault/identity_mapping.csv')
    assert {'participant_id','display_name'}<=set(identity)
    detected=pd.read_csv(private/'analysis/detected_messages.csv')
    assert not {'display_name','text','message','message_id'} & set(detected)

def test_public_candidates_exclude_sensitive_fields_and_small_groups(processed):
    private, candidate, _=processed
    themes=pd.read_csv(candidate/'recurring_themes.csv')
    topics=pd.read_csv(candidate/'topic_counts.csv')
    assert len(themes)==2
    assert themes.question_count.sum()==12  # Small 2-person course-support group stays restricted.
    assert topics.question_count.sum()==12  # Parent totals don't disclose its suppressed contribution.
    assert 'Course Support' not in topics.topic.tolist()
    assert themes.participant_count.min()>=5
    assert (themes.release_status=='Draft - human review required').all()
    sensitive={'message_id','display_name','timestamp','message','text','participant_id','source_record_numbers','cluster_id'}
    assert not sensitive & set(themes.columns)
    assert not sensitive & set(topics.columns)
    payload=''.join(p.read_text(encoding='utf-8') for p in candidate.iterdir())
    for row in synthetic_inventory().itertuples():
        assert row.display_name not in payload
        assert row.message not in payload
        assert row.message_id not in payload
    questions=pd.read_csv(private/'analysis/classified_questions.csv')
    assert not any(value in payload for value in questions.participant_id)

def test_pseudonymous_deduplication_stable_and_keyed():
    key=b'A'*32
    assert participant_id('  Fictional   Learner 1 ',key)==participant_id('fictional learner 1',key)
    assert participant_id('Fictional learner 1',key)!=participant_id('Fictional learner 2',key)
    assert participant_id('Fictional learner 1',key)!=participant_id('Fictional learner 1',b'B'*32)
    assert 'Fictional' not in participant_id('Fictional learner 1',key)
    with pytest.raises(InventoryValidationError):
        participant_id('name',b'short')

def test_duplicate_ids_rejected_without_explicit_deduplication():
    frame=synthetic_inventory()
    duplicated=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
    with pytest.raises(InventoryValidationError,match='Duplicate'):
        validate_inventory(duplicated)
    validated, audit=validate_inventory(duplicated,deduplicate_exact=True)
    assert len(validated)==17
    assert audit=={'input_records':18,'unique_messages':17,'exact_duplicates_removed':1}
    assert validated.iloc[0].source_record_numbers=='1,18'
    duplicated.loc[17,'message']='A conflicting fictional question?'
    with pytest.raises(InventoryValidationError,match='Conflicting'):
        validate_inventory(duplicated,deduplicate_exact=True)

def test_exact_duplicate_removal_reconciles_end_to_end(tmp_path,local_model):
    frame=synthetic_inventory()
    frame=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
    path=fixture_csv(tmp_path,'dedup',frame)
    manifest=process_inventory(path,mode='synthetic',run_id='dedup',project_root=tmp_path,
                               deduplicate_exact=True,_embedder=local_model)
    assert manifest['input_records']==manifest['unique_messages']+manifest['exact_duplicates_removed']
    assert manifest['detected_requests']==14

@pytest.mark.parametrize('column',['message_id','display_name','timestamp','message'])
def test_required_missing_values_fail_without_leaking_values(column):
    frame=synthetic_inventory()
    frame.loc[0,column]=''
    with pytest.raises(InventoryValidationError) as error:
        validate_inventory(frame)
    assert 'Fictional learner' not in str(error.value)
    assert 'How does an AI' not in str(error.value)

def test_missing_schema_and_unexpected_columns():
    with pytest.raises(InventoryValidationError):
        validate_inventory(synthetic_inventory().drop(columns='timestamp'))
    with pytest.raises(InventoryValidationError):
        validate_inventory(synthetic_inventory().assign(extra_private_field='never release'))

@pytest.mark.parametrize('timestamp',['15/01/2026 14:00','invalid PRIVATE_SENTINEL','2026-02-30T14:00:00Z'])
def test_invalid_timestamps_are_safely_reported(timestamp):
    frame=synthetic_inventory()
    frame.loc[0,'timestamp']=timestamp
    with pytest.raises(InventoryValidationError) as error:
        validate_inventory(frame)
    assert 'PRIVATE_SENTINEL' not in str(error.value)

def test_timezone_and_dst_validation():
    frame=synthetic_inventory()
    frame.loc[0,'timestamp']='2026-01-15T09:00:00'
    with pytest.raises(InventoryValidationError,match='Naive'):
        validate_inventory(frame)
    validated,_=validate_inventory(frame,assume_timezone='America/Toronto')
    assert validated.iloc[0].timestamp_utc=='2026-01-15T14:00:00+00:00'
    frame.loc[0,'timestamp']='2026-03-08T02:30:00'
    with pytest.raises(InventoryValidationError):
        validate_inventory(frame,assume_timezone='America/Toronto')

def test_missing_preliminary_labels_need_review_not_ground_truth():
    frame=synthetic_inventory()
    frame.loc[0,['topic','message_type','is_question','speaker_role']]=''
    validated,_=validate_inventory(frame)
    assert validated.iloc[0].metadata_review
    assert pd.isna(validated.iloc[0].preliminary_question)
    prepared=prepare_messages(validated,b'A'*32,'synthetic')
    assert set(prepared.columns)=={'message_id','participant_id','text','source_topic'}
    assert prepared.source_topic.eq('').all()  # Preliminary topic isn't fed into classifier/filter.
    assert 'is_question' not in prepared

def test_review_queue_includes_ambiguity_and_preliminary_disagreement(processed):
    private, _, manifest=processed
    queue=pd.read_csv(private/'identity_vault/review_queue.csv')
    assert len(queue)==manifest['review_records']
    assert queue.loc[queue.message_id=='SYN-001','review_reasons'].str.contains('Preliminary question').all()
    assert 'SYN-016' in set(queue.message_id)  # The fictional "help" message.
    assert queue.loc[queue.message_id=='SYN-016','review_reasons'].str.contains('Ambiguous').all()
    assert set(queue.review_status)=={'Pending human review'}
    assert {'message_id','display_name','message','cluster_id','participant_id'}<=set(queue.columns)

def test_embedding_question_detection_support_is_advisory(processed):
    private,_,_=processed
    queue=pd.read_csv(private/'identity_vault/review_queue.csv')
    assert {'question_support_similarity','question_support_margin','question_support_review'}<=set(queue.columns)
    questions=pd.read_csv(private/'analysis/classified_questions.csv')
    trace=pd.read_csv(private/'identity_vault/source_trace.csv')
    excluded=trace.loc[trace.message_id=='SYN-016','question_ref'].iloc[0]
    assert excluded not in set(questions.question_ref)  # Not silently promoted by model similarity.

def test_empty_inventory_still_uses_local_model_and_writes_empty_aggregates(tmp_path,local_model):
    path=fixture_csv(tmp_path,'empty',pd.DataFrame(columns=REQUIRED_COLUMNS))
    manifest=process_inventory(path,mode='synthetic',run_id='empty',project_root=tmp_path,_embedder=local_model)
    assert manifest['unique_messages']==0 and manifest['detected_requests']==0
    assert manifest['question_counts_by_cluster']==0 and manifest['review_records']==0
    candidate=tmp_path/'data/staging/synthetic/aggregate_candidates/empty'
    assert pd.read_csv(candidate/'topic_counts.csv').empty
    assert pd.read_csv(candidate/'recurring_themes.csv').empty

def test_missing_model_fails_even_for_empty_inventory(tmp_path):
    path=fixture_csv(tmp_path,'missing-model',pd.DataFrame(columns=REQUIRED_COLUMNS))
    with patch('src.local_processing.pd.read_csv',side_effect=AssertionError('Must fail before input read')):
        with pytest.raises(ModelUnavailableError):
            process_inventory(path,mode='synthetic',run_id='missing-model',project_root=tmp_path,model_dir=tmp_path/'no-model')

def test_real_mode_without_approval_cannot_read_any_input(tmp_path):
    # No real fixture is created or processed: exercise the pre-read rejection only.
    with patch('src.local_processing.pd.read_csv',side_effect=AssertionError('No real CSV read')):
        with pytest.raises(InventoryValidationError,match='approval'):
            process_inventory(tmp_path/'data/private/real/unused.csv',mode='real',run_id='blocked',project_root=tmp_path)
    assert not (tmp_path/'data').exists()

def test_cli_real_approval_gate_and_synthetic_input_rejection():
    spec=importlib.util.spec_from_file_location('inventory_cli',Path(__file__).parent/'scripts/process_inventory.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch.object(module,'process_inventory',side_effect=AssertionError('No processing permitted')):
        with pytest.raises(SystemExit):
            module.main(['--mode','real','--run-id','blocked','--input','DO_NOT_READ.csv'])
        with pytest.raises(SystemExit):
            module.main(['--mode','synthetic','--run-id','blocked','--input','DO_NOT_READ.csv'])

def test_path_boundaries_mode_separation_and_no_overwrites(tmp_path):
    with pytest.raises(InventoryValidationError):
        restricted_path(tmp_path/'public'/'raw.csv',tmp_path)
    with pytest.raises(InventoryValidationError):
        run_paths(tmp_path,'synthetic','../../escape')
    private,candidate=run_paths(tmp_path,'synthetic','same')
    real_private,real_candidate=run_paths(tmp_path,'real','same')
    assert private!=real_private and candidate!=real_candidate
    private.mkdir(parents=True)
    with pytest.raises(InventoryValidationError,match='already exists'):
        run_paths(tmp_path,'synthetic','same')

def test_low_suppression_threshold_and_uncontrolled_labels_are_rejected(local_model):
    frame,_=validate_inventory(synthetic_inventory())
    prepared=prepare_messages(frame,b'A'*32,'synthetic')
    result=analyze_messages(prepared,local_model)
    with pytest.raises(InventoryValidationError):
        aggregate_candidates(result.questions,'synthetic',minimum_participants=2)
    result.questions.loc[0,'theme']='=PRIVATE_SENTINEL()'
    with pytest.raises(InventoryValidationError,match='controlled'):
        aggregate_candidates(result.questions,'synthetic')

def test_authorized_entry_does_not_weaken_dashboard_synthetic_gate():
    frame,_=validate_inventory(synthetic_inventory())
    prepared=prepare_messages(frame,b'A'*32,'synthetic')
    with pytest.raises(ValueError,match='authorization'):
        analyze_authorized_messages(prepared,None)
