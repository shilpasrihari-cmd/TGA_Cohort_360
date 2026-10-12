"""Week 1 fixed-release policy: no public queries on restricted records."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from .storage_security import checked_path, verify_mode_isolation, secure_directory, restrict_permissions
from .restricted_inventory import InventoryValidationError
from .learning_analytics import theme_statistics, ALLOWED_THEME_NAMES
from .topic_classifier import TOPICS

POLICY_FILE = Path(__file__).resolve().parents[1]/'policies/week1-release.json'

def canonical_json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()

def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()

def policy():
    return json.loads(POLICY_FILE.read_text(encoding='utf-8'))

def release_request(questions, mode):
    """Private review request, never an approval or public artifact."""
    if mode not in {'synthetic','real'}:
        raise InventoryValidationError('Invalid release mode.')
    columns = ['participant_id','cluster_id','theme','topic','needs_review']
    if not set(columns)<=set(questions):
        raise InventoryValidationError('Restricted release input has invalid schema.')
    if 'question_ref' in questions:
        columns = ['question_ref',*columns]
    stats = theme_statistics(questions)
    if not stats.theme.isin(ALLOWED_THEME_NAMES).all() or not stats.topic.isin(TOPICS).all():
        raise InventoryValidationError('Release labels must use the controlled vocabulary.')
    # Fingerprint binds row membership and counts, not only displayed aggregates.
    fingerprint = digest(sorted([canonical_json(row).decode() for row in questions[columns].to_dict('records')]))
    return {'mode':mode,'policy_sha256':digest(policy()),'dataset_sha256':fingerprint,
            'approved':False,'identity_scope_verified':False,'privacy_review_completed':False,
            'review_reference':''}

def _tables(questions):
    # Collapse same-vocabulary clusters before release: never expose overlapping
    # six-person/five-person rows under the same theme label within one snapshot.
    stats = questions.groupby(['theme','topic'],sort=True).agg(
        question_count=('participant_id','size'),participant_count=('participant_id','nunique'),
        needs_review=('needs_review','any')).reset_index()
    # Suppress even the presence of a minority review flag within an eligible group.
    eligible = stats[(stats.participant_count>=policy()['minimum_participants']) & (~stats.needs_review)]
    themes = eligible.sort_values(['question_count','theme'],ascending=[False,True])[['theme','topic','question_count','participant_count']].to_dict('records')
    # Do not release parent distinct-person totals: these expose intersections.
    topic_frame = eligible.groupby('topic').question_count.sum().reset_index()
    return {'themes':themes,'topics':topic_frame.to_dict('records')}

def publish_fixed_snapshot(root, mode, questions, approval):
    request = release_request(questions,mode)
    required = ['approved','identity_scope_verified','privacy_review_completed']
    if any(approval.get(key) is not True for key in required) or not approval.get('review_reference'):
        raise InventoryValidationError('Explicit privacy and identity review approval is required before public generation.')
    if any(approval.get(key)!=request[key] for key in ['mode','policy_sha256','dataset_sha256']):
        raise InventoryValidationError('Release approval does not cover this exact dataset and policy.')
    verify_mode_isolation(root)
    control = checked_path(Path(root)/'data/private'/mode/'release_control',root)
    destination = checked_path(Path(root)/'data/public'/mode/'snapshot.json',root,('public',))
    slot = control/'release-slot.json'
    if slot.exists() or destination.exists():
        raise InventoryValidationError('A release is already reserved or published; overlapping/repeated releases are forbidden.')
    payload = {'snapshot_version':policy()['version'],'data_mode':mode,'policy':policy(),
               'tables':_tables(questions)}
    data = canonical_json(payload)
    secure_directory(control,root)
    # Exclusive reservation consumes this mode's release slot even if a crash occurs.
    # It must not be automatically cleared or moved to obtain another export.
    with slot.open('x',encoding='utf-8') as stream:
        json.dump({'approval':approval,'snapshot_sha256':hashlib.sha256(data).hexdigest()},stream)
    restrict_permissions(slot)
    secure_directory(destination.parent,root,('public',))
    with destination.open('xb') as stream:
        stream.write(data)
    restrict_permissions(destination)
    return destination

def export_snapshot_view(root,mode,view='all',geography='global'):
    """Read sealed aggregates only; takes no raw data, participant filters or dates."""
    if mode not in {'synthetic','real'}:
        raise InventoryValidationError('Invalid snapshot mode.')
    verify_mode_isolation(root)
    path = checked_path(Path(root)/'data/public'/mode/'snapshot.json',root,('public',))
    slot = checked_path(Path(root)/'data/private'/mode/'release_control/release-slot.json',root)
    try:
        data = path.read_bytes()
        seal = json.loads(slot.read_text(encoding='utf-8'))
        payload = json.loads(data)
    except Exception:
        raise InventoryValidationError('Approved sealed snapshot is unavailable.') from None
    if hashlib.sha256(data).hexdigest()!=seal.get('snapshot_sha256') or digest(payload.get('policy'))!=seal['approval']['policy_sha256']:
        raise InventoryValidationError('Snapshot integrity verification failed.')
    approved = payload['policy']
    if view not in approved['views'] or geography not in approved['geographies']:
        raise InventoryValidationError('Requested export is not covered by the approved fixed policy.')
    topics = approved['views'][view]
    return {name:[row for row in rows if not topics or row['topic'] in topics]
            for name,rows in payload['tables'].items()}
