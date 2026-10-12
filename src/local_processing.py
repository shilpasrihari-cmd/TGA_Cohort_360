"""Restricted local processing; no public-dashboard integration or network setup."""
import hashlib
import json
import os
import re
import secrets
import uuid
from pathlib import Path
import pandas as pd
from .embedding_model import LocalEmbedder, MODEL_DIR
from .pipeline import analyze_messages, analyze_authorized_messages
from .question_detection_support import detection_support
from .restricted_inventory import REQUIRED_COLUMNS, InventoryValidationError, validate_inventory, prepare_messages
from .learning_analytics import summarize_learning, theme_statistics, ALLOWED_THEME_NAMES
from .topic_classifier import TOPICS
from .storage_security import checked_path, verify_mode_isolation, verify_no_overlap, secure_directory, secure_csv, restrict_permissions
from .public_release import release_request

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def restricted_path(path, root):
    return checked_path(path,root)

def run_paths(root, mode, run_id):
    if mode not in {'synthetic','real'} or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,59}', run_id):
        raise InventoryValidationError('Use a valid mode and a simple alphanumeric run ID (up to 60 characters).')
    verify_mode_isolation(root)
    root = Path(root).resolve()
    private = restricted_path(root / 'data' / 'private' / mode / 'runs' / run_id, root)
    candidate = restricted_path(root / 'data' / 'staging' / mode / 'aggregate_candidates' / run_id, root)
    if private.exists() or candidate.exists():
        raise InventoryValidationError('Run ID already exists; use a new ID to avoid overwriting results.')
    return private, candidate

def pseudonym_key(root, mode):
    location = restricted_path(Path(root) / 'data' / 'private' / mode / 'identity_vault' / 'pseudonym.key', root)
    secure_directory(location.parent,root)
    if not location.exists():
        with location.open('xb') as stream:
            stream.write(secrets.token_bytes(32))
    restrict_permissions(location)
    key = location.read_bytes()
    if len(key) != 32:
        raise InventoryValidationError('Stored pseudonym key is invalid; processing stopped.')
    return key

def synthetic_inventory():
    rows = []
    groups = [
        ('How does an AI agent select and call tools?', 'Agentic AI', 6),
        ('How can I improve document retrieval accuracy in my RAG system?', 'Models & Architecture', 6),
        ('Where can I access the course recording?', 'Course Support', 2),
    ]
    for text, topic, count in groups:
        for number in range(count):
            rows.append({'message_id':f'SYN-{len(rows)+1:03}', 'display_name':f'Fictional learner {number+1}',
                         'timestamp':'2026-01-15T14:00:00Z', 'message':text, 'message_type':'chat',
                         'topic':topic, 'is_question':'false' if not rows else 'true', 'speaker_role':'learner'})
    for text in ['Thanks!', 'help', 'I finished the exercise today.']:
        rows.append({'message_id':f'SYN-{len(rows)+1:03}', 'display_name':'Fictional learner 1',
                     'timestamp':'2026-01-15T14:05:00Z', 'message':text, 'message_type':'chat',
                     'topic':'Other / Unclear', 'is_question':'false', 'speaker_role':'learner'})
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

def aggregate_candidates(questions, mode, minimum_participants=5):
    """Controlled labels only; derive topic totals solely from released themes.

    No full totals or small suppressed counts are emitted, preventing simple
    subtraction of released themes from unsuppressed parent topic totals.
    These are local draft candidates, not approved dashboard publication.
    """
    if mode not in {'synthetic','real'} or minimum_participants < 5:
        raise InventoryValidationError('Candidate release requires a known mode and k >= 5.')
    stats = theme_statistics(questions)
    if not stats.topic.isin(TOPICS).all() or not stats.theme.isin(ALLOWED_THEME_NAMES).all():
        raise InventoryValidationError('Aggregate labels must come from the controlled vocabulary.')
    eligible = stats[(stats.question_count >= 2) & (stats.participant_count >= minimum_participants)]
    released = questions[questions.cluster_id.isin(eligible.cluster_id)]
    themes = eligible[['theme','topic','question_count','participant_count','needs_review']].copy()
    topics = released.groupby('topic').agg(question_count=('message_id','size'),
                                         participant_count=('participant_id','nunique')).reset_index()
    for frame in [themes, topics]:
        frame.insert(0,'data_mode',mode)
        frame.insert(1,'release_status','Draft - human review required')
    return themes, topics

def review_records(inventory, detections, questions, support):
    restricted = inventory.merge(detections[['message_id','participant_id','detection_kind','is_question','detection_review','detection_reason']],
                                 on='message_id', validate='one_to_one', suffixes=('_preliminary',''))
    support = support.copy()
    support['message_id'] = detections.message_id.to_numpy()
    restricted = restricted.merge(support, on='message_id', validate='one_to_one')
    predictions = questions[['message_id','topic','topic_similarity','topic_margin','topic_review','cluster_id','theme','cluster_review','cluster_min_similarity','needs_review']].rename(columns={'topic':'predicted_topic'})
    restricted = restricted.merge(predictions, on='message_id', how='left', validate='one_to_one')
    reasons = []
    for row in restricted.itertuples():
        flags = []
        if row.detection_review:
            flags.append('Ambiguous rule detection')
        if bool(row.question_support_review):
            flags.append('Embedding support suggests a potentially missed request; review only')
        if row.metadata_review:
            flags.append('Missing preliminary metadata')
        if pd.notna(row.preliminary_question) and bool(row.preliminary_question) != bool(row.is_question):
            flags.append('Preliminary question label disagrees; neither is validated ground truth')
        if pd.notna(row.predicted_topic) and row.topic and row.topic != row.predicted_topic:
            flags.append('Preliminary topic differs from model classification')
        if pd.notna(row.needs_review) and row.needs_review:
            flags.append('Uncertain topic, cluster, or potentially truncated message')
        reasons.append('; '.join(flags))
    restricted['review_reasons'] = reasons
    restricted['review_status'] = 'Pending human review'
    return restricted[restricted.review_reasons.ne('')].copy()

def process_inventory(input_path, *, mode, run_id, project_root=PROJECT_ROOT, authorized=False,
                      assume_timezone=None, deduplicate_exact=False, minimum_participants=5,
                      model_dir=MODEL_DIR, _embedder=None):
    # This must precede reading, stat-ing, or loading the restricted input.
    if mode == 'real' and not authorized:
        raise InventoryValidationError('Real CSV processing requires explicit approval; input was not read.')
    if mode == 'synthetic' and authorized:
        raise InventoryValidationError('Do not supply real-processing authorization in synthetic mode.')
    if minimum_participants < 5:
        raise InventoryValidationError('Suppression threshold must be at least five.')
    private, candidate = run_paths(project_root, mode, run_id)
    path = restricted_path(input_path, project_root)
    if mode == 'synthetic':
        expected = (Path(project_root) / 'data' / 'staging' / 'synthetic' / 'fixtures' / f'{run_id}.csv').resolve()
        if path != expected:
            raise InventoryValidationError('Synthetic mode only reads its generated fixture path.')
    elif not any(path.is_relative_to((Path(project_root)/'data'/folder/'real').resolve()) for folder in ['private','staging']):
        raise InventoryValidationError('Real inputs must stay in the separate data/private/real/ or data/staging/real/ namespace.')
    verify_no_overlap(path,[private,candidate])
    # Real and empty runs must fail if the local model is absent, before data ingestion.
    embedder = _embedder if _embedder is not None else LocalEmbedder(model_dir)
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except Exception:
        raise InventoryValidationError('CSV could not be read as a valid UTF-8 inventory; no source values are logged.') from None
    inventory, reconciliation = validate_inventory(frame, assume_timezone=assume_timezone, deduplicate_exact=deduplicate_exact)
    prepared = prepare_messages(inventory, pseudonym_key(project_root, mode), mode)
    result = (analyze_messages(prepared, embedder) if mode == 'synthetic' else
              analyze_authorized_messages(prepared, embedder, authorized=authorized))
    support = detection_support(result.detections, embedder)
    review = review_records(inventory, result.detections, result.questions, support)
    stats = summarize_learning(result.questions)
    themes, topics = aggregate_candidates(result.questions, mode, minimum_participants)
    secure_directory(private,project_root)
    identity = secure_directory(private/'identity_vault',project_root)
    analysis = secure_directory(private/'analysis',project_root)
    references = {value:uuid.uuid4().hex for value in inventory.message_id}
    trace = pd.DataFrame({'question_ref':inventory.message_id.map(references),
                          'message_id':inventory.message_id,'source_record_numbers':inventory.source_record_numbers})
    identities = pd.DataFrame({'participant_id':prepared.participant_id,'display_name':inventory.display_name}).drop_duplicates()
    # Raw inventory, identities and source traces are confined to the vault.
    secure_csv(inventory,identity/'validated_inventory.csv',project_root)
    secure_csv(identities,identity/'identity_mapping.csv',project_root)
    secure_csv(trace,identity/'source_trace.csv',project_root)
    secure_csv(review,identity/'review_queue.csv',project_root)
    secure_csv(result.questions[['message_id','participant_id','cluster_id']],identity/'cluster_membership.csv',project_root)
    classified = result.questions[['participant_id','topic','theme','cluster_id','category','topic_similarity','topic_margin','topic_review','cluster_review','cluster_min_similarity','needs_review']].copy()
    classified.insert(0,'question_ref',result.questions.message_id.map(references))
    secure_csv(classified,analysis/'classified_questions.csv',project_root)
    detected = result.detections[['participant_id','detection_kind','is_question','detection_review']].copy()
    detected.insert(0,'question_ref',result.detections.message_id.map(references))
    secure_csv(detected,analysis/'detected_messages.csv',project_root)
    secure_csv(stats['by_topic'],analysis/'topic_counts.csv',project_root)
    secure_csv(stats['recurring'],analysis/'recurring_themes.csv',project_root)
    reconciliation.update({'detected_requests':len(result.questions), 'nonrequests':len(result.detections)-len(result.questions),
                           'question_counts_by_topic':int(stats['by_topic'].question_count.sum()),
                           'question_counts_by_cluster':int(stats['themes'].question_count.sum()),
                           'review_records':len(review), 'data_mode':mode,
                           'preliminary_labels_used_as_ground_truth':False,
                           'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'question_detection':'Rules with advisory local embedding support',
                           'model':'all-MiniLM-L6-v2 local CPU', 'k':minimum_participants})
    (private / 'run_manifest.json').write_text(json.dumps(reconciliation, indent=2), encoding='utf-8')
    restrict_permissions(private/'run_manifest.json')
    secure_directory(candidate,project_root)
    # Drafts remain restricted; they are not public exports or dashboard inputs.
    secure_csv(themes,candidate/'recurring_themes.csv',project_root)
    secure_csv(topics,candidate/'topic_counts.csv',project_root)
    (candidate/'approval-request.json').write_text(json.dumps(release_request(classified,mode),indent=2),encoding='utf-8')
    restrict_permissions(candidate/'approval-request.json')
    return reconciliation
