"""Orchestrate local detection, embedding classification and clustering."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .question_detector import detect_messages
from .topic_classifier import TopicClassifier
from .question_clusterer import QuestionClusterer

@dataclass
class IntelligenceResult:
    detections: pd.DataFrame
    questions: pd.DataFrame
    model_name: str = 'all-MiniLM-L6-v2 (local CPU)'

def analyze_messages(messages, embedder, classifier=None):
    """Milestone safety gate: only explicitly synthetic fixtures accepted.

    This provenance flag is a guardrail, not automatic proof of anonymization.
    No file loaders, upload controls, or private datasets are connected.
    """
    if messages.attrs.get('synthetic') is not True:
        raise ValueError('This milestone only accepts explicitly marked synthetic messages.')
    return _analyze_prepared_messages(messages, embedder, classifier)

def analyze_authorized_messages(messages, embedder, classifier=None, *, authorized=False):
    """Restricted CLI entry; the dashboard's synthetic-only entry remains unchanged."""
    if not authorized or messages.attrs.get('data_mode') != 'real':
        raise ValueError('Restricted analysis requires explicit processing authorization.')
    return _analyze_prepared_messages(messages, embedder, classifier)

def _analyze_prepared_messages(messages, embedder, classifier=None):
    """Shared inference; inputs and outputs remain internal to their caller."""
    required = {'message_id', 'participant_id', 'text'}
    if not required <= set(messages.columns):
        raise ValueError('Required columns: message_id, participant_id, text.')
    if messages.message_id.isna().any() or messages.message_id.duplicated().any():
        raise ValueError('Message IDs must be nonmissing and unique.')
    if messages.participant_id.isna().any() or not messages.participant_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Each synthetic message requires a nonempty pseudonymous participant ID.')
    if len(messages) > 5000:
        raise ValueError('This local prototype supports at most 5,000 messages per batch.')
    if 'has_identifiable_answer' in messages and not messages.has_identifiable_answer.map(
            lambda value: pd.isna(value) or isinstance(value, (bool, np.bool_))).all():
        raise ValueError('Answer flags must be nullable booleans, not inferred strings.')
    detected = detect_messages(messages).reset_index(drop=True)
    questions = detected[detected.is_question].copy().reset_index(drop=True)
    if 'has_identifiable_answer' not in questions:
        questions['has_identifiable_answer'] = pd.Series([None]*len(questions), dtype='object')
    if 'source_topic' not in questions:
        questions['source_topic'] = ''
    if questions.empty:
        for column in ['topic', 'topic_similarity', 'topic_margin', 'topic_review', 'cluster_id',
                       'cluster_review', 'cluster_min_similarity', 'theme', 'category', 'needs_review']:
            questions[column] = pd.Series(dtype='bool' if column.endswith('review') else 'object')
        questions.attrs.update(messages.attrs)
        return IntelligenceResult(detected, questions)
    classifier = classifier or TopicClassifier(embedder)
    vectors = embedder.encode(questions.text.tolist())
    predictions = classifier.classify(questions.text.tolist(), embeddings=vectors)
    questions = pd.concat([questions, predictions], axis=1)
    groups = QuestionClusterer().cluster(questions, vectors)
    questions = pd.concat([questions, groups], axis=1)
    questions['theme'] = ''
    for _, group in questions.groupby('cluster_id'):
        centroid = vectors[group.index].mean(axis=0)
        name = classifier.name_theme(group.topic.iloc[0], centroid)
        questions.loc[group.index, 'theme'] = name
    questions['category'] = ['Support' if kind == 'support_request' or topic == 'Course Support' else 'Learning'
                             for kind, topic in zip(questions.detection_kind, questions.topic)]
    questions['needs_review'] = questions.detection_review | questions.topic_review | questions.cluster_review | (questions.text.str.split().str.len() > 120)
    questions.attrs.update(messages.attrs)
    return IntelligenceResult(detected, questions)

def filter_intelligence(questions, participant_ids, source_topics=(), classified_topics=()):
    selected = questions[questions.participant_id.isin(participant_ids)].copy()
    if source_topics:
        selected = selected[selected.source_topic.isin(source_topics)]
    if classified_topics:
        selected = selected[selected.topic.isin(classified_topics)]
    return selected.reset_index(drop=True)
