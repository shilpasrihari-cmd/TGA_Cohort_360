"""Real local model tests plus explicit rule/aggregation and privacy tests."""
import io
import os
import socket
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity
from src.embedding_model import LocalEmbedder, ModelUnavailableError
from src.question_detector import detect_question, detect_messages
from src.topic_classifier import TopicClassifier
from src.question_clusterer import QuestionClusterer
from src.learning_analytics import summarize_learning, privacy_safe_export
from src.pipeline import analyze_messages, filter_intelligence
from src.synthetic_chat import synthetic_messages
from data import synthetic_data

@pytest.fixture(scope='module')
def embedder():
    # A real model must load and infer with all outbound sockets blocked.
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden during local inference')), \
         patch.object(socket, 'create_connection', side_effect=AssertionError('Network forbidden during local inference')):
        model = LocalEmbedder()
        yield model

@pytest.fixture(scope='module')
def classifier(embedder):
    return TopicClassifier(embedder)

def test_file_backed_mock_inference_offline(embedder, classifier):
    from src.mock_dataset import load_mock
    people, messages, truth = load_mock()
    result = analyze_messages(messages.rename(columns={'message':'text'}).drop(columns=['topic']), embedder, classifier)
    assert len(result.questions) == int(truth.is_question.sum()) == 176
    assert set(result.questions.participant_id) <= set(people.participant_id)
    assert result.questions.cluster_id.notna().all()
    assert result.questions.groupby('cluster_id').size().max() >= 5
    assert 'recurring_question_theme' not in result.questions
    exported = pd.read_csv(io.BytesIO(privacy_safe_export(result.questions)))
    assert not {'participant_id','message_id','text','message'} & set(exported.columns)
    assert (exported.participant_count >= 3).all()

@pytest.fixture(scope='module')
def intelligence(embedder, classifier):
    people, questions = synthetic_data()
    return analyze_messages(synthetic_messages(people, questions), embedder, classifier)

@pytest.mark.parametrize('text,kind,is_question', [
    ('How does a transformer work?', 'question', True),
    ('How do I evaluate agents', 'question', True),
    ('Please clarify what embeddings mean', 'clarification', True),
    ('I do not understand how tool calling works', 'clarification', True),
    ('I cannot access the course recording', 'support_request', True),
    ('My Python package is not working', 'support_request', True),
    ('Thanks!', 'reaction', False),
    ('I finished the exercise today.', 'statement', False),
    ('help', 'ambiguous', False),
    ('???', 'ambiguous', False),
    ('Thanks?', 'ambiguous', False),
    ('', 'ambiguous', False),
    (None, 'ambiguous', False),
])
def test_question_detection(text, kind, is_question):
    result = detect_question(text)
    assert result.kind == kind
    assert result.is_question is is_question
    assert result.reason
    assert result.needs_review is (kind == 'ambiguous')

def test_local_embedding_and_semantic_similarity(embedder):
    texts = ['How can I improve document retrieval accuracy in my RAG system?',
             'What can improve the accuracy of retrieved documents in RAG?',
             'Where is the link for the next course meeting?']
    vectors = embedder.encode(texts)
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    similarity = cosine_similarity(vectors)
    assert similarity[0, 1] > .70
    assert similarity[0, 1] > similarity[0, 2] + .30
    assert os.environ['HF_HUB_OFFLINE'] == '1'
    assert os.environ['HF_HUB_DISABLE_TELEMETRY'] == '1'

@pytest.mark.parametrize('text,expected', [
    ('What is machine learning and how does it learn from data?', 'AI Foundations'),
    ('How do I coordinate multiple AI agents in a workflow?', 'Agentic AI'),
    ('How do I write a prompt with few shot examples?', 'Prompt Engineering'),
    ('How does the attention mechanism of a transformer work?', 'Models & Architecture'),
    ('How can I protect an AI agent against prompt injection?', 'AI Safety & Security'),
    ('How do I fix a Python package installation error in my virtual environment?', 'Development Tools'),
    ('How do I deploy my application and monitor it in production?', 'Project Implementation'),
    ('Where is the recording of the course session?', 'Course Support'),
    ('What do you mean by that?', 'Other / Unclear'),
])
def test_topic_classification_uses_real_model(classifier, text, expected):
    prediction = classifier.classify([text]).iloc[0]
    assert prediction.topic == expected
    assert -1 <= prediction.topic_similarity <= 1.00001
    if expected == 'Other / Unclear':
        assert prediction.topic_review

def test_clustering_semantic_paraphrases(embedder):
    texts = ['How can I improve document retrieval accuracy in my RAG system?',
             'What can improve the accuracy of retrieved documents in RAG?',
             'What is the attention mechanism in a transformer?']
    questions = pd.DataFrame({'message_id':['A','B','C'], 'topic':['Models & Architecture']*3})
    groups = QuestionClusterer().cluster(questions, embedder.encode(texts))
    assert groups.cluster_id[0] == groups.cluster_id[1]
    assert groups.cluster_id[2] != groups.cluster_id[0]

def test_complete_link_prevents_transitive_chaining():
    angles = np.radians([0, 35, 70])
    vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    questions = pd.DataFrame({'message_id':['A','B','C'], 'topic':['Agentic AI']*3})
    groups = QuestionClusterer().cluster(questions, vectors)
    assert groups.cluster_id.nunique() == 2  # A~B and B~C does not imply A~C.
    assert groups.cluster_id[0] != groups.cluster_id[2]

def test_borderline_cluster_is_flagged_for_review():
    vectors = np.array([[1., 0.], [.74, np.sqrt(1-.74**2)]])
    questions = pd.DataFrame({'message_id':['A','B'], 'topic':['Agentic AI']*2})
    groups = QuestionClusterer().cluster(questions, vectors)
    assert groups.cluster_id.nunique() == 1
    assert groups.cluster_review.all()
    assert groups.cluster_min_similarity.iloc[0] == pytest.approx(.74)

def test_cluster_ids_are_stable_for_row_reordering(embedder):
    questions = pd.DataFrame({'message_id':['A','B','C'], 'topic':['Agentic AI']*3})
    vectors = embedder.encode(['How can an agent call tools?', 'How can an agent call tools?', 'How do I coordinate several agents?'])
    a = QuestionClusterer().cluster(questions, vectors)
    order = [2,0,1]
    b = QuestionClusterer().cluster(questions.iloc[order].reset_index(drop=True), vectors[order])
    assert dict(zip(questions.message_id,a.cluster_id)) == dict(zip(questions.iloc[order].message_id,b.cluster_id))

def test_pipeline_aggregate_reconciliation(intelligence):
    questions = intelligence.questions
    stats = summarize_learning(questions)
    assert len(intelligence.detections) == 792
    assert len(questions) == 784
    assert stats['by_topic'].question_count.sum() == len(questions)
    assert stats['themes'].question_count.sum() == len(questions)
    assert stats['distribution'].question_count.sum() == len(questions)
    assert stats['metrics'][0] == 784
    assert questions.needs_review.any()
    assert (stats['followups'].question_count >= 2).all()
    assert (stats['followups'].participant_count >= 2).all()
    assert not stats['followups'].needs_review.any()
    assert not (stats['followups'].topic == 'Course Support').any()

def test_distinct_people_not_question_count():
    questions = pd.DataFrame({'cluster_id':['C1']*3, 'theme':['Tool calling']*3,
        'topic':['Agentic AI']*3, 'category':['Learning']*3,
        'participant_id':['P1','P1','P2'], 'needs_review':[False]*3,
        'has_identifiable_answer':[True,False,True]})
    stats = summarize_learning(questions)
    assert stats['themes'].iloc[0].question_count == 3
    assert stats['themes'].iloc[0].participant_count == 2
    assert stats['metrics'][3] == 1
    assert len(stats['followups']) == 1
    questions.attrs['synthetic'] = True
    export = pd.read_csv(io.BytesIO(privacy_safe_export(questions)))
    assert export.empty  # Three questions from only two people are suppressed.

def test_empty_input_without_model():
    empty = pd.DataFrame(columns=['message_id','participant_id','text'])
    empty.attrs['synthetic'] = True
    result = analyze_messages(empty, None)
    assert result.questions.empty
    assert detect_messages(empty).empty
    assert summarize_learning(result.questions)['metrics'] == [0,0,0,0]
    assert pd.read_csv(io.BytesIO(privacy_safe_export(result.questions))).empty
    assert QuestionClusterer().cluster(result.questions, []).empty

def test_no_questions_without_model():
    messages = pd.DataFrame({'message_id':['A'], 'participant_id':['P1'], 'text':['Thanks!']})
    messages.attrs['synthetic'] = True
    assert analyze_messages(messages, None).questions.empty

def test_empty_embeddings_and_classification(embedder, classifier):
    assert embedder.encode([]).shape == (0,384)
    assert classifier.classify([]).empty

def test_filter_counts_and_cluster_identity(intelligence):
    people, _ = synthetic_data()
    subset = filter_intelligence(intelligence.questions, people[people.country=='India'].participant_id,
                                source_topics=['RAG'], classified_topics=['Models & Architecture'])
    assert set(subset.source_topic) == {'RAG'}
    assert set(subset.topic) == {'Models & Architecture'}
    assert subset.cluster_id.isin(intelligence.questions.cluster_id).all()
    assert summarize_learning(subset)['themes'].question_count.sum() == len(subset)
    assert filter_intelligence(intelligence.questions, []).empty

def test_privacy_safe_export_allowlist_and_suppression(intelligence):
    exported = privacy_safe_export(intelligence.questions)
    frame = pd.read_csv(io.BytesIO(exported))
    assert set(frame.columns) == {'data_status','theme','topic','question_count','participant_count','needs_review'}
    assert (frame.data_status == 'Synthetic').all()
    assert (frame.participant_count >= 3).all()
    text = exported.decode()
    assert 'participant_id' not in text and 'message_id' not in text and 'detection_reason' not in text
    assert not any(message in text for message in intelligence.questions.text)
    assert not any(participant in text for participant in intelligence.questions.participant_id.unique())
    poisoned = intelligence.questions.copy()
    poisoned.loc[0,'theme'] = '=PRIVATE_MESSAGE()'
    with pytest.raises(ValueError, match='controlled'):
        privacy_safe_export(poisoned)
    with pytest.raises(ValueError):
        privacy_safe_export(intelligence.questions, minimum_participants=1)

def test_missing_model_fails_closed(tmp_path):
    with pytest.raises(ModelUnavailableError, match='missing'):
        LocalEmbedder(tmp_path / 'nonexistent-model')

def test_synthetic_gate_and_validation():
    messages = pd.DataFrame({'message_id':['A'], 'participant_id':['P1'], 'text':['How do tools work?']})
    with pytest.raises(ValueError, match='synthetic'):
        analyze_messages(messages, None)
    messages.attrs['synthetic'] = True
    duplicates = pd.concat([messages,messages])
    duplicates.attrs['synthetic'] = True
    with pytest.raises(ValueError, match='unique'):
        analyze_messages(duplicates, None)
    messages['has_identifiable_answer'] = ['false']
    with pytest.raises(ValueError, match='booleans'):
        analyze_messages(messages, None)

def test_missing_answer_flags_are_not_fabricated(embedder, classifier):
    messages = pd.DataFrame({'message_id':['A'], 'participant_id':['P1'], 'text':['How do I coordinate multiple AI agents?']})
    messages.attrs['synthetic'] = True
    result = analyze_messages(messages, embedder, classifier)
    assert summarize_learning(result.questions)['metrics'][3] is None
