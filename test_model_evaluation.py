"""Evaluation invariants and offline inference, entirely fictional."""
import socket
from unittest.mock import patch
import pandas as pd
import pytest
from src.mock_dataset import load_mock
from src.embedding_model import LocalEmbedder
from src.topic_classifier import TopicClassifier
from src.pipeline import analyze_messages
from src.model_evaluation import evaluate, binary_metrics, PAIRS

def test_metric_arithmetic():
    assert binary_metrics([True,True,False],[True,False,True]) == dict(precision=.5,recall=.5,f1=.5)

def test_offline_evaluation_and_label_independence():
    with patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden')), patch.object(socket,'create_connection',side_effect=AssertionError('Network forbidden')):
        _,messages,truth=load_mock()
        prepared=messages.rename(columns={'message':'text'}).drop(columns=['topic'])
        model=LocalEmbedder(); classifier=TopicClassifier(model)
        result=analyze_messages(prepared,model,classifier)
        scores=evaluate(result,truth,model,classifier)
        assert scores['evaluated_messages']==250
        assert scores['evaluated_questions']==176
        assert scores['detection']==dict(precision=1.,recall=1.,f1=1.)
        assert scores['confusion'].to_numpy().sum()==176
        assert len(scores['errors'])==5
        assert scores['clustering']['pairs']==len(PAIRS)==12
        assert scores['review_count']==int(result.questions.needs_review.sum())
        changed=truth.copy(); changed['topic']='Course Support'
        second=evaluate(result,changed,model,classifier)
        assert second['topic_accuracy']!=scores['topic_accuracy']
        # Poison fixture channels: these are filter metadata, never model targets.
        changed_messages=prepared.copy(); changed_messages['source_topic']='NOT A CLASS'
        other=analyze_messages(changed_messages,model,classifier)
        pd.testing.assert_frame_equal(result.questions[['topic','cluster_id']],other.questions[['topic','cluster_id']])
        result.questions.attrs.clear()
        with pytest.raises(ValueError): evaluate(result,truth,model,classifier)

def test_evaluation_section_only_in_ai_tab():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=60)
    assert not app.exception
    assert sum(s.value=='AI Model Evaluation' for s in app.subheader)==1
    assert any(s.value=='AI Model Evaluation' for s in app.tabs[2].subheader)
    assert not any(s.value=='AI Model Evaluation' for s in app.tabs[1].subheader)
