"""Evaluate only the fixed fictional fixture, with network connections blocked."""
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mock_dataset import load_mock
from src.embedding_model import LocalEmbedder
from src.topic_classifier import TopicClassifier
from src.pipeline import analyze_messages
from src.model_evaluation import evaluate

def main():
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('Evaluation network disabled')), patch.object(socket,'create_connection',side_effect=RuntimeError('Evaluation network disabled')):
        _,messages,truth=load_mock()
        model=LocalEmbedder(); classifier=TopicClassifier(model)
        result=analyze_messages(messages.rename(columns={'message':'text'}).drop(columns=['topic']),model,classifier)
        scores=evaluate(result,truth,model,classifier)
    report=ROOT/'reports'/'synthetic_model_evaluation.json'
    report.parent.mkdir(exist_ok=True)
    serial={key:(value.to_dict(orient='split') if hasattr(value,'to_dict') else value) for key,value in scores.items()}
    report.write_text(json.dumps(serial,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in serial.items() if k not in ['confusion','errors','pair_results']},indent=2))
    print(scores['errors'].to_string(index=False))

if __name__=='__main__': main()
