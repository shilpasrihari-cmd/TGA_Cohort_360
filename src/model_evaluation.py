"""Synthetic-only evaluation; labels never enter the prediction pipeline."""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from .topic_classifier import TOPICS
from .pipeline import analyze_messages

# Independently authored before inference; not copied from classifier references.
# Each binary label means 'same question intent', not merely the same broad topic.
PAIRS = [
 ('Can a planner ask a calculator to do arithmetic?', 'How would an agent invoke a calculator function?', True),
 ('How can two agents divide a job?', 'What lets several agents share the work on one task?', True),
 ('Can I enforce a JSON schema on a response?', 'How do I make the output conform to a JSON schema?', True),
 ('How long should passages in my retrieval index be?', 'What length should each indexed text segment have?', True),
 ('How do I block malicious instructions in retrieved pages?', 'How can retrieved content be prevented from overriding system instructions?', True),
 ('Where can I replay the lesson?', 'How can I watch the class again?', True),
 ('How do I install numpy in a virtual environment?', 'How do I keep an API secret out of Git?', False),
 ('How does a planner invoke tools?', 'How do several agents coordinate their work?', False),
 ('How can I format instructions clearly?', 'How can I require valid JSON from the model?', False),
 ('How do I choose passage length for retrieval?', 'How do I improve retrieval ranking?', False),
 ('How do I prevent prompt injection?', 'How do I remove personal information before inference?', False),
 ('Where is the lesson replay?', 'Where is the invitation to the next class?', False),
]

def binary_metrics(actual, predicted):
    p,r,f,_ = precision_recall_fscore_support(actual,predicted,average='binary',zero_division=0)
    return dict(precision=float(p),recall=float(r),f1=float(f))

def evaluate(result, truth, embedder, classifier):
    if result.questions.attrs.get('synthetic') is not True or not truth.data_status.eq('SYNTHETIC DEMO DATA').all():
        raise ValueError('Evaluation accepts synthetic data only.')
    detection = truth.merge(result.detections[['message_id','is_question']],on='message_id',suffixes=('_truth','_prediction'),validate='one_to_one')
    if len(detection)!=len(truth): raise ValueError('Incomplete evaluation predictions.')
    topics = truth[truth.is_question].merge(result.questions[['message_id','topic','text','needs_review']],on='message_id',suffixes=('_truth','_prediction'),how='left',validate='one_to_one')
    predicted=topics.topic_prediction.fillna('Other / Unclear')
    _,_,macro,_=precision_recall_fscore_support(topics.topic_truth,predicted,labels=TOPICS[:-1],average='macro',zero_division=0)
    # Run the unchanged pipeline on fresh pair texts; labels remain outside it.
    texts=list(dict.fromkeys(text for left,right,_ in PAIRS for text in (left,right)))
    messages=pd.DataFrame({'message_id':[f'EVAL-{i:03}' for i in range(len(texts))], 'participant_id':['FICTIONAL-EVAL']*len(texts),'text':texts})
    messages.attrs['synthetic']=True
    paired=analyze_messages(messages,embedder,classifier).questions.set_index('text')
    pair_predictions=[left in paired.index and right in paired.index and paired.loc[left,'cluster_id']==paired.loc[right,'cluster_id'] for left,right,_ in PAIRS]
    pair_labels=[label for _,_,label in PAIRS]
    matrix=pd.DataFrame(confusion_matrix(topics.topic_truth,predicted,labels=TOPICS),index=TOPICS,columns=TOPICS)
    errors=topics[topics.topic_truth!=predicted][['text','topic_truth','topic_prediction','needs_review']].fillna({'topic_prediction':'Other / Unclear'})
    return dict(model=result.model_name,evaluated_messages=len(detection),evaluated_questions=len(topics),
                detection=binary_metrics(detection.is_question_truth,detection.is_question_prediction),
                topic_accuracy=float(accuracy_score(topics.topic_truth,predicted)),topic_macro_f1=float(macro),
                clustering={**binary_metrics(pair_labels,pair_predictions),'accuracy':float(accuracy_score(pair_labels,pair_predictions)),'pairs':len(PAIRS)},
                review_count=int(result.questions.needs_review.sum()),confusion=matrix,errors=errors,
                pair_results=pd.DataFrame([dict(left=l,right=r,expected_same=label,predicted_same=pred) for (l,r,label),pred in zip(PAIRS,pair_predictions)]))
