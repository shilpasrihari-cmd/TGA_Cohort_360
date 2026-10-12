"""Generate fictional chat text from existing demo rows, never read chat files."""
import pandas as pd

PARAPHRASES = {
    'Tool calling': ['How does an AI agent choose which tool to call?', 'Can you explain how an agent selects and calls tools?'],
    'Agent orchestration': ['How can I coordinate multiple AI agents in one workflow?', 'What is a good way to orchestrate several AI agents together?'],
    'Agent evaluation': ['How do I evaluate whether an AI agent completes tasks correctly?', 'How can I test if my AI agent successfully finishes a task?'],
    'Retrieval quality': ['How can I improve document retrieval accuracy in my RAG system?', 'What can improve the accuracy of retrieved documents in RAG?'],
    'Chunking strategies': ['What chunk size should I use when splitting documents for RAG?', 'How should I choose document chunk sizes for a RAG pipeline?'],
    'Vector databases': ['How do I store and search embeddings in a vector database?', 'Can you explain how vector databases store and retrieve embeddings?'],
    'Prompt structure': ['How should I structure instructions and examples in my prompt?', 'What is a good prompt structure for clear instructions and examples?'],
    'Output consistency': ['How do I make a language model return consistent JSON output?', 'How can I get reliable structured JSON responses from my language model?'],
    'Evaluation datasets': ['How do I build an evaluation dataset for my AI application?', 'What examples should go into an evaluation dataset for an AI project?'],
    'Quality metrics': ['What quality metrics should I use to measure my AI application?', 'How can I measure the quality of my AI application using metrics?'],
    'Production monitoring': ['How do I monitor a deployed AI application in production?', 'What should I monitor after deploying my AI application to production?'],
    'Deployment patterns': ['How can I deploy my AI application as a production service?', 'What deployment pattern should I choose for an AI application?'],
    'Recording access': ['Where can I access the recording of our course session?', 'I cannot access the course recording; can someone help?'],
    'Session links': ['Where is the meeting link for the next class session?', 'I cannot find the link to join the next course session.'],
    'Package installation': ['How do I install Python packages in my virtual environment?', 'I cannot install a Python package in my virtual environment; please help.'],
    'API key setup': ['How do I configure an API key in my local development environment?', 'My API key is not working in my local Python project; can you help?'],
}
EXTRA_QUESTIONS = [
    ('What is machine learning and how does it learn from data?', 'AI agents'),
    ('Can you explain the difference between training and inference in machine learning?', 'AI agents'),
    ('How can I protect an AI agent against prompt injection?', 'AI agents'),
    ('How do I prevent private information leaking from an AI application?', 'AI agents'),
    ('How does the attention mechanism in a transformer work?', 'RAG'),
    ('What do you mean by that?', 'AI agents'),
]

def synthetic_messages(participants, fixture_questions):
    required = {'theme', 'question_id', 'participant_id', 'topic', 'has_identifiable_answer'}
    if not required <= set(fixture_questions.columns):
        raise ValueError('Legacy synthetic generation requires themed question fixtures; use prepare_synthetic_messages for chat rows.')
    rows = []
    for i, question in enumerate(fixture_questions.itertuples()):
        alternatives = PARAPHRASES[question.theme]
        rows.append({'message_id': question.question_id, 'participant_id': question.participant_id,
                     'text': alternatives[i % len(alternatives)], 'source_topic': question.topic,
                     'has_identifiable_answer': question.has_identifiable_answer})
    ids = participants.participant_id.tolist()
    if ids:
        for i, (text, scope) in enumerate(EXTRA_QUESTIONS):
            for repeat in range(4):
                rows.append({'message_id': f'EX{i:02}-{repeat}', 'participant_id': ids[(i*4+repeat) % len(ids)],
                             'text': text, 'source_topic': scope, 'has_identifiable_answer': True})
        for i, text in enumerate(['Thanks!', 'Great session.', 'I finished the exercise today.', 'help', '???', '', None, 'This is a statement about my project.']):
            rows.append({'message_id': f'NO{i:02}', 'participant_id': ids[i % len(ids)], 'text': text,
                         'source_topic': '', 'has_identifiable_answer': None})
    result = pd.DataFrame(rows, columns=['message_id', 'participant_id', 'text', 'source_topic', 'has_identifiable_answer'])
    result.attrs['synthetic'] = True
    return result

def prepare_synthetic_messages(participants, frame):
    """Adapt fictional chat rows directly; generate only explicit legacy fixtures."""
    if 'message' in frame.columns or 'text' in frame.columns:
        if frame.attrs.get('synthetic') is not True:
            raise ValueError('Only explicitly synthetic chat rows are supported.')
        result = frame.rename(columns={'message':'text'}) if 'message' in frame.columns else frame.copy()
        return result.drop(columns=['topic'], errors='ignore')
    return synthetic_messages(participants, frame)
