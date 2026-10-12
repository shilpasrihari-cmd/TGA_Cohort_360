"""Embedding-based advisory review signals, never validated question labels."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

REFERENCES = {
    'question_or_request': ['How does an AI agent call tools?', 'Could you clarify how document retrieval works?',
                            'Can someone help me fix this Python error?', 'Where can I find the course recording?'],
    'statement_or_reaction': ['I finished the project exercise today.', 'Thanks, that explanation was helpful.',
                              'The course meeting is scheduled for tomorrow.', 'Here is a summary of our discussion.'],
}

def detection_support(messages, embedder):
    result = pd.DataFrame(index=messages.index, columns=['question_support_similarity','question_support_margin','question_support_review'])
    if messages.empty:
        return result
    vectors = embedder.encode(messages.text.tolist())
    references = embedder.encode([text for values in REFERENCES.values() for text in values])
    similarities = cosine_similarity(vectors, references)
    question_score = similarities[:,:4].max(axis=1)
    other_score = similarities[:,4:].max(axis=1)
    margin = question_score-other_score
    result['question_support_similarity'] = question_score
    result['question_support_margin'] = margin
    # Only propose review of possibly missed requests; no automatic promotion.
    result['question_support_review'] = (~messages.is_question.to_numpy()) & (question_score >= .45) & (margin >= .05)
    return result
