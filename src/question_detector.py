"""Explainable English rules; this stage does not use an AI model."""
import re
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Detection:
    kind: str
    is_question: bool
    needs_review: bool
    reason: str

SUPPORT = re.compile(r"\b(can(?:not|'t)|unable to|having trouble|does(?:n't| not) work|not working|help me (?:fix|install|access)|module(?:notfound)?error|modulenotfounderror|permission denied|login failed|missing (?:recording|link))\b", re.I)
CLARIFY = re.compile(r'\b(?:please (?:explain|clarify)|can (?:you|someone) (?:explain|clarify)|could (?:you|someone) (?:explain|clarify)|what do you mean|i (?:do not|don.t) understand|need clarification)\b', re.I)
ASK = re.compile(r'^(?:(?:hi|hello|hey)[,! ]+)?(?:how|why|what|when|where|which|who|can|could|would|should|is|are|do|does|will)\b', re.I)
REACTION = re.compile(r'^(?:thanks(?: you)?|thank you|great|awesome|nice|agreed|ok(?:ay)?|got it|yes|no|wow)[!.\s]*$', re.I)

def detect_question(text):
    if not isinstance(text, str) or not text.strip():
        return Detection('ambiguous', False, True, 'Empty or missing message.')
    message = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[A-Za-z]', message):
        return Detection('ambiguous', False, True, 'Reaction or punctuation without enough language.')
    if SUPPORT.search(message):
        return Detection('support_request', True, False, 'Operational difficulty or explicit troubleshooting request.')
    if CLARIFY.search(message):
        return Detection('clarification', True, False, 'Explicit request for an explanation or clarification.')
    if REACTION.fullmatch(message):
        return Detection('reaction', False, False, 'Short acknowledgement or reaction.')
    if '?' in message and len(message.split()) >= 3:
        return Detection('question', True, False, 'Question mark in a substantive message.')
    if ASK.search(message) and len(message.split()) >= 4:
        return Detection('question', True, False, 'Interrogative opening with enough context.')
    if '?' in message or len(message.split()) < 4 or re.search(r'\b(?:help|confused|stuck)\b', message, re.I):
        return Detection('ambiguous', False, True, 'Possible request with insufficient context; human review needed.')
    return Detection('statement', False, False, 'No clear question or request cue.')

def detect_messages(messages):
    """One detection per message; multiple questions in a message count once."""
    if 'text' not in messages:
        raise ValueError('Messages require a text column.')
    result = messages.copy()
    detections = [detect_question(text) for text in result.text]
    result['detection_kind'] = pd.Series([d.kind for d in detections], index=result.index, dtype='object')
    result['is_question'] = pd.Series([d.is_question for d in detections], index=result.index, dtype='bool')
    result['detection_review'] = pd.Series([d.needs_review for d in detections], index=result.index, dtype='bool')
    result['detection_reason'] = pd.Series([d.reason for d in detections], index=result.index, dtype='object')
    return result
