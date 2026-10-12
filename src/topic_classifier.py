"""Local model embeddings plus deterministic nearest-description classification."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

TOPIC_DESCRIPTIONS = {
    'AI Foundations': [
        'Understand artificial intelligence, machine learning, training data, supervised learning and basic AI concepts.',
        'What is machine learning and how does artificial intelligence learn from data?',
    ],
    'Agentic AI': [
        'AI agents, tool calling, autonomous planning, multi-agent orchestration, agent evaluation and workflows.',
        'How can an AI agent call a tool and coordinate multiple agents?',
    ],
    'Prompt Engineering': [
        'Prompt engineering: write clear instructions, few-shot examples, structured outputs and reliable prompts.',
        'How should I structure a prompt to get consistent JSON output from a language model?',
    ],
    'Models & Architecture': [
        'Language model architecture, transformers, attention, embeddings, retrieval augmented generation, RAG, vector databases and chunking.',
        'How do transformer attention and embedding models work? How do I retrieve documents for RAG?',
    ],
    'AI Safety & Security': [
        'AI safety and security, prompt injection, privacy, responsible AI, bias, guardrails and sensitive data protection.',
        'How can I defend my AI agent against prompt injection and avoid leaking private information?',
    ],
    'Development Tools': [
        'Developer tools: Python packages, virtual environments, pip installation, IDE configuration, API keys and debugging errors.',
        'How do I fix a Python package installation error in my virtual environment?',
    ],
    'Project Implementation': [
        'Implement and deploy AI projects, applications, production monitoring, evaluation datasets, quality metrics, integration and testing.',
        'How can I deploy my application and monitor its performance in production?',
    ],
    'Course Support': [
        'Course support: access course recordings, session links, schedules, enrollment and learning platform login.',
        'Where is the recording of the course session and how do I access the class meeting link?',
    ],
}
TOPICS = [*TOPIC_DESCRIPTIONS, 'Other / Unclear']

# Controlled names, not snippets from messages. Used to name model-discovered groups.
THEME_DESCRIPTIONS = {
    'AI Foundations': {'Machine learning basics': 'What is machine learning and how does it learn from data?',
                       'Training and inference': 'What is the difference between training an AI model and using it for inference?'},
    'Agentic AI': {'Tool calling': 'How does an AI agent select and call external tools?',
                  'Multi-agent orchestration': 'How can I coordinate multiple AI agents in a workflow?',
                  'Agent evaluation': 'How do I evaluate whether an AI agent completes a task correctly?'},
    'Prompt Engineering': {'Prompt structure': 'How should I structure instructions and examples in a prompt?',
                           'Structured outputs': 'How do I get a language model to return consistent JSON output?'},
    'Models & Architecture': {'Retrieval quality': 'How do I improve document retrieval accuracy in RAG?',
                              'Document chunking': 'What chunk size should I choose for splitting documents in RAG?',
                              'Vector databases': 'How do I store and search embeddings in a vector database?',
                              'Transformer attention': 'How does the attention mechanism in a transformer work?'},
    'AI Safety & Security': {'Prompt injection defenses': 'How can I protect an AI application against prompt injection?',
                             'Privacy and data protection': 'How can I avoid leaking private data to an AI model?'},
    'Development Tools': {'Python environment setup': 'How do I install Python packages in a virtual environment?',
                          'API credential setup': 'How do I configure an API key locally without exposing it?'},
    'Project Implementation': {'Production monitoring': 'How do I monitor a deployed application in production?',
                               'Application deployment': 'How can I deploy my AI application?',
                               'Evaluation datasets': 'How do I build an evaluation dataset for my AI project?',
                               'Quality metrics': 'What quality metrics should I use to measure my application?'},
    'Course Support': {'Recording access': 'Where can I access the recording of a course session?',
                       'Session links': 'Where is the link for the next class meeting?'},
    'Other / Unclear': {'Unclear question theme': 'A question whose topic is not identifiable.'},
}

class TopicClassifier:
    def __init__(self, embedder, minimum_similarity=.30, review_margin=.035):
        self.embedder = embedder
        self.minimum_similarity = minimum_similarity
        self.review_margin = review_margin
        self.names = list(TOPIC_DESCRIPTIONS)
        texts = [text for descriptions in TOPIC_DESCRIPTIONS.values() for text in descriptions]
        self.reference_vectors = embedder.encode(texts)
        self.reference_topics = [topic for topic, descriptions in TOPIC_DESCRIPTIONS.items() for _ in descriptions]
        self.theme_keys = [(topic, name) for topic, themes in THEME_DESCRIPTIONS.items() for name in themes]
        self.theme_vectors = embedder.encode([THEME_DESCRIPTIONS[topic][name] for topic, name in self.theme_keys])

    def classify(self, texts, embeddings=None):
        texts = list(texts)
        columns = ['topic', 'topic_similarity', 'topic_margin', 'topic_review']
        if not texts:
            return pd.DataFrame(columns=columns)
        vectors = self.embedder.encode(texts) if embeddings is None else np.asarray(embeddings)
        if len(vectors) != len(texts):
            raise ValueError('Embeddings and texts must have the same length.')
        similarities = cosine_similarity(vectors, self.reference_vectors)
        scores = np.column_stack([similarities[:, [i for i, topic in enumerate(self.reference_topics) if topic == name]].max(axis=1) for name in self.names])
        order = np.argsort(-scores, axis=1, kind='stable')
        rows = []
        for i, ranked in enumerate(order):
            best = float(scores[i, ranked[0]])
            margin = best - float(scores[i, ranked[1]])
            label = self.names[ranked[0]] if best >= self.minimum_similarity else 'Other / Unclear'
            rows.append({'topic': label, 'topic_similarity': best, 'topic_margin': margin,
                         'topic_review': best < self.minimum_similarity + .08 or margin < self.review_margin})
        return pd.DataFrame(rows, columns=columns)

    def name_theme(self, topic, centroid):
        choices = [i for i, key in enumerate(self.theme_keys) if key[0] == topic]
        similarities = cosine_similarity(np.asarray(centroid).reshape(1, -1), self.theme_vectors[choices])[0]
        best = choices[int(similarities.argmax())]
        return self.theme_keys[best][1]
