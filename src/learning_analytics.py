"""Deterministic aggregation and allowlisted exports; never export message snippets."""
import pandas as pd
from .topic_classifier import TOPICS, THEME_DESCRIPTIONS

THEME_COLUMNS = ['cluster_id', 'theme', 'topic', 'question_count', 'participant_count', 'needs_review']
ALLOWED_THEME_NAMES = {name for themes in THEME_DESCRIPTIONS.values() for name in themes}

def theme_statistics(questions):
    rows = []
    for cluster_id, group in questions.groupby('cluster_id', sort=True):
        rows.append({'cluster_id': cluster_id, 'theme': group.theme.iloc[0], 'topic': group.topic.iloc[0],
                     'question_count': len(group), 'participant_count': group.participant_id.nunique(),
                     'needs_review': bool(group.needs_review.any())})
    return pd.DataFrame(rows, columns=THEME_COLUMNS).sort_values(['question_count', 'cluster_id'], ascending=[False, True]).reset_index(drop=True)

def summarize_learning(questions):
    themes = theme_statistics(questions)
    by_topic = questions.groupby('topic').size().reset_index(name='question_count')
    distribution = questions.groupby('category').size().reset_index(name='question_count')
    recurring = themes[themes.question_count >= 2].reset_index(drop=True)
    followups = recurring[(recurring.participant_count >= 2) & (~recurring.needs_review) & (recurring.topic != 'Course Support')].head(6).copy()
    followups['suggested_session'] = followups.theme.map(lambda theme: f'{theme}: practical walkthrough and Q&A')
    flags = questions.has_identifiable_answer.dropna()
    answer_value = int((~flags.astype(bool)).sum()) if len(flags) == len(questions) and len(questions) else (0 if questions.empty else None)
    return {'themes': themes, 'recurring': recurring, 'by_topic': by_topic, 'distribution': distribution,
            'followups': followups,
            'metrics': [len(questions), questions[questions.category == 'Learning'].topic.nunique(), len(recurring), answer_value],
            'review_count': int(questions.needs_review.sum())}

def privacy_safe_export(questions, minimum_participants=3):
    """Aggregate CSV, suppressed below k distinct people. Not a formal privacy guarantee.

    Build through an allowlist, never drop a few sensitive columns from raw rows.
    Controlled labels prevent arbitrary input text masquerading as an aggregate name.
    """
    if questions.attrs.get('synthetic') is not True or questions.attrs.get('data_mode') == 'real':
        raise ValueError('Legacy demo export is synthetic-only; restricted data requires an approved fixed snapshot.')
    if minimum_participants < 3:
        raise ValueError('Public demo exports require at least three distinct participants per group.')
    themes = theme_statistics(questions)
    if not themes.theme.isin(ALLOWED_THEME_NAMES).all() or not themes.topic.isin(TOPICS).all():
        raise ValueError('Export labels must come from the controlled topic/theme vocabulary.')
    export = themes[themes.participant_count >= minimum_participants][['theme', 'topic', 'question_count', 'participant_count', 'needs_review']].copy()
    export.insert(0, 'data_status', 'Synthetic')
    return export.to_csv(index=False).encode('utf-8')
