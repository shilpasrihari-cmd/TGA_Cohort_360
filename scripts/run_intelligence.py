"""Run the synthetic local pipeline and save aggregate-only review artifacts."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import synthetic_data
from src.synthetic_chat import synthetic_messages
from src.embedding_model import LocalEmbedder
from src.pipeline import analyze_messages
from src.learning_analytics import summarize_learning, privacy_safe_export

def main():
    people, fixture_questions = synthetic_data()
    result = analyze_messages(synthetic_messages(people, fixture_questions), LocalEmbedder())
    stats = summarize_learning(result.questions)
    manifest = json.loads((ROOT / 'models' / 'all-MiniLM-L6-v2' / 'download_manifest.json').read_text(encoding='utf-8'))
    # No IDs, raw examples, or representative message snippets in saved artifacts.
    report = {'data_status':'Synthetic', 'model':manifest['repository'], 'model_revision':manifest['revision'],
              'messages':len(result.detections), 'questions':len(result.questions), 'review_questions':stats['review_count'],
              'topic_counts':json.loads(stats['by_topic'].to_json(orient='records')),
              'category_counts':json.loads(stats['distribution'].to_json(orient='records')),
              'detector':'Python rules', 'classifier':'Local embedding similarity to curated topic descriptions',
              'clusterer':'Complete-link clustering of scikit-learn cosine distances (SciPy hierarchy)',
              'theme_naming':'Local embedding similarity to curated names; no generated summaries',
              'followup_selection':'Deterministic recurrence, participant and review rules'}
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    (output / 'synthetic_intelligence.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'synthetic_learning_themes.csv').write_bytes(privacy_safe_export(result.questions))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
