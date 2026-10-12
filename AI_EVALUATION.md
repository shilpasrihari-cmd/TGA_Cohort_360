# Synthetic AI verification

Reproduce with `..\venv\Scripts\python.exe scripts/evaluate_model.py`. This script blocks outbound socket connections, loads only four fixed synthetic CSVs, and writes `reports/synthetic_model_evaluation.json`. It accepts no transcript path. The dashboard computes the same scores using cached local inference; it does not read scores from the report.

## Verified data flow

1. `src/mock_dataset.py:load_mock` validates the four fictional CSVs. Ground truth participates in fixture consistency validation; it is not an inference target or feature. The source topic is fictional channel metadata for filters.
2. `learning_ui.py:render_learning` renames message text and drops the fixture `topic` column. Only authored message content reaches `src/pipeline.py:analyze_messages`.
3. `src/question_detector.py` applies deterministic Python rules. Nonquestions do not enter classification or clustering.
4. `src/embedding_model.py:LocalEmbedder` loads local MiniLM safetensors with `local_files_only=True`, disabled remote code, offline environment flags, and CPU inference. Missing model files fail closed.
5. `src/topic_classifier.py` compares normalized message embeddings against curated topic descriptions; it does not read ground truth. Low similarity produces Other / Unclear. Scores and margins drive review flags.
6. `src/question_clusterer.py` uses complete-link cosine clustering within predicted topics, requiring similarity >=0.70 for every cluster pair. Cluster IDs derive from these groups. Theme names are nearest curated descriptions, not generated summaries.
7. `src/learning_analytics.py` calculates counts, unique fictional participants, review totals, and deterministic eligible follow-ups. `learning_ui.py` displays these predictions in both learning tabs after existing filters.
8. `src/model_evaluation.py` joins predictions to separate authored labels only after inference. Tests poison channel metadata without changing model topics or cluster IDs, and change evaluation labels without altering predictions. The evaluation section covers the fixed full fixture and explicitly ignores sidebar filters.

## Actual results

Model: all-MiniLM-L6-v2 (local CPU). Inventory: 250 messages, including 176 labeled requests. Question detection precision/recall/F1: 1.0000/1.0000/1.0000. Topic accuracy: 0.971591; macro-F1 across the eight authored topics: 0.987745. Other / Unclear is an abstention, counted as a false negative for the authored topic rather than a ninth supported truth class. 76 question predictions require human review.

Five topic errors are repeats of “How do I configure an API key locally without exposing it?” Authored topic: Development Tools. Prediction: Other / Unclear. All five are flagged. Credential setup combines tool configuration and security vocabulary, and similarity falls below the classifier threshold. These are one repeated wording error, not five independent failure patterns.

The 12 separately authored pairs contain six same-intent positives and six different-intent negatives, including negatives within a broad topic. They were not copied from curated classifier/theme questions. We run all pair texts through the unchanged model pipeline, then compare final cluster membership against the binary labels. Precision 1.0000; recall 0.166667; F1 0.285714; accuracy 0.583333. This means one true positive, five false negatives, six true negatives, and no false positives. Semantic paraphrases can be separated by the strict complete-link threshold, broad-topic boundary, or low topic similarity. These results do not justify claiming reliable paraphrase clustering. No thresholds were tuned against these pairs.

The full nine-column confusion matrix, errors, and pair outcomes are included in the JSON report and dashboard expander. Rows are ground truth; columns are predictions. Of 28 Development Tools requests, 23 are correct and five abstain. Each of the other seven topics has all its requests on the diagonal (20 each, except Course Support with 28).

## Limits

The mock inventory repeats templates, including curated theme descriptions used to name clusters. Inventory topic scores therefore overestimate generalization. The independent pairs are a small developer-authored benchmark, not an independent human-adjudicated study. No real chat has been evaluated. Review flags indicate pending validation, not completed review. The globe and geographic analytics are untouched. No external AI API is used.
