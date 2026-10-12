# Local embedding model

Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0, 384-dimensional embeddings. The model has been downloaded to `all-MiniLM-L6-v2/` in this workspace; large model files are excluded from Git.

Tested repository revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

From the project directory, one-time setup:

```powershell
..\venv\Scripts\python.exe scripts\download_model.py
```

This is the only model setup operation that accesses the internet. It downloads public model files and the license, sends no chat data, pins every download to a single repository commit, and records SHA-256 hashes and the revision in `download_manifest.json`. No untrusted Python model code or pickle weights are downloaded. Inference uses a local path, safe tensor weights, `local_files_only=True`, and `trust_remote_code=False`; Hugging Face offline mode and telemetry suppression are enabled before model imports.

The model is a general-purpose English similarity model, not a TGA-specific trained classifier. Taxonomy descriptions and thresholds are heuristic and need validation before connecting any real data. Long messages may be truncated by the model; messages over 120 whitespace tokens are conservatively flagged for review.
