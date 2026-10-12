"""CPU inference from a real local Sentence Transformers model, fail closed."""
import os
from pathlib import Path
from threading import RLock
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'all-MiniLM-L6-v2'

class ModelUnavailableError(RuntimeError):
    pass

class LocalEmbedder:
    def __init__(self, model_dir=MODEL_DIR):
        location = Path(model_dir).resolve()
        if not location.is_dir() or not (location / 'modules.json').is_file() or not (location / 'model.safetensors').is_file():
            raise ModelUnavailableError('Local all-MiniLM-L6-v2 files are missing. Run scripts/download_model.py once; inference never downloads a model.')
        # Set before importing HF/Transformers; offline mode is also forced on load.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            self.model = SentenceTransformer(str(location), device='cpu', local_files_only=True,
                                            trust_remote_code=False, model_kwargs={'use_safetensors': True})
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            raise ModelUnavailableError('The local embedding model could not be loaded; no fallback AI results were generated.') from error
        self.dimension = self.model.get_embedding_dimension()
        self._lock = RLock()

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Embedding inputs must be nonempty strings.')
        # Identical fixture questions are encoded once, then expanded, preserving counts.
        unique = list(dict.fromkeys(texts))
        with self._lock:
            vectors = self.model.encode(unique, normalize_embeddings=True, convert_to_numpy=True,
                                        batch_size=32, show_progress_bar=False)
        lookup = dict(zip(unique, vectors))
        return np.asarray([lookup[text] for text in texts], dtype=np.float32)
