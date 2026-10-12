"""Explicit one-time network setup; never imported by the app or inference pipeline."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPO = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
ROOT = Path(__file__).resolve().parents[1] / 'models' / 'all-MiniLM-L6-v2'
FILES = {'modules.json', 'config.json', 'config_sentence_transformers.json',
         'sentence_bert_config.json', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt',
         '1_Pooling/config.json', 'README.md', 'LICENSE'}

def download():
    with urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}', timeout=60) as response:
        metadata = json.load(response)
    revision = metadata['sha']
    available = {entry['rfilename'] for entry in metadata['siblings']}
    needed = FILES & available
    if not {'modules.json', 'model.safetensors', 'tokenizer.json', '1_Pooling/config.json'} <= needed:
        raise RuntimeError('Expected safe model files are missing from the repository.')
    hashes = {}
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in sorted(needed):
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f'https://huggingface.co/{REPO}/resolve/{revision}/{name}', timeout=120) as response:
            payload = response.read()
        temporary = target.with_suffix(target.suffix + '.part')
        temporary.write_bytes(payload)
        temporary.replace(target)
        hashes[name] = hashlib.sha256(payload).hexdigest()
        print(f'Stored {name} ({len(payload):,} bytes)', flush=True)
    # Keep the upstream Apache license with the local model even if absent in repo.
    if 'LICENSE' not in needed:
        with urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt', timeout=60) as response:
            (ROOT / 'LICENSE').write_bytes(response.read())
    manifest = {'repository': REPO, 'revision': revision, 'license': 'Apache-2.0', 'sha256': hashes}
    (ROOT / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Local model ready: {ROOT}; revision {revision}', flush=True)

if __name__ == '__main__':
    download()
