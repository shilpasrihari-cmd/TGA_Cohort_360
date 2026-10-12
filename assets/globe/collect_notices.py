"""Collect license notices after npm install in vendor-build (development only)."""
import json
from pathlib import Path
root = Path(__file__).parent
notices = []
for metadata in sorted((root / 'vendor-build' / 'node_modules').rglob('package.json')):
    package = json.loads(metadata.read_text(encoding='utf-8'))
    if not package.get('name') or not package.get('version'):
        continue
    licenses = [p for p in metadata.parent.iterdir() if p.is_file() and p.name.lower().startswith(('license', 'licence', 'copying'))]
    if licenses:
        notices.append(f"\n{'=' * 72}\n{package['name']} {package['version']} ({package.get('license', 'see notice')})\n")
        notices.extend(p.read_text(encoding='utf-8', errors='replace') for p in licenses)
(root / 'THIRD_PARTY_NOTICES.txt').write_text('\n'.join(notices), encoding='utf-8')
print(f'Collected {len(notices)} notice sections')
