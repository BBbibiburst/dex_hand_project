"""Verify the archive without training or reading historical raw datasets."""
from pathlib import Path
import ast,hashlib,json
B=Path(__file__).resolve().parent
source=json.loads((B/'source_inventory.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
for f in source['files']:
 if f['archived']:assert sha(B/f['archive_path'])==f['sha256'],f['archive_path']
for p in B.rglob('*.py'):ast.parse(p.read_text())
manifest=json.loads((B/'archive_hashes.json').read_text())
for name,h in manifest.items():assert sha(B/name)==h,name
print('PASS: archived source hashes, Python syntax and snapshot manifest')
