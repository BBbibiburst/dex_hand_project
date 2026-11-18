import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=json.loads((ROOT/'SHA256SUMS.json').read_text())
for name,digest in files.items():
 p=ROOT/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError('Missing/changed: '+name)
print('SHA256 PASS',len(files),'files')
