from pathlib import Path
import hashlib,json,ast
import numpy as np
O=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
protected=json.loads((O/'protected_hashes_before.json').read_text())
changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h]
a=json.loads((O/'protocol_amendment_2026-11-19.json').read_text())
assert not changed,changed
assert sha(O/'registration.json')==a['original_registration_sha256']
assert sha(O/'checkpoints/D1.pt')=='21e56c3d94a0e8c0c38ed47f06ab6f629e3a08b7151bf7c1d12440381f1fad7d'
for p in (O/'code').glob('*.py'):ast.parse(p.read_text())
for k in range(4):
 x=np.load(O/f'outputs/F{k}_validation.npz')['values']
 assert x.shape==(905,4,712) and x.dtype==np.float32 and np.isfinite(x).all() and (x>=0).all()
c=json.loads((O/'complete.json').read_text());assert c['rounds_G_completed']==3 and c['D_optimizer_steps']==3300 and c['F_optimizer_steps']==3840 and c['cumulative_optimization_wall_seconds']<1800
result=dict(protected_files_checked=len(protected),protected_files_changed=changed,original_registration_unchanged=True,D1_unchanged_since_review=True,all_validation_outputs_valid=True,python_AST_valid=True,round_and_budget_verified=True,classification='B: FR passed, quality failed',no_additional_training=True)
(O/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
manifest={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','finalize.log']}
(O/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(result,indent=2))
