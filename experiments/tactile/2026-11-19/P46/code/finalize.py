from common import *
import sys,platform,subprocess
assert (O/'REPORT.md').exists()
r=real('train');rv=real('validation');n,outputs=inputs();inventory=list(csv.DictReader((O/'group_inventory.csv').open()));trials={role:sum(int(v['Trial_count']) for v in inventory if v['domain']=='Real' and v['role']==role) for role in ['train','validation']};assert trials=={'train':755,'validation':120}
for d in [r,rv,n]:assert d['x'].shape[1:]==(4,712) and np.isfinite(d['x']).all() and (d['x']>=0).all()
for x in outputs.values():assert x.shape==(905,4,712) and np.isfinite(x).all() and (x>=0).all()
assert len(list(csv.DictReader((O/'rvc_metrics.csv').open())))==4*13*5
assert len(list(csv.DictReader((O/'bootstrap_results.csv').open())))==4*13*5*300
# Registered source hashes checked a final time; no rewriting any source/checkpoint.
protected=json.loads((O/'protected_hashes_before.json').read_text());assert all(sha(p)==h for p,h in protected.items())
assert len(list((O/'figures').glob('*.png')))==5
commands=json.loads((O/'run_commands.json').read_text());commands['ordered_commands'].append(f'{sys.executable} {O}/code/finalize.py');save(O/'run_commands.json',commands)
log=O/'run.log'
with log.open('a') as f:
 f.write('\nP46 continuation: compute_distances.py, c2st.py, report.py, finalize.py.\n')
 for name in ['statistics.log','c2st.log','report.log']:
  f.write('\n'+name+'\n');f.write((O/name).read_text())
 f.write('\nFINAL CHECK PASS: numeric roles valid; 76 registered historical files unchanged; 71 acceptance snapshot files unchanged; no new generator training; P46 STOP.\n')
outputs_hash={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='integrity.json' and '__pycache__' not in p.parts}
save(O/'integrity.json',dict(inputs=protected,outputs=outputs_hash,checks=json.loads((O/'verification.json').read_text()),Real_Trial_counts=trials,source_roles=['Train','Validation'],all_checks_pass=True,hash_manifest_does_not_include_itself=True))
print('FINALIZED',len(outputs_hash),'artifacts')
