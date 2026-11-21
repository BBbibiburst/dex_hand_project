from pathlib import Path
import json,csv,hashlib,ast
O=Path('docs/writing_audit/ch05_material_audit_v1_2026-11-20');d=json.loads((O/'verified_observations.json').read_text());sources=json.loads((O/'audited_sources.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):sources[str(p)]=sha(p);return p.read_text()
C4=Path('outputs/tactile_p43_c4_root_cause_audit_2026-11-16');rr=list(csv.DictReader(read(C4/'distance_summary.csv').splitlines()));d['distance_summary']=[r for r in rr if r['root_mode']=='cross_root' or (r['query']=='Paired_validation' and r['root_mode']=='same_root_allowed')]
for pat in ['tactile_c26*/C0_complete.json','tactile_c25*/coordinate_gate.json','tactile_c24*/stage0.json','tactile_c24*/training_summary.json','tactile_p43_c10_r1*/registration.json','tactile_p43_c13*/parameter_change_audit.json','tactile_p43_c13*/validation_decode_equivalence.json']:
 for p in Path('outputs').glob(pat):d[str(p)]=json.loads(read(p))
for pat in ['tactile_c27_c3*/candidate/checkpoint.pt','tactile_p39*/code/run.py','tactile_p39*/stage_metrics.csv','tactile_p39*/temporal_statistics.csv','tactile_p39*/input_counts.csv','tactile_p43_c10_r1*/oracle_tests.json','tactile_p43_c10_r1*/training_history.csv']:
 for p in Path('outputs').glob(pat):sources[str(p)]=sha(p)
# Add reliable source coordinates for subsequent report lookup without importing any model.
idx=json.loads((O/'source_code_index.json').read_text())
for pat in ['tactile_p43_c2*/code/*.py','tactile_p43_c5*/code/*.py','tactile_p43_c6*/code/*.py','tactile_p43_c10_r1*/code/*.py','tactile_p39*/code/run.py']:
 for p in Path('outputs').glob(pat):
  t=read(p)
  try:tree=ast.parse(t);defs=[{'name':n.name,'line':n.lineno,'end':n.end_lineno} for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
  except SyntaxError:defs=[]
  idx.append({'path':str(p),'definitions':defs,'lines':len(t.splitlines())})
(O/'source_code_index.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n');(O/'audited_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n');(O/'verified_observations.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print('sources',len(sources),'distance rows',len(d['distance_summary']))
