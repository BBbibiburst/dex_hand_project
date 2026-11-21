#!/usr/bin/env python3
"""Verify planning V2 and preservation, without compiling or training."""
from pathlib import Path
import hashlib,json,re,csv
R=Path(__file__).resolve().parents[1]; P=R.parent; O=R/'docs/writing_audit/v2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
errors=[]
required=['THESIS_PLAN_V2.md','ROBOTICS_EVIDENCE.md','THESIS_WRITING_STANDARD.md','REFERENCE_THESIS_ANALYSIS.md','EXPERIMENT_EVIDENCE.md','SYMBOLS_AND_TERMS.md','AGENTS.md']
for name in required:
 if not (R/name).is_file(): errors.append('missing '+name)
plan=(R/'THESIS_PLAN_V2.md').read_text(); cards=re.split(r'^### \d+\.\d+ ',plan,flags=re.M)[1:]
for i,card in enumerate(cards):
 for f in ['问题','核心论点','实验依据','公式','图表需求','前后关系','材料缺口']:
  if f not in card: errors.append(f'missing {f} in card{i}')
if len(cards)!=42: errors.append('expected 42 cards')
old=json.loads((R/'docs/writing_audit/workflow_validation.json').read_text())
if sha(R/'THESIS_PLAN.md')!=old['files']['THESIS_PLAN.md']: errors.append('V1 plan modified')
for name in ['AGENTS.md','THESIS_WRITING_STANDARD.md','REFERENCE_THESIS_ANALYSIS.md','EXPERIMENT_EVIDENCE.md','SYMBOLS_AND_TERMS.md','README.md']:
 if sha(R/'docs/history/workflow_v1_2026-11-19'/name)!=old['files'][name]: errors.append('old snapshot mismatch '+name)
base=json.loads((R/'docs/project_integrity.json').read_text())['files']; protected=[]
for name,h in base.items():
 if Path(name).suffix in ('.tex','.cls','.cfg','.sty','.bst'):
  protected.append(name)
  if sha(R/name)!=h: errors.append('LaTex source changed '+name)
robot=json.loads((O/'robotics_source_audit.json').read_text())
for v in robot:
 if sha(P/v['path'])!=v['sha256']: errors.append('robotics source changed '+v['path'])
for v in json.loads((R/'docs/writing_audit/evidence_catalog.json').read_text()):
 if sha(P/v['path'])!=v['sha256']: errors.append('historical report changed '+v['path'])
p=P/'outputs/tactile_p48_unified_four_route_audit_2026-11-19/metrics_swd.csv'
row=next(x for x in csv.DictReader(p.open()) if x['model']=='A_K1')
assert abs(float(row['R_W'])-.4977202817338595)<1e-12
report={'date':'2026-11-19','errors':errors,'chapter_cards':len(cards),'V1_plan_unchanged':True if not any('V1 plan' in s for s in errors) else False,'protected_latex_files':len(protected),'robotics_artifacts_and_code_hashed':len(robot),'P48_CSV':{'path':str(p.relative_to(P)),'sha256':sha(p),'row':row},'status':'planning only; not chapter task brief, no official prose, no training, no LaTeX build, no git commit/push','limitations':['server current data UNKNOWN','not all robotics episodes independently accepted','seven-chapter structure awaits user/ChatGPT review']}
(O/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='P48_CSV'},ensure_ascii=False,indent=2))
if errors: raise SystemExit(1)
