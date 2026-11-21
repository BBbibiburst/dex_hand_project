"""Read-only writing evidence inventory. No arrays, models or experiments are run."""
from pathlib import Path
import json,hashlib,csv,re
R=Path(__file__).resolve().parents[2];O=R/'thesis/docs/writing_audit';O.mkdir(exist_ok=True)
files=set((R/'docs/progress').glob('*.md'))
for base in ['outputs','experiments/tactile']:
 for p in (R/base).rglob('*.md'):
  rel=p.relative_to(R/base)
  if len(rel.parts)>4:continue
  if not any(k in str(rel).lower() for k in ['tactile','dp_p','five_task_p31','five_task_storage','five_task_runtime']):continue
  if not any(k in p.name.lower() for k in ['report','recommendation','readme','audit','scope']):continue
  files.add(p)
records=[]
for p in sorted(files):
 txt=p.read_text(errors='replace');rel=str(p.relative_to(R));route='TACTILE' if 'tactile' in rel else ('IMITATION' if 'dp_' in rel else 'DATASET')
 records.append(dict(path=rel,route=route,title=next((x.lstrip('# ') for x in txt.splitlines() if x.startswith('#')),''),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),characters=len(txt),review_status='indexed; see curated EXPERIMENT_EVIDENCE for verified conclusions',opening='\n'.join([s for s in txt.splitlines() if s.strip()][:12])))
(O/'evidence_catalog.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
with (O/'evidence_catalog.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['route','title','path','sha256','characters','review_status']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in records)
claims=[]
def claim(key,path,select,field,scope):
 p=R/path;rows=list(csv.DictReader(p.open()));matches=[r for r in rows if all(r.get(k)==v for k,v in select.items())];assert len(matches)==1,(key,len(matches));claims.append(dict(claim_id=key,source=path,selector=json.dumps(select,ensure_ascii=False),field=field,value=matches[0][field],scope=scope,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
p='outputs/tactile_p51_discriminator_trust_audit_2026-11-19/'
for name in ['F50','Original']:
 for field in ['FR_macro','Real_acceptance_macro','BA','AUC']:
  claim('P51_'+name+'_'+field,p+'classification_metrics.csv',{'model':name,'branch':'Joint'},field,'frozenD50, existing developmentValidation,Root/Session macro')
claim('P51_F50_above_envelope',p+'score_summary.csv',{'domain':'F50','role':'evaluation','branch':'Joint'},'above_envelope','inner8 calibrated logit interval; auxiliary only')
p='outputs/tactile_p50_residual_multiview_2026-11-19/'
for name in ['F0','P49_F3','F50','F50_without_refiner']:
 for field in ['SWD','MMD']:claim('P50_'+name+'_'+field,p+'distribution_metrics.csv',{'model':name},field,'same fixed RealTrain preprocessing and groupweighted developmentValidation geometry')
p='outputs/tactile_p49_round_frozen_adversarial_2026-11-19/'
for field in ['FR_pre','FR_post','Real_acceptance']:claim('P49_round3_'+field,p+'round_metrics.csv',{'round':'3'},field,'D3 frozen pre/post; cannot compare differentD as sameboundary')
with (O/'verified_claims.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(claims[0]));w.writeheader();w.writerows(claims)
# Minimal metadata index for source modules, read as text, no importing training scripts.
code=['outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/code/model.py','outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/code/vae.py','outputs/tactile_p49_round_frozen_adversarial_2026-11-19/code/evaluate.py','outputs/tactile_p49_round_frozen_adversarial_2026-11-19/code/train.py','outputs/tactile_p50_residual_multiview_2026-11-19/code/architecture.py','outputs/tactile_p51_discriminator_trust_audit_2026-11-19/code/run.py','outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/c0.py']
(O/'verified_code_sources.json').write_text(json.dumps([dict(path=p,sha256=hashlib.sha256((R/p).read_bytes()).hexdigest()) for p in code],indent=2)+'\n')
print('Writing inventory:',len(records),'reports indexed,',len(claims),'exact CSV claims checked; no training')
