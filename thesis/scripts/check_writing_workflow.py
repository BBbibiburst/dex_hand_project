#!/usr/bin/env python3
"""Check writing workflow provenance without loading experimental arrays/models."""
from pathlib import Path
import csv, hashlib, json, re, subprocess
R=Path(__file__).resolve().parents[1]; P=R.parent; A=R/'docs/writing_audit'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
errors=[]
names=['REFERENCE_THESIS_ANALYSIS.md','THESIS_WRITING_STANDARD.md','THESIS_PLAN.md','EXPERIMENT_EVIDENCE.md','SYMBOLS_AND_TERMS.md']
for n in names:
 if not (R/n).is_file() or (R/n).stat().st_size<100: errors.append('missing document '+n)
 if n not in (R/'AGENTS.md').read_text(): errors.append('missing instruction '+n)
plan=(R/'THESIS_PLAN.md').read_text(); sections=re.split(r'^### \d+\.\d+ ',plan,flags=re.M)[1:]
for i,s in enumerate(sections):
 for field in ['问题','核心论点','实验依据','公式','图表需求','前后关系','材料缺口']:
  if field not in s: errors.append(f'section {i+1} lacks {field}')
catalog=json.loads((A/'evidence_catalog.json').read_text())
for row in catalog:
 p=P/row['path']
 if not p.is_file() or sha(p)!=row['sha256']: errors.append('source changed '+row['path'])
claims=list(csv.DictReader((A/'verified_claims.csv').open()))
for c in claims:
 p=P/c['source']
 if sha(p)!=c['source_sha256']: errors.append('claim source changed '+c['claim_id'])
 rows=list(csv.DictReader(p.open())); selector=json.loads(c['selector'])
 found=[v for v in rows if all(v[k]==str(x) for k,x in selector.items())]
 if len(found)!=1 or abs(float(found[0][c['field']])-float(c['value']))>1e-12: errors.append('claim mismatch '+c['claim_id'])
meta=json.loads((A/'reference_pdf_metadata.json').read_text())
if sha(P/meta['source'])!=meta['sha256']: errors.append('reference PDF changed')
baseline=json.loads((R/'docs/project_integrity.json').read_text())['files']; protected=[]
for n,h in baseline.items():
 if Path(n).suffix in ['.tex','.cls','.cfg','.sty','.bst']:
  protected.append(n)
  if sha(R/n)!=h: errors.append('protected thesis source changed '+n)
files=names+['AGENTS.md','README.md','docs/school_requirements.md','scripts/build_writing_evidence.py','scripts/check_writing_workflow.py']
files+= [str(p.relative_to(R)) for p in sorted(A.iterdir()) if p.is_file() and p.name not in ['workflow_validation.json','writing_workflow_files.txt']]
(A/'writing_workflow_files.txt').write_text('\n'.join(files)+'\n')
result={'date':'2026-11-19','errors':errors,'section_cards':len(sections),'indexed_reports':len(catalog),'verified_csv_claims':len(claims),'unchanged_tex_and_template_files':len(protected),'reference_pages':meta['pages'],'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),'scope':'writing workflow only; no training, no official prose, no commit/push','limitations':['indexed reports are not all independently re-evaluated','auxiliary theses unavailable/unread as recorded','literature and school administrative fields still require verification'],'files':{n:sha(R/n) for n in files}}
(A/'workflow_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
if errors: raise SystemExit(1)
