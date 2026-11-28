"""Phase0 read-only catalog: never load model/data arrays or execute experiments."""
from pathlib import Path
import json,re,hashlib,csv,collections,subprocess,sys
R=Path('/home/zmy/dex_hand_project');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
active=[];seen=set()
def walk(p):
 if p in seen:return
 seen.add(p);s=p.read_text();active.append(p)
 s=re.sub(r'(?<!\\)%.*','',s)
 for name in re.findall(r'\\(?:input|include)\{([^}]+)\}',s):
  q=R/'thesis'/(name if name.endswith('.tex') else name+'.tex')
  if q.is_file():walk(q)
walk(R/'thesis/thesis.tex')
figs=[];termrows=[];expr=[];lineitems=[]
for p in active:
 s=p.read_text();lines=s.splitlines()
 for m in re.finditer(r'\\begin\{figure\}(.*?)\\end\{figure\}',s,re.S):
  b=m[1];figs.append({'source':str(p.relative_to(R)),'line':s[:m.start()].count('\n')+1,'labels':re.findall(r'\\label\{([^}]+)\}',b),'external_graphics':re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',b),'placeholder':any(x in b for x in ['占位','图位'])})
 for token in ['GraspQP','DexEvolve','Wrist Lattice','Strict','Root','Session','Natural Sim','C0','FR','SWD','MMD','PRDC','HSIC','Translator','Refiner']:
  for i,l in enumerate(lines):
   if token in l:termrows.append({'term':token,'source':str(p.relative_to(R)),'first_file_line':i+1,'excerpt':l[:360]});break
 for phrase in ['不能','不等于','不代表','不保证','尚未','链路','依据','形成了']:
  expr.append({'source':str(p.relative_to(R)),'phrase':phrase,'count':s.count(phrase)})
 for i,l in enumerate(lines):
  if any(t in l for t in ['形成了','链路','不能','不等于']) and len(l)>80:lineitems.append({'source':str(p.relative_to(R)),'line':i+1,'text':l})
(O/'source_inventory.json').write_text(json.dumps({'active_sources':{str(p.relative_to(R)):sha(p) for p in active},'figures':figs,'tables':sum(p.read_text().count('\\begin{table}') for p in active),'equations':sum(p.read_text().count('\\begin{equation}') for p in active),'algorithms':sum(p.read_text().count('\\begin{algorithm}') for p in active),'first_mentions':termrows,'expression_counts':expr,'editing_candidates':lineitems},ensure_ascii=False,indent=2))
# Names and declared metadata only. L0 does not become result evidence.
catalog=[]
for p in sorted((R/'outputs').iterdir()):
 if not p.is_dir() or not any(x in p.name for x in ['tactile','grasp','top100','collection_quality','five_task_p3','dp_p']):continue
 docs=[q for q in p.iterdir() if q.is_file() and q.suffix in ['.md','.json','.csv','.yaml']]
 catalog.append({'directory':str(p.relative_to(R)),'direct_records':[{'name':q.name,'bytes':q.stat().st_size} for q in sorted(docs)],'level':'L0 unless separately audited'})
(O/'asset_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
# BibTeX entry inventory: quote local metadata, do not assert publication verification.
s=(R/'thesis/reference.bib').read_text();entries=[];starts=list(re.finditer(r'@(\w+)\s*\{\s*([^,]+),',s))
for i,m in enumerate(starts):
 b=s[m.end():starts[i+1].start() if i+1<len(starts) else len(s)];fields={}
 for key in ['title','author','year','journal','booktitle','doi','url']:
  n=re.search(r'\b'+key+r'\s*=\s*\{([^\n]*)\}',b,re.I)
  fields[key]=n[1].rstrip(',') if n else 'UNKNOWN'
 entries.append({'key':m[2],'entry_type':m[1],**fields,'verification':'local metadata only; Phase1 fulltext/publication check pending'})
(O/'bibliography_inventory.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2))
# Static test-function inventory, no pytest import or execution.
tests=[]
for p in sorted((R/'tests').glob('test_*.py')):
 if not any(x in p.name for x in ['production','collection','tactile_layout']):continue
 for i,l in enumerate(p.read_text().splitlines()):
  if l.startswith('def test_'):tests.append({'source':str(p.relative_to(R)),'line':i+1,'function':l.split('(')[0][4:]})
(O/'existing_test_inventory.json').write_text(json.dumps(tests,ensure_ascii=False,indent=2))
# P28 only extracts old Train-CV status, never reads test arrays/predictions.
p=R/'outputs/tactile_p28_transport_feasibility_2026-10-27/summary.json';x=json.loads(p.read_text());(O/'p28_status_extract.json').write_text(json.dumps({k:x[k] for k in ['status','folds','p24_train_groups_only','historical_validation_read','historical_test_read','upper_bound_worst_required_domain_confusion','upper_bound_best_required_domain_confusion','upper_bound_passed','transport_c_executed','conclusion','recommended_next_step']},ensure_ascii=False,indent=2))
# Small audited result tables only, not raw array/model inference.
rows=[]
for name in ['tactile_c13_c22_research_decision_2026-11-18/experiment_protocol_summary.csv','tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv','tactile_c27_c3_route_improvement_2026-11-18/C3_candidate_comparison.csv','tactile_p48_unified_four_route_audit_2026-11-19/metrics_swd.csv']:
 p=R/'outputs'/name
 if p.exists():
  with p.open() as f:table=list(csv.DictReader(f))
  rows.append({'source':str(p.relative_to(R)),'sha256':sha(p),'row_count':len(table),'rows':table})
(O/'audited_result_tables.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
# Resource description, no runtime benchmark.
(O/'resources.json').write_text(json.dumps({'machine':'current local workstation; not server','cpu_logical':subprocess.check_output(['nproc'],text=True).strip(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip(),'disk':subprocess.check_output(['df','-h','.'],cwd=R,text=True),'all_future_costs_are_planning_caps':True},ensure_ascii=False,indent=2))
print('active',len(active),'figures',len(figs),'bib entries',len(entries),'asset dirs',len(catalog),'test functions located',len(tests))
