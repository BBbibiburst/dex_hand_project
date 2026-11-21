"""Static chapter audit only. No TeX build, data arrays, models or inference."""
from pathlib import Path
import json,hashlib,csv,re,importlib.util,subprocess,collections
D=Path(__file__).resolve().parent;ROOT=Path.cwd();inputs={};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):p=Path(p);inputs[str(p)]=sha(p);return p.read_text()
def js(p):return json.loads(read(p))
def rows(p):return list(csv.DictReader(read(p).splitlines()))
def save(n,v):(D/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
base=js('docs/writing_audit/ch05_writing_v1_2026-11-20/section_5_6_v1_2026-11-21/revision_r1/freeze_final/integrity.json')['protected_hashes_before'];protected={p:sha(p) for p in base}
for root in [Path('thesis/releases'),Path('docs/writing_audit/ch05_writing_v1_2026-11-20/section_5_6_v1_2026-11-21/revision_r1/freeze_final')]:
 for p in root.rglob('*'):
  if p.is_file():protected[str(p)]=sha(p)
save('baseline.json',{'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'protected':protected})
# Actual text search; exclude caches, wordcount derivatives and third-party template assets.
search_command=['rg','--files','thesis','docs/writing_audit','-g','*.tex','-g','!**/wordcount/**','-g','!**/build/**','-g','!thesis/.tools/**','-g','!thesis/third_party/**','-g','!thesis/vendor/**']
paths=[Path(p) for p in subprocess.check_output(search_command,text=True).splitlines() if not str(p).startswith(str(D.relative_to(ROOT))+'/')]
search=[]
for p in paths:
 s=p.read_text()
 for m in re.finditer(r'\\(section|chapter)\{([^}]+)\}',s):
  if any(w in m[2] for w in ['触觉','伪配对','潜变量','双域','实验结果','非配对']):search.append({'path':str(p),'line':s.count('\n',0,m.start())+1,'command':m[1],'title':m[2]})
save('source_search.json',{'command':search_command,'tex_files_searched':len(paths),'matching_headings':search,'scope':'current repository thesis and writing audit only; excludes generated/build/wordcount/third-party'})
selected={
 '5.3':Path('thesis/releases/ch05_5_3_2026-11-21/section_5_3.tex'),
 '5.4':Path('thesis/releases/ch05_5_4_2026-11-21/section_5_4.tex'),
 '5.5':Path('docs/writing_audit/ch05_writing_v1_2026-11-20/section_5_5_v1_2026-11-21/revision_r1/section_5_5.tex'),
 '5.6':Path('thesis/releases/ch05_5_6_2026-11-21/section_5_6.tex')}
sp=importlib.util.spec_from_file_location('existing_wc','scripts/count_thesis_words.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);counter=m.Counter(D/'wordcount');resolver=m.Resolver(Path('thesis').resolve());resolver.expand(Path('thesis/thesis.tex'));mainfiles={str(p.resolve()) for p in resolver.files};save('main_include_graph.json',resolver.record());read('thesis/thesis.tex');read('thesis/body/ch05_experiments.tex');read('thesis/body/ch03_dual_vae.tex');read('thesis/body/ch02_data_and_problem.tex')
manifest=[];texts={};counts={};alllabels=[];allrefs=[];cites=[];structures={}
for section in ['5.1','5.2','5.3','5.4','5.5','5.6']:
 if section not in selected:
  manifest.append({'section':section,'authority':None,'sha256':None,'status':'MISSING_NOT_LOCATED','formal_body_found':False,'integrated_into_main':False,'wordcount':None,'search_evidence':'source_search.json; no standalone section or combined formal prose found'});continue
 p=selected[section];s=read(p);texts[section]=s;entry={'section':section,'authority':str(p),'sha256':sha(p),'status':'REVIEWED_R1_NOT_FROZEN' if section=='5.5' else 'FROZEN_DRAFT','formal_body_found':True,'integrated_into_main':str(p.resolve()) in mainfiles,'title':re.search(r'\\section\{([^}]+)\}',s)[1]}
 if section!='5.5':
  status=js(p.parent/'FREEZE_STATUS.json');entry['freeze_record']=str(p.parent/'FREEZE_STATUS.json');entry['freeze_record_matches_source']=status['section_sha256']==sha(p);assert entry['freeze_record_matches_source']
 else:
  entry['freeze_record']=None;entry['release_directory_exists']=Path('thesis/releases/ch05_5_5_2026-11-21').exists();entry['compile_copy_matches_R1']=Path(str(p.parent/'compile_review/section_5_5.tex')).read_bytes()==p.read_bytes();entry['temporary_compile_not_freeze']=True
 candidates=[]
 for q in paths:
  if q.name==p.name and q!=p:candidates.append({'path':str(q),'sha256':sha(q),'same_as_authority':sha(q)==sha(p)})
 entry['other_candidates']=candidates
 c=counter.run(s,'section_'+section.replace('.','_'));counts[section]=c;entry['wordcount']=c['total'];manifest.append(entry)
 structures[section]={'sections':len(re.findall(r'\\section\{',s)),'figures':len(re.findall(r'\\begin\{figure\}',s)),'tables':len(re.findall(r'\\begin\{table\}',s)),'equations':len(re.findall(r'\\begin\{equation\}',s)),'cite_occurrences':len(re.findall(r'\\cite\{',s))}
 for kind,pattern,dest in [('label',r'\\label\{([^}]+)\}',alllabels),('ref',r'\\(?:ref|eqref)\{([^}]+)\}',allrefs),('cite',r'\\cite\{([^}]+)\}',cites)]:
  for match in re.finditer(pattern,s):dest.append({'section':section,'path':str(p),'line':s.count('\n',0,match.start())+1,'value':match[1]})
save('SOURCE_MANIFEST.json',{'date':'2026-11-21','sections':manifest,'chapter5_7_context':'MISSING_NOT_LOCATED; outside requested5.1–5.6 but V2 complete chapter also needs summary','current_main_chapter5':{'path':'thesis/body/ch05_experiments.tex','sha256':sha('thesis/body/ch05_experiments.tex'),'status':'OLD_SIX_CHAPTER_PLACEHOLDER'},'main_path':'thesis/thesis.tex','main_sha256':sha('thesis/thesis.tex')})
merged='\n\n'.join(texts.values());combined=counter.run(merged,'available_sections_combined');counter.run(merged,'available_sections_combined',verbose=True);legacy=counter.run(read('thesis/body/ch05_experiments.tex'),'main_placeholder_chapter5');bib=read('thesis/reference.bib');bibkeys=re.findall(r'@\w+\s*\{\s*([^,]+),',bib);known=[x['value'] for x in alllabels]
static={'labels':alllabels,'duplicate_labels':{k:v for k,v in collections.Counter(known).items() if v>1},'references':allrefs,'unresolved_refs':[x for x in allrefs if x['value'] not in known],'citations':cites,'missing_citations':[x for x in cites if x['value'] not in bibkeys],'duplicate_used_bibkeys':[x for x in cites if bibkeys.count(x['value'])>1],'structures':structures,'combined':combined,'sum_available_counts':sum(c['total'] for c in counts.values()),'main_placeholder_count':legacy,'complete_chapter_count':None,'5_1_5_2_actual_counts':None,'no_compilation':True}
assert not static['duplicate_labels'] and not static['unresolved_refs'] and not static['missing_citations'];save('static_checks.json',static);save('wordcount_results.json',counts);save('wordcount/commands.json',counter.commands)
# Read normative sources and prior audited restrictions, then direct relevant implementation/validation scalar sources only.
for name in ['THESIS_PLAN_V2.md','THESIS_WRITING_STANDARD.md','ROBOTICS_EVIDENCE.md','EXPERIMENT_EVIDENCE.md','REFERENCE_THESIS_ANALYSIS.md','SYMBOLS_AND_TERMS.md','THESIS_LENGTH_BUDGET.md']:read(Path('thesis')/name)
for root in [Path('docs/writing_audit/ch05_material_audit_v1_2026-11-20'),Path('docs/writing_audit/ch05_54_material_audit_v1_2026-11-21'),Path('docs/writing_audit/ch05_55_material_audit_v1_2026-11-21')]:
 for p in root.glob('*.md'):read(p)
C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');C4=Path('outputs/tactile_p43_c4_root_cause_audit_2026-11-16')
for p in [C3/'code/models.py',C3/'code/pipeline_cpu.py',C3/'code/inference.py',C3/'code/expand.py',C3/'code/matching.py',C4/'code/pairs.py',Path('outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/code/report.py'),Path('outputs/tactile_c27_c3_route_improvement_2026-11-18/code/train.py')]:read(p)
refsources={'C3':('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/probe_metrics.csv',lambda r:r['role']=='validation' and r['group']=='F'),'C13':('outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv',lambda r:True),'C27':('outputs/tactile_c27_c3_route_improvement_2026-11-18/metrics.csv',lambda r:True),'C23':('outputs/tactile_c23_real_reconstruction_c0_2026-11-18/metrics.csv',lambda r:True),'C24':('outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/metrics.csv',lambda r:True)}
checks={}
for n,(p,select) in refsources.items():
 rr=[r for r in rows(p) if select(r)];assert len(rr)==11;out=[]
 for r in rr:
  ba=float(r['balanced_accuracy']);auc=float(r['roc_auc']);d=float(r['domain_confusion']);assert abs(d-(1-2*abs(ba-.5)))<1e-12;passed=.45<=ba<=.55 and .475<=auc<=.525 and d>=.9;out.append({'instance':r.get('instance',r.get('probe')),'BA':ba,'AUC':auc,'D':d,'passed':passed})
 checks[n]={'source':p,'instances':out,'passed':sum(r['passed'] for r in out),'BA_D_passed':sum(r['D']>=.9 for r in out),'role':'Real reconstruction diagnostic' if n in ['C23','C24'] else 'Natural Validation transfer'}
for p in [C4/'paired_cross_split.json',Path('outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/Real_C13_C24_structure.csv'),Path('outputs/tactile_c27_c3_route_improvement_2026-11-18/config.json'),Path('outputs/tactile_c25_latent_gmm_ot_feasibility_2026-11-18/decision.json'),Path('outputs/tactile_c26_shared_frozen_encoder_2026-11-18/decision.json')]:read(p)
save('validation_evidence_check.json',checks);save('source_hashes.json',inputs)
print(json.dumps({'manifest':[(r['section'],r['status'],r.get('wordcount')) for r in manifest],'available_total':combined['total'],'structures':structures,'current_main_ch5_count':legacy['total'],'all_joint':{n:r['passed'] for n,r in checks.items()},'sources':len(inputs)},ensure_ascii=False,indent=2))
