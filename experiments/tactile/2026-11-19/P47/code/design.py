from common import *
import time,urllib.request,subprocess,platform
P46=Path('outputs/tactile_p46_real_variability_envelope_audit_2026-11-18')
assert not (O/'design_complete.json').exists()
config=dict(date='2026-11-18',seed=4701,PRDC_k=[3,5,10,20],PRDC_reference='original author strict< definitions, kth neighbor excludes self by including diagonal at rankk',main_features='P46 full2848 log1p RealTrain scaler frozen',auxiliary='fixed global52 descriptor; RealTrain mean/std floor.05; not substitute main',equal_samples=780,equal_sampling='100 draws without replacement, root-proportional largest-remainder counts preserving source mixture; sharedindices across methods',MC_repeats=100,MC_are_independent_experiments=False,group_bootstrap=300,PRDC_bootstrap='conditional cluster bootstrap of fixed-neighborhood perpoint contributions, realSession and task-stratifiedSimRoot; not refitted neighborhoods',RR_reference='P46 four fixed8Session cohorts vs complementary24;100 draws equal780 each; shared originalregisteredcohorts',HSIC='biased centeredRBF HSIC and normalized centeredkernelalignment',HSIC_bandwidth='separate X/Y median strictlypositive pairdistances of registered developmentdata, fixedrule; scales .5,1,2; no metric-driven tuning',HSIC_median_fit_role='Validation unsupervised descriptive kernel setting, NOT new blind evaluation; no new Scaler',HSIC_within='10Root equalmacro;eachRoot has own median bandwith; report RootStage centered diagnostic separately',HSIC_null='199 perRoot circularwindow shifts preserving order except wrap; not iid-window permutation; descriptive since trajectory exchangeability not certified',HSIC_Root_null='all32 withinTask swaps of2Roots/task; only10Root-levelmean descriptors; exploratory',pair_shuffle='fixed seed independent permutation withinRoot/episode/stage, singleton unchanged; output multiset exact; MC PRDC uses inverseperm index bank to retain identical output subsets',temporal_shuffle='fixed [0,2,1,3] withinwindow for all13candidate datasets; reversal for Original andA_K1 extra falsification',task_stage='root-level labelkernelHSIC forTask; withinRoot stagekernelHSIC; no new networks;199Root-level label permutations and withinRoot cyclicStage nulls',FJD='onlyif certified commonphysicalconditions and outputrepresentation;otherwise NOT_APPLICABLE',no_new_090_definition=True,no_C0=True,no_Test_Holdout=True,no_generator_training=True,no_P48=True)
save(O/'experiment_config.json',config)
protected=json.loads((P46/'protected_hashes_before.json').read_text());p46manifest=json.loads((P46/'integrity.json').read_text())
for k,h in p46manifest['outputs'].items():assert sha(P46/k)==h;protected[str(P46/k)]=h
protected[str(P46/'integrity.json')]=sha(P46/'integrity.json')
assert all(sha(p)==h for p,h in protected.items());save(O/'protected_hashes_before.json',protected)
r=real('train');v=real('validation');nt=natural('train');n,out=inputs();counts={}
for name,d in [('RealTrain',r),('RealValidation',v),('NaturalTrain',nt),('NaturalValidation',n)]:
 assert d['x'].shape[1:]==(4,712) and np.isfinite(d['x']).all() and (d['x']>=0).all();counts[name]=dict(windows=len(d['x']),groups=len(np.unique(d['g'])))
assert counts=={'RealTrain':dict(windows=5153,groups=32),'RealValidation':dict(windows=780,groups=4),'NaturalTrain':dict(windows=4049,groups=50),'NaturalValidation':dict(windows=905,groups=10)}
inv=[]
for name,p in candidate_paths().items():
 x=out[name];assert x.shape==(905,4,712) and np.isfinite(x).all() and (x>=0).all();inv.append(dict(name=name,path=str(p),sha256=sha(p),shape=str(x.shape),role='Validation',source_groups=10,identity='P45B reuses P44' if name.startswith('B') else 'P45 own savedoutput'))
for p,h in protected.items():inv.append(dict(name='protected input',path=p,sha256=h,not_loaded_as_model=True))
table(O/'input_inventory.csv',inv);save(O/'actual_data_counts.json',counts)
rng=np.random.default_rng(4701);donors=rng.choice(len(r['x']),905,replace=True);pair=np.arange(905);align=[]
for g in sorted(set(n['g'])):
 for stage in sorted(set(n['s'][n['g']==g])):
  ix=np.flatnonzero((n['g']==g)&(n['s']==stage));pair[ix]=rng.permutation(ix);align.append(dict(Root=g,task=n['task'][ix[0]],stage=stage,windows=len(ix),scope='sameRoot sameEpisode sameStage',fixed_points=int((pair[ix]==ix).sum()),singletons=len(ix)==1))
assert sorted(pair.tolist())==list(range(905));assert np.array_equal(n['g'],n['g'][pair]) and np.array_equal(n['s'],n['s'][pair]);table(O/'alignment_audit.csv',align)
roots=sorted(set(n['g']));size=np.array([(n['g']==g).sum() for g in roots]);ideal=size*780/905;quota=np.floor(ideal).astype(int);remainder=780-quota.sum();quota[np.argsort(-(ideal-quota),kind='stable')[:remainder]]+=1;assert quota.sum()==780
banks=np.stack([np.concatenate([rng.choice(np.flatnonzero(n['g']==g),count,replace=False) for g,count in zip(roots,quota)]) for _ in range(100)]);np.savez_compressed(O/'index_banks.npz',equal_indices=banks,pair_permutation=pair,donor_indices=donors,Root=np.array(roots),Root_quota=quota,temporal_order=[0,2,1,3])
# Read and preserve official reference source only, not executed.
urls={'prdc.py':'https://raw.githubusercontent.com/clovaai/generative-evaluation-prdc/master/prdc/prdc.py','PRDC_LICENSE':'https://raw.githubusercontent.com/clovaai/generative-evaluation-prdc/master/LICENSE.txt'}
for name,url in urls.items():
 if not (O/'references'/name).exists():
  try:
   with urllib.request.urlopen(url,timeout=30) as response:(O/'references'/name).write_bytes(response.read())
  except Exception as exc:
   if name=='prdc.py':raise
   print('OPTIONAL_LICENSE_DOWNLOAD_FAILED',repr(exc),flush=True)
save(O/'references/sources.json',dict(PRDC_paper='https://proceedings.mlr.press/v119/naeem20a.html',precision_recall='https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html',FJD='https://arxiv.org/abs/1907.08175',official_source=urls))
save(O/'run_commands.json',dict(cwd=str(Path.cwd()),python='/home/zmy/miniconda3/envs/mujoco/bin/python',git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),environment=dict(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),commands=['design.py','prdc_eval.py','dependence.py','report.py'],all_scripts_in=str(O/'code')))
save(O/'design_complete.json',dict(all_hashes_pass=True,data_counts=counts,root_quota=quota.tolist(),code_hashes={p.name:sha(p) for p in (O/'code').glob('*.py')}));print('P47 DESIGN COMPLETE',counts,flush=True)
