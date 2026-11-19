from support47 import *
import time,itertools
from scipy.stats import spearmanr
assert (O/'design_complete.json').exists() and not (O/'dependence_complete.json').exists();start=time.time();n,d=datasets();bank=np.load(O/'index_banks.npz');pair=bank['pair_permutation'];z=np.load(P46/'RealTrain_preprocessing.npz');X=transform(n['x'],z['mean'],z['scale']);base=list(d)
for name in candidate_paths():d[name+'_PairShuffle']=d[name][pair]
roots=sorted(set(n['g']));rootix=[np.flatnonzero(n['g']==g) for g in roots];root_tasks=np.array([n['task'][ix[0]] for ix in rootix]);within=[];hsics=[];nullrows=[];retention=[];temporal=[];boot=[];rng=np.random.default_rng(4701);root_boot=np.concatenate([rng.choice(np.flatnonzero(root_tasks==task),(300,2),replace=True) for task in sorted(set(root_tasks))],axis=1)
# Predeclared cyclic-shift nulls preserve each full trajectory block except wrap; no iid-window inference.
rootshifts={g:np.random.default_rng(group_seed('P47shift'+str(g))).integers(1,len(ix),199) for g,ix in zip(roots,rootix)}
for model,y in d.items():
 Y=transform(y,z['mean'],z['scale']);xroot=np.stack([X[ix[0]]+(X[ix]-X[ix[0]]).mean(0) for ix in rootix]);yroot=np.stack([Y[ix[0]]+(Y[ix]-Y[ix[0]]).mean(0) for ix in rootix]);kxroot,sxroot=kernel(xroot);kyroot,syroot=kernel(yroot);crx,cry=center(kxroot),center(kyroot);obsroot=hsic_from_centered(crx,cry);groupnull=[]
 # All withinTask swaps: 2Roots/task gives only32 permutations, including identity.
 taskpairs=[np.flatnonzero(root_tasks==t) for t in sorted(set(root_tasks))]
 for bits in itertools.product([0,1],repeat=5):
  pp=np.arange(10)
  for bit,ix in zip(bits,taskpairs):
   if bit:pp[ix]=ix[::-1]
  groupnull.append(hsic_from_centered(crx,cry[np.ix_(pp,pp)])['normalized_HSIC'])
 hsics.append(dict(model=model,scope='10Root means;all32withinTask swaps inclidentity',bandwidth_scale=1,units=10,**obsroot,null_tail_fraction=float((np.array(groupnull)>=obsroot['normalized_HSIC']-1e-12).mean()),X_sigma=sxroot,Y_sigma=syroot,inference='descriptive task-preserving Root null; only2Roots/task'))
 for scale in [.5,1.,2.]:
  kx,sx=kernel(X,scale);ky,sy=kernel(Y,scale);globalvalue=hsic_from_centered(center(kx),center(ky));hsics.append(dict(model=model,scope='all905window global descriptive',bandwidth_scale=scale,units=10,X_sigma=sx,Y_sigma=sy,**globalvalue,inference='no window-iid pvalue'))
  perroot=[];nulls=[];stage_values=[]
  for ri,(g,ix) in enumerate(zip(roots,rootix)):
   a,s1=kernel(X[ix],scale);b,s2=kernel(Y[ix],scale);a,b=center(a),center(b);obs=hsic_from_centered(a,b);perroot.append(obs['normalized_HSIC']);nv=[]
   for shift in rootshifts[g]:
    pp=np.roll(np.arange(len(ix)),int(shift));nv.append(hsic_from_centered(a,b[np.ix_(pp,pp)])['normalized_HSIC'])
   nulls.append(nv);within.append(dict(model=model,Root=g,task=root_tasks[ri],scope='withinRoot',bandwidth_scale=scale,windows=len(ix),X_sigma=s1,Y_sigma=s2,**obs,null_mean=float(np.mean(nv)),cyclic_tail_fraction=float((1+(np.array(nv)>=obs['normalized_HSIC']-1e-12).sum())/200),inference='cyclic trajectory null; exchangeability not certified'))
   blockvalues=[]
   for stage in sorted(set(n['s'][ix])):
    ii=ix[n['s'][ix]==stage]
    if len(ii)<4:continue
    aa,s1=kernel(X[ii],scale);bb,s2=kernel(Y[ii],scale);vv=hsic_from_centered(center(aa),center(bb));blockvalues.append((len(ii),vv['normalized_HSIC']));within.append(dict(model=model,Root=g,task=root_tasks[ri],stage=stage,scope='withinRootStage',bandwidth_scale=scale,windows=len(ii),X_sigma=s1,Y_sigma=s2,**vv,inference='atleast4windows/block; descriptive only'))
   stage_values.append(float(np.average([v for _,v in blockvalues],weights=[w for w,_ in blockvalues])) if blockvalues else None)
  nulls=np.array(nulls);pv=float(np.mean(perroot));nullmean=nulls.mean(0);vv=np.array(perroot);bb=vv[root_boot].mean(1);hsics.append(dict(model=model,scope='withinRoot equalmacro',bandwidth_scale=scale,units=10,normalized_HSIC=pv,null_mean=float(nullmean.mean()),cyclic_tail_fraction=float((1+(nullmean>=pv-1e-12).sum())/200),CI_low=float(np.quantile(bb,.025)),CI_high=float(np.quantile(bb,.975)),inference='Rootcluster bootstrap; cyclicnull is exploratory'))
  hsics.append(dict(model=model,scope='withinRootStage rootmacro',bandwidth_scale=scale,units=sum(v is not None for v in stage_values),normalized_HSIC=float(np.mean([v for v in stage_values if v is not None])),inference='removes Root andliteralStage block offsets; no semantic correspondence claim'))
  for bi in range(300):boot.append(dict(model=model,bandwidth_scale=scale,replicate=bi,scope='HSIC withinRoot macro Rootbootstrap',normalized_HSIC=float(bb[bi])))
  for bi in range(199):nullrows.append(dict(model=model,bandwidth_scale=scale,replicate=bi,scope='withinRoot cyclicmean',normalized_HSIC=float(nullmean[bi])))
 # Task information: only10Root means; not 905independent tasks.
 taskkernel=(root_tasks[:,None]==root_tasks[None,:]).astype(float);taskobs=hsic_from_centered(center(kyroot),center(taskkernel));rrng=np.random.default_rng(4701);tasknull=[]
 for bi in range(199):
  pp=rrng.permutation(10);tasknull.append(hsic_from_centered(center(kyroot),center(taskkernel[np.ix_(pp,pp)]))['normalized_HSIC'])
 retention.append(dict(model=model,information='Task',scope='10Rootmean;2Roots/task',units=10,**taskobs,null_mean=float(np.mean(tasknull)),permutation_tail_fraction=float((1+(np.array(tasknull)>=taskobs['normalized_HSIC']-1e-12).sum())/200),interpretation='task-associated root differences, not physicaltask semantics'))
 for g,ix in zip(roots,rootix):
  ky,_=kernel(Y[ix]);labels=n['s'][ix];l=(labels[:,None]==labels[None,:]).astype(float);v=hsic_from_centered(center(ky),center(l));retention.append(dict(model=model,information='Stage',scope='withinRoot categoricalStage kernel',Root=g,task=n['task'][ix[0]],windows=len(ix),unique_stages=len(set(labels)),**v,interpretation='Sim internal source labels;Real stages not matched'))
 # Frozen-source global52 descriptors supply explicit simple paired rank associations, not physical correspondence.
 dx,dy=descriptor(n['x']),descriptor(y)
 for j,label in [(0,'frame0_mean'),(2,'frame0_active_fraction'),(32,'frame01_absolute_delta')]:
  aa,bb=dx[:,j],dy[:,j];rho=float(spearmanr(aa,bb).statistic) if np.std(aa)>0 and np.std(bb)>0 else None;retention.append(dict(model=model,information='paired_descriptor',scope='global descriptive, no windowindependent CI',descriptor=label,Spearman=rho,interpretation='association only, not correctsemantic mapping'))
 for g,ix in [('ALL',np.arange(905))]+list(zip(roots,rootix)):
  for stage in ['ALL']+sorted(set(n['s'][ix])):
   ii=ix if stage=='ALL' else ix[n['s'][ix]==stage];w=y[ii];delta=np.diff(w,axis=1);corr=safe_corr(w[:,:-1],w[:,1:]);mask=w>0;prev=mask[:,:-1];nxt=mask[:,1:];temporal.append(dict(model=model,Root=g,stage=stage,windows=len(ii),lag1_correlation=corr,mean_absolute_frame_delta=float(np.abs(delta).mean()),mean_window_range=float(np.ptp(w,axis=1).mean()),signed_frame_delta=float(delta.mean()),transition01=float((~prev&nxt).sum()/max((~prev).sum(),1)),transition10=float((prev&~nxt).sum()/max(prev.sum(),1)),interpretation='peradjacent retainedframe; NOT persecond rate'))
 print('DEPENDENCE',model,'seconds',round(time.time()-start,1),flush=True)
table(O/'hsic_metrics.csv',hsics);table(O/'hsic_root_diagnostics.csv',within);table(O/'hsic_permutation_nulls.csv',nullrows);table(O/'hsic_bootstrap_results.csv',boot);table(O/'task_stage_retention.csv',retention);table(O/'temporal_diagnostics.csv',temporal)
# Prove data-level control properties, not hypothetical expected scores.
controlchecks=[]
for model in candidate_paths():
 y=d[model];yp=d[model+'_PairShuffle'];assert sorted(sha_bytes(v) if False else hashlib.sha256(v.tobytes()).hexdigest() for v in y)==sorted(hashlib.sha256(v.tobytes()).hexdigest() for v in yp);assert np.array_equal(yp,y[pair]);controlchecks.append(dict(model=model,pair_output_multiset_identical=True,within_RootStage=True,frames_insidewindow_unchanged=True,temporal_shuffle_changes_values=int(np.any(y!=d[model+'_TemporalShuffle'],axis=(1,2)).sum()),temporal_reverse_NOT_always_detectable_by_lag_delta=True))
table(O/'control_construction_checks.csv',controlchecks);assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'dependence_complete.json',dict(seconds=time.time()-start,models=len(d),Root_count=10,seeds=[4701],no_C0=True,no_training=True));print('P47 DEPENDENCE COMPLETE',flush=True)
