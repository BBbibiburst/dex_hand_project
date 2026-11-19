from support47 import *
import time
assert (O/'dependence_complete.json').exists();start=time.time();n,d=inputs();rv=real('validation');prep=np.load(P46/'RealTrain_preprocessing.npz');R=transform(rv['x'],prep['mean'],prep['scale']);sigma=float(prep['sigma']);pair=np.load(O/'index_banks.npz')['pair_permutation'];aa=cdist(R,R,'sqeuclidean');rows=[]
for model in candidate_paths():
 Y=transform(d[model],prep['mean'],prep['scale']);bb=cdist(Y,Y,'sqeuclidean');ab=cdist(R,Y,'sqeuclidean');before=[];after=[]
 for sc in [.5,1,2]:
  rr=np.exp(-aa/(2*(sigma*sc)**2));ff=np.exp(-bb/(2*(sigma*sc)**2));rf=np.exp(-ab/(2*(sigma*sc)**2));before.append(rr.mean()+ff.mean()-2*rf.mean());after.append(rr.mean()+ff[np.ix_(pair,pair)].mean()-2*rf[:,pair].mean())
 eb=float(2*np.sqrt(ab).mean()-np.sqrt(aa).mean()-np.sqrt(bb).mean());ea=float(2*np.sqrt(ab[:,pair]).mean()-np.sqrt(aa).mean()-np.sqrt(bb[np.ix_(pair,pair)]).mean());err=max(float(np.max(np.abs(np.array(before)-after))),abs(eb-ea));assert err<1e-10;rows.append(dict(model=model,MMD2V_before=float(np.mean(before)),MMD2V_after=float(np.mean(after)),Energy_V_before=eb,Energy_V_after=ea,max_abs_error=err,full905_multiset_invariant=True));print('MARGINAL_INVARIANCE',model,round(time.time()-start,1),flush=True)
table(O/'pair_marginal_invariance.csv',rows)
wr=list(csv.DictReader((O/'hsic_root_diagnostics.csv').open()));roots=sorted(set(n['g']));tasks=[n['task'][n['g']==g][0] for g in roots];rng=np.random.default_rng(4701);bank=np.concatenate([rng.choice(np.flatnonzero(np.array(tasks)==t),(300,2),replace=True) for t in sorted(set(tasks))],axis=1);paired=[]
for model in candidate_paths():
 for sc in [.5,1.,2.]:
  vals=[]
  for g in roots:
   def get(m):return float(next(r['normalized_HSIC'] for r in wr if r['model']==m and r['scope']=='withinRoot' and r['Root']==g and float(r['bandwidth_scale'])==sc))
   vals.append(get(model)-get(model+'_PairShuffle'))
  delta=np.array(vals);bd=delta[bank].mean(1);paired.append(dict(model=model,bandwidth_scale=sc,mean_original_minus_shuffle=float(delta.mean()),Root_boot_CI_low=float(np.quantile(bd,.025)),Root_boot_CI_high=float(np.quantile(bd,.975)),independent_Roots=10,bootstrap=300,multiple_comparison_adjusted=False,semantic_correctness_proven=False))
table(O/'paired_dependence_differences.csv',paired)
# Target temporal summaries: do not imply cross-domain physicaltime alignment.
a=rv['x'];trows=[]
for g,ix in [('ALL',np.arange(len(a)))]+[(g,np.flatnonzero(rv['g']==g)) for g in sorted(set(rv['g']))]:
 w=a[ix];trows.append(dict(dataset='RealValidation',Session=g,windows=len(ix),lag1_correlation=safe_corr(w[:,:-1],w[:,1:]),mean_absolute_frame_delta=float(np.abs(np.diff(w,axis=1)).mean()),mean_window_range=float(np.ptp(w,axis=1).mean()),time='irregular retainedframe indices;not persecond'))
table(O/'Real_temporal_reference.csv',trows)
# Actual pairing, root identity and source timestamps, without dereferencing new numeric files.
alignment=list(csv.DictReader((O/'alignment_audit.csv').open()))
for model,p in candidate_paths().items():
 z=np.load(p);alignment.append(dict(model=model,scope='complete input-output alignment',windows=len(z['values']),Root_count=len(np.unique(z['groups'])),same_window_ID=np.array_equal(z['units'],n['u']),same_Root=np.array_equal(z['groups'],n['g']),same_Stage=np.array_equal(z['stage'],n['s']),role='Validation',path=str(p)))
table(O/'alignment_audit.csv',alignment)
assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'control_verification_complete.json',dict(seconds=time.time()-start,MMD_Energy_pair_invariance=True,paired_Root_intervals=True,models=12));print('P47 CONTROL VERIFICATION COMPLETE',flush=True)
