from audit_utils import *
import time,shutil
assert (O/'generate_complete.json').exists();start=time.time();rv=real('validation');s,out=inputs();r=real('train');pbank=np.load(P47/'index_banks.npz');out['Constant']=np.broadcast_to(r['x'].mean(0),(905,4,712));out['RealDonor']=r['x'][pbank['donor_indices']];rows=[];neg=[];proof=[];pipeline=[]
def q(v,w,p):
 order=np.argsort(v);return float(np.interp(p,np.cumsum(w[order])/w.sum(),v[order]))
for name,x in dict(out,RealValidation=rv['x']).items():
 groups=rv['g'] if name=='RealValidation' else s['g'];w=weights(groups)
 for g,ix in [('ALL',np.arange(len(x)))]+[(g,np.flatnonzero(groups==g)) for g in np.unique(groups)]:
  a=x[ix].astype(float);ww=w[ix]/w[ix].sum();flat=a.reshape(len(a),-1);wm=np.repeat(ww/2848,2848);ff=flat.reshape(-1);pos=ff>0;mean=float(ww@flat.mean(1));variance=float(ww@(flat**2).mean(1)-mean**2);d=np.diff(a,axis=1);u=a[:,:-1].reshape(len(a),-1);v=a[:,1:].reshape(len(a),-1);pw=np.repeat(ww/u.shape[1],u.shape[1]);uf=u.reshape(-1);vf=v.reshape(-1);um=float(pw@uf);vm=float(pw@vf);den=np.sqrt(float(pw@(uf-um)**2)*float(pw@(vf-vm)**2));corr=float(pw@((uf-um)*(vf-vm))/den) if den>0 else None;row=dict(model=name,group=g,windows=len(ix),weights='equal groups,uniform windows within group',finite=bool(np.isfinite(a).all()),nonnegative=bool((a>=0).all()),zero_fraction=float(ww@(flat==0).mean(1)),mean=mean,variance=variance,mean_absolute_frame_delta=float(ww@np.abs(d).mean((1,2))),lag1_correlation=corr,unique_windows=len({z.tobytes() for z in x[ix]}),mean_across_window_std=float(flat.std(0).mean()),max_value=float(a.max()))
  for qq,label in zip([.01,.1,.5,.9,.95,.99],['P1','P10','P50','P90','P95','P99']):row['positive_'+label]=q(ff[pos],wm[pos],qq) if pos.any() else None
  rows.append(row)
# All preexisting P47 PRDC/HSIC use EXACT same registeredvalidation arrays, so reuseverifiedresults not change their weights silently.
pr=list(csv.DictReader((P47/'prdc_metrics.csv').open()));hs=list(csv.DictReader((P47/'hsic_metrics.csv').open()));pair=list(csv.DictReader((P47/'paired_dependence_differences.csv').open()))
for fname in ['prdc_metrics.csv','hsic_metrics.csv','paired_dependence_differences.csv','root_metrics.csv','negative_control_metrics.csv','control_construction_checks.csv']:
 shutil.copyfile(P47/fname,O/f'P47_reused_{fname}');proof.append(dict(path=str(P47/fname),sha256=sha(P47/fname),copy_sha=sha(O/f'P47_reused_{fname}'),meaning='existingP47contract,not rerunC0;PRDCwindow-proportional MC separate fromP48group-equal distribution metrics'))
# Full indexed pipeline path adds Original->Dr(h0), while Eulerpath summary excludes this nonintegrator jump.
orig=feature(s['x']).reshape(905,-1).astype(float);w=weights(s['g'])
for a in ['A','B','C']:
 for K in [1,2,4,8]:
  z=np.load(O/f'paths/{a}_K{K}.npz');y=z['decoded'];steps=[orig]+[feature(y[:,j]).reshape(905,-1).astype(float) for j in range(K+1)];length=sum(np.linalg.norm(steps[j+1]-steps[j],axis=1) for j in range(K+1));end=np.linalg.norm(steps[-1]-orig,axis=1);ratio=length/np.maximum(end,1e-12);pipeline.append(dict(model=a,K=K,mean_full_pipeline_length=float(w@length),mean_Original_to_final_distance=float(w@end),mean_full_pipeline_ratio=float(w@ratio),includes_source_to_Drzero=True,initial_jump_NOT_Euler_update=True))
for name in ['Constant','RealDonor']+[f'{a}_K{k}_PairShuffle' for a in ['A','B','C'] for k in [1,2,4,8]]:
 h=next(v for v in hs if v['model']==name and v['scope']=='withinRoot equalmacro' and v['bandwidth_scale']=='1.0');p=next(v for v in pr if v['model']==name and v['space']=='full2848' and v['k']=='5');neg.append(dict(model=name,PRDC_precision=p['precision'],PRDC_recall=p['recall'],PRDC_coverage=p['coverage'],Root_HSIC=h['normalized_HSIC'],HSIC_null_mean=h['null_mean'],formal_combined_pass='NOT_DEFINED inP47',semantic_correctness=False))
table(O/'quality_metrics.csv',rows);table(O/'negative_controls.csv',neg);table(O/'quality_reuse_provenance.csv',proof);table(O/'full_pipeline_path_metrics.csv',pipeline);save(O/'quality_complete.json',dict(seconds=time.time()-start,all12candidate_outputs_sameasP47=True,P47_thresholds='no combinedpass thresholds existed;none invented',no_new_neuralmodels=True));print('P48 QUALITY COMPLETE',flush=True)
