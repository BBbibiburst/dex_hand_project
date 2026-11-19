from audit_utils import *
import time
from scipy.stats import wasserstein_distance
assert (O/'generate_complete.json').exists() and not (O/'swd_complete.json').exists();start=time.time();reg=json.loads((O/'registration.json').read_text());rv=real('validation');nv,out=inputs();P=np.load(O/'projection_directions.npy');R=feature(rv['x']).reshape(len(rv['x']),-1).astype(float)@P;groupsR=np.unique(rv['g']);groupsS=np.unique(nv['g']);rg=np.array([np.flatnonzero(groupsR==g)[0] for g in rv['g']]);sg=np.array([np.flatnonzero(groupsS==g)[0] for g in nv['g']]);boot=np.load(O/'group_bootstrap_indices.npz');coef=np.zeros((1000,14))
for i in range(4):coef[:,i]=(boot['Real']==i).sum(1)/4
for i in range(10):coef[:,4+i]=-(boot['Sim']==i).sum(1)/10
point=np.r_[np.full(4,.25),np.full(10,-.1)];rows=[];bs=[];projected={};perdirection=[]
def swd_from_projected(Y,bootstrap=False):
 values=np.concatenate([R,Y]).T;order=np.argsort(values,axis=1,kind='stable');sortedv=np.take_along_axis(values,order,1);dt=np.diff(sortedv,axis=1);tags=np.r_[rg,4+sg][order];cdf=np.empty((14,len(P.T)*(len(values.T)-1)))
 for g in range(14):cdf[g]=((tags==g).cumsum(1)[:,:-1]/np.sum(np.r_[rg,4+sg]==g)).reshape(-1)
 # Accumulate exact 1D CDF integral on every common support interval; no quantile approximation.
 dd=dt.reshape(-1);diff=point@cdf;direction=(np.abs(diff).reshape(1024,-1)*dt).sum(1);result=float(direction.mean());bb=None
 if bootstrap:
  bb=np.empty(1000)
  for offset in range(0,1000,8):bb[offset:offset+8]=(np.abs(coef[offset:offset+8]@cdf)@dd)/1024
 else:del cdf
 return result,direction,bb
# Controls use identical P47 indices, no new donor search.
controlbank=np.load(P47/'index_banks.npz');rt=real('train')['x'];controls={'Constant':np.broadcast_to(rt.mean(0),(905,4,712)),'RealDonor':rt[controlbank['donor_indices']]};del rt
allvalues=dict(out,**controls)
for name,x in allvalues.items():
 Y=feature(x).reshape(len(x),-1).astype(float)@P;value,dd,bb=swd_from_projected(Y,True);projected[name]=Y
 # Independent exact scipycheck for first projection and declaredgroupweights.
 reference=wasserstein_distance(R[:,0],Y[:,0],weights(rv['g']),weights(nv['g']));assert abs(reference-dd[0])<1e-10
 if name=='Original':base=value;baseboot=bb.copy()
 reduction=1-value/base;rr=1-bb/baseboot;valid=float(np.quantile(baseboot,.025))>0
 rows.append(dict(model=name,SWD=value,Original_SWD=base,R_W=reduction,SWD_CI_low=float(np.quantile(bb,.025)),SWD_CI_high=float(np.quantile(bb,.975)),R_W_CI_low=float(np.quantile(rr,.025)),R_W_CI_high=float(np.quantile(rr,.975)),denominator_valid=valid,point90=bool(reduction>=.9 and valid),metric='mean1024 weightedprojectedW1;notfullW1',weights='equal4Session and10Root'))
 for j,v in enumerate(dd):perdirection.append(dict(model=name,direction=j,W1=float(v)))
 for j,(v,ratio) in enumerate(zip(bb,rr)):bs.append(dict(route='SWD',model=name,replicate=j,distance=float(v),reduction=float(ratio),baseline=float(baseboot[j])))
 print('SWD',name,value,'reduction',reduction,'seconds',round(time.time()-start,1),flush=True)
table(O/'metrics_swd.csv',rows);table(O/'swd_per_direction.csv',perdirection);table(O/'swd_bootstrap.csv',bs)
# Path analyses at each K within SAME Real-coordinate latent and SAME decoded outputspace.
paths=[];summaries=[];w=weights(nv['g']);orig=feature(nv['x']).reshape(905,-1).astype(float)
def quantile(v,weights_,qq):
 ii=np.argsort(v);return float(np.interp(qq,np.cumsum(weights_[ii])/weights_.sum(),v[ii]))
for group in ['A','B','C']:
 for K in [1,2,4,8]:
  z=np.load(O/f'paths/{group}_K{K}.npz');x=z['decoded'];trajectory=[];deltas=[];swd=[]
  for step in range(K+1):
   f=feature(x[:,step]).reshape(905,-1).astype(float);trajectory.append(f);value,_,_=swd_from_projected(f@P,False);swd.append(value);row=dict(model=group,K=K,step=step,time=step/K,SWD=value,step0='Dr(Real-coordinate zero latent);NOT Original')
   if step:
    delta=f-trajectory[-2];deltas.append(delta);norm=np.linalg.norm(delta,axis=1);rawnorm=np.linalg.norm((x[:,step]-x[:,step-1]).reshape(905,-1),axis=1);row.update(update_mean_L2=float(w@norm),update_P50=quantile(norm,w,.5),update_P90=quantile(norm,w,.9),update_P95=quantile(norm,w,.95),update_raw_mean_L2=float(w@rawnorm))
    if step>1:
     prior=deltas[-2];den=norm*np.linalg.norm(prior,axis=1);ok=den>1e-12;cos=np.sum(delta*prior,axis=1)[ok]/den[ok];row.update(direction_cosine=float(w[ok]@cos/w[ok].sum()) if ok.any() else None,direction_defined_mass=float(w[ok].sum()))
   paths.append(row)
  norms=np.stack([np.linalg.norm(v,axis=1) for v in deltas]);length=norms.sum(0);endpoint=np.linalg.norm(trajectory[-1]-trajectory[0],axis=1);ratio=length/np.maximum(endpoint,1e-12);increase=np.diff(swd);initial_jump=np.linalg.norm(trajectory[0]-orig,axis=1);summaries.append(dict(model=group,K=K,integration_interval='0to1',integration_start_SWD=swd[0],Original_SWD=base,final_SWD=swd[-1],best_intermediate_SWD=min(swd),final_minus_best=mincrease if False else swd[-1]-min(swd),distance_increase_steps=int((increase>1e-12).sum()),maximum_rebound=float(max(0,increase.max())),mean_PathLength=float(w@length),mean_EndpointDistance=float(w@endpoint),mean_PathRatio=float(w@ratio),PathRatio_P95=quantile(ratio,w,.95),zero_endpoint_mass=float(w[endpoint<=1e-12].sum()),source_to_Drzero_jump_mean=float(w@initial_jump),source_jump_is_not_Euler_step=True))
  print('PATH',group,K,'seconds',round(time.time()-start,1),flush=True)
table(O/'path_metrics.csv',paths);table(O/'path_summary.csv',summaries);assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'swd_complete.json',dict(seconds=time.time()-start,directions=1024,groupbootstrap=1000,exact_1D_verified=True,shared_field_no_training=True));print('P48 SWD/PATH COMPLETE',flush=True)
