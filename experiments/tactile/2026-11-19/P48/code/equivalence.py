from audit_utils import *
import time
assert (O/'register_complete.json').exists() and not (O/'equivalence_complete.json').exists();start=time.time();r=real('train');rv=real('validation');s,out=inputs();bank=np.load(P47/'index_banks.npz');rt=r['x'];out['Constant']=np.broadcast_to(rt.mean(0),(905,4,712));out['RealDonor']=rt[bank['donor_indices']];z=np.load(P46/'RealTrain_preprocessing.npz');sigma=float(z['sigma']);rng=np.random.default_rng(4804);gs=np.unique(r['g']);rg=np.unique(rv['g']);sg=np.unique(s['g']);trainids=np.stack([rng.choice(np.flatnonzero(r['g']==g),7,replace=False) for g in gs]);rids=np.stack([rng.choice(np.flatnonzero(rv['g']==g),7,replace=False) for g in rg]);sids=np.stack([rng.choice(np.flatnonzero(s['g']==g),7,replace=False) for g in sg]);np.savez(O/'equivalence_window_bank.npz',RealTrain_groups=gs,RealTrain_indices=trainids,RealVal_groups=rg,RealVal_indices=rids,Sim_groups=sg,Sim_indices=sids)
def gram(x,y=None):
 dd=cdist(x,x if y is None else y,'sqeuclidean');return np.mean([np.exp(-dd/(2*(sigma*a)**2)) for a in [.5,1,2]],axis=0)
def grouped_gram(x,G):
 g=gram(x);return g.reshape(G,7,G,7).mean((1,3))
R=feature(r['x']).reshape(len(r['x']),-1).astype(float);G=grouped_gram(R[trainids.reshape(-1)],32);del R
# Primary1000 Session-disjoint4v10 partitions; sevenwindow bank fixed perSession.
calgroups=np.stack([rng.permutation(32)[:14] for _ in range(1000)]);cw=np.r_[np.full(4,.25),np.full(10,-.1)];vals=np.sqrt(np.maximum(np.einsum('i,bij,j->b',cw,G[calgroups[:,:,None],calgroups[:,None,:]],cw),0));eps=float(np.quantile(vals,.95));rrrows=[dict(draw=i,Real_A=json.dumps(gs[ix[:4]].tolist()),Real_B=json.dumps(gs[ix[4:]].tolist()),MMD=float(v),windows_pergroup=7,independent_total_Sessions=32,draws_are_not_independent=True) for i,(ix,v) in enumerate(zip(calgroups,vals))];table(O/'equivalence_rr_calibration.csv',rrrows)
# True Session-cluster bootstrap: resample32identities, then reestimate q95 of1000disjoint partitions in each empirical population.
rng=np.random.default_rng(4805);epsboot=np.full(1000,np.nan);distinct=[]
for b in range(1000):
 counts=np.bincount(rng.integers(0,32,32),minlength=32);distinct.append(int((counts>0).sum()))
 if (counts>0).sum()<14:continue
 # Exponential-race weighted sampling without replacement; absence of a Session forbids selecting it.
 keys=-np.log(np.maximum(rng.random((1000,32)),1e-300))/np.where(counts>0,counts,np.nan)[None,:];keys[:,counts==0]=np.inf;ix=np.argsort(keys,axis=1)[:,:14];vv=np.sqrt(np.maximum(np.einsum('i,bij,j->b',cw,G[ix[:,:,None],ix[:,None,:]],cw),0));epsboot[b]=np.quantile(vv,.95)
valid=np.isfinite(epsboot);stable=bool(valid.mean()>=.95 and eps>0 and np.nanquantile(epsboot,.025)>0);eblo=float(np.nanquantile(epsboot,.025));ebhi=float(np.nanquantile(epsboot,.975));np.savez(O/'RR_calibration_bootstrap.npz',epsilon=epsboot,unique_Sessions=distinct);save(O/'equivalence_calibration_summary.json',dict(epsilon_RR=eps,CI_low=eblo,CI_high=ebhi,CI_relative_width=(ebhi-eblo)/eps,valid_bootstraps=int(valid.sum()),valid=stable,independent_Sessions=32,partitions=1000,partition_draws_not_independent=True,fixed7windowbank=True,does_not_certify_physical_exchangeability=True))
rvf=feature(rv['x']).reshape(len(rv['x']),-1).astype(float)[rids.reshape(-1)];bg=np.load(O/'group_bootstrap_indices.npz');coeff=np.zeros((1000,14))
for i in range(4):coeff[:,i]=(bg['Real']==i).sum(1)/4
for i in range(10):coeff[:,4+i]=-(bg['Sim']==i).sum(1)/10
rows=[];bootrows=[]
for name,x in out.items():
 xf=feature(x).reshape(len(x),-1).astype(float)[sids.reshape(-1)];GG=grouped_gram(np.r_[rvf,xf],14);mmd=float(np.sqrt(max(cw@GG@cw,0)));vv=np.sqrt(np.maximum(np.einsum('bi,ij,bj->b',coeff,GG,coeff),0));delta=vv-epsboot;lo=float(np.nanquantile(delta,.025));hi=float(np.nanquantile(delta,.975));rows.append(dict(model=name,epsilon_RR=eps,MMD=mmd,Delta_eq=mmd-eps,Delta_CI_low=lo,Delta_CI_high=hi,calibration_valid=stable,marginal_equivalent=bool(stable and hi<0),scope='fixed7window pergroup;development marginal only,not physical/conditional equivalence'))
 for b in range(1000):bootrows.append(dict(route='Equivalence',model=name,replicate=b,MMD=float(vv[b]),epsilon=float(epsboot[b]) if valid[b] else None,Delta_eq=float(delta[b]) if valid[b] else None))
 print('EQUIVALENCE',name,mmd,'epsilon',eps,'CI',lo,hi,flush=True)
# P47 permutation stayswithinRoot so inverse permutation of7window indices leaves the same marginal sample.
for name in candidate_paths():rows.append(dict(next(r for r in rows if r['model']==name),model=name+'_PairShuffle',sampling='inverse permutation ofsevenbank to preserve identical outputsubset; verifiedP47 multiset invariance'))
table(O/'equivalence_metrics.csv',rows);table(O/'equivalence_bootstrap.csv',bootrows);assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'equivalence_complete.json',dict(seconds=time.time()-start,calibration_valid=stable,epsilon=eps,bootstrap_includes_calibration=True,SessionRoot_exchangeability_NOT_certified=True));print('P48 EQUIVALENCE COMPLETE',flush=True)
