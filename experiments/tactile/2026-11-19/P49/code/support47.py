from common import *
P46=Path('outputs/tactile_p46_real_variability_envelope_audit_2026-11-18')
KS=[3,5,10,20]
def datasets():
 n,d=inputs();r=real('train');bank=np.load(O/'index_banks.npz');d['Constant']=np.broadcast_to(r['x'].mean(0),(905,4,712)).copy();d['RealDonor']=r['x'][bank['donor_indices']]
 base=list(d)
 for name in base:
  if name not in ['Constant','RealDonor']:d[name+'_TemporalShuffle']=d[name][:,[0,2,1,3],:]
 for name in ['Original','A_K1']:d[name+'_TemporalReverse']=d[name][:,::-1,:]
 return n,d

def descriptor(x):
 a=np.asarray(x,dtype=np.float64);f=[a.mean(-1),a.std(-1),(a>0).mean(-1)]
 for q in [.5,.75,.9,.95,.99]:f.append(np.quantile(a,q,axis=-1))
 fs=np.stack(f,-1).reshape(len(a),32);delta=np.abs(np.diff(a,axis=1));dt=np.stack([delta.mean(-1),delta.std(-1),np.quantile(delta,.5,axis=-1),np.quantile(delta,.9,axis=-1)],-1).reshape(len(a),12);whole=[]
 for w in a:
  pos=w[w>0];whole.append([w.mean(),w.std(),(w>0).mean()]+(np.quantile(pos,[.1,.25,.5,.75,.9]).tolist() if len(pos) else [0.]*5))
 return np.c_[fs,dt,np.array(whole)]
def representations(r,v,d):
 z=np.load(P46/'RealTrain_preprocessing.npz');dr=descriptor(r['x']);mean=dr.mean(0);scale=np.maximum(dr.std(0),.05);np.savez(O/'descriptor_Train_preprocessing.npz',mean=mean,scale=scale);return dict(full2848=(transform(v['x'],z['mean'],z['scale']),{k:transform(x,z['mean'],z['scale']) for k,x in d.items()},transform(r['x'],z['mean'],z['scale'])),descriptor52=((descriptor(v['x'])-mean)/scale,{k:(descriptor(x)-mean)/scale for k,x in d.items()},(dr-mean)/scale))
def neighbor_radii(dd):
 assert len(dd)>max(KS);return np.partition(dd,KS,axis=1)[:,KS]
def contributions(cross,rr,ff,k):
 hit=cross<rr[:,None];return dict(precision=hit.any(0).astype(float),recall=(cross<ff[None,:]).any(1).astype(float),density=hit.sum(0)/float(k),coverage=(cross.min(1)<rr).astype(float))
def prdc(cross,rr,ff,k):return {key:float(v.mean()) for key,v in contributions(cross,rr,ff,k).items()}
def center(k):return k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean()
def bandwidth(d):
 p=d[np.triu_indices(len(d),1)];p=p[p>0];return float(np.sqrt(np.median(p))) if len(p) else None
def kernel(x,scale=1):
 dd=cdist(x,x,'sqeuclidean');sig=bandwidth(dd)
 if sig is None:return np.ones_like(dd),None
 return np.exp(-dd/(2*(sig*scale)**2)),sig

def hsic_from_centered(k,l):
 numerator=float((k*l).sum());den=float(np.sqrt((k*k).sum()*(l*l).sum()));return dict(HSIC=max(0.,numerator/(len(k)-1)**2),normalized_HSIC=max(0.,numerator/den) if den>1e-20 else 0.,constant_kernel=den<=1e-20)
def safe_corr(a,b):
 a=np.ravel(a).astype(float);b=np.ravel(b).astype(float);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den>1e-20 else None
