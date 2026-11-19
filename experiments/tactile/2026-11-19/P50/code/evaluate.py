from audit_utils import *
from support47 import center,kernel,hsic_from_centered,neighbor_radii,prdc
import time
P48=Path('outputs/tactile_p48_unified_four_route_audit_2026-11-19')
def quantiles(x,g):
 w=weights(g);a=x.reshape(len(x),-1);v=a.reshape(-1);ww=np.repeat(w/a.shape[1],a.shape[1]);ix=v>0
 if not ix.any():return np.zeros(6)
 p=np.argsort(v[ix]);vv=v[ix][p];cw=np.cumsum(ww[ix][p]);return np.interp([.01,.1,.5,.9,.95,.99],cw/cw[-1],vv)
def structure(x,g):
 a=x.astype(float);w=weights(g);flat=a.reshape(len(a),-1);prev=a[:,:-1].reshape(len(a),-1);nxt=a[:,1:].reshape(len(a),-1);mu=w@prev.mean(1);mv=w@nxt.mean(1);varu=w@((prev-mu)**2).mean(1);varv=w@((nxt-mv)**2).mean(1);den=np.sqrt(varu*varv);lag=float(w@((prev-mu)*(nxt-mv)).mean(1)/den) if den>0 else 0.;mask=a>0;before=mask[:,:-1];after=mask[:,1:];p01=float(w@((~before)&after).mean((1,2))/max(float(w@(~before).mean((1,2))),1e-15));p10=float(w@(before&(~after)).mean((1,2))/max(float(w@before.mean((1,2))),1e-15));runs=np.zeros((len(a),712),int);longest=runs.copy()
 for t in range(4):runs=(runs+1)*mask[:,t];longest=np.maximum(longest,runs)
 act=np.einsum('n,ntj->j',w,mask)/4;unique=len({row.tobytes() for row in x});stdmean=float(flat.std(0).mean());posq=quantiles(x,g);mean=float(w@flat.mean(1));var=float(w@(flat**2).mean(1)-mean**2);out=dict(zero_fraction=float(w@(flat==0).mean(1)),mean=mean,variance=var,lag1=lag,frame_delta=float(w@np.abs(np.diff(a,axis=1)).mean((1,2))),transition01=p01,transition10=p10,mean_longest_contact_frames=float(w@longest.mean(1)),unique_windows=unique,window_std=stdmean,maximum=float(a.max()),finite=bool(np.isfinite(a).all()),nonnegative=bool((a>=0).all()))
 for j,p in enumerate(['P1','P10','P50','P90','P95','P99']):out['positive_'+p]=float(posq[j])
 return out,act,posq
class Evaluation:
 def __init__(self):
  self.r=real('validation');self.s=natural('validation');self.train=real('train');self.z=np.load(P46/'RealTrain_preprocessing.npz');self.sigma=float(self.z['sigma']);self.R=transform(self.r['x'],self.z['mean'],self.z['scale']);self.X=transform(self.s['x'],self.z['mean'],self.z['scale']);self.dirs=np.load(P48/'projection_directions.npy');self.Rp=self.R@self.dirs;self.wr=weights(self.r['g']);self.ws=weights(self.s['g']);self.rgroups=np.unique(self.r['g']);self.sgroups=np.unique(self.s['g']);self.boot=np.load(P48/'group_bootstrap_indices.npz');bank=np.load(P47/'index_banks.npz');self.pair=bank['pair_permutation'];self.samples=bank['equal_indices'];self.realDD=cdist(self.R,self.R);self.realrad=neighbor_radii(self.realDD);self.trainstats,self.trainact,self.trainq=structure(self.train['x'],self.train['g']);self.realstats,_,_=structure(self.r['x'],self.r['g']);self.dist=[];self.struct=[];self.root=[];self.patch=[];self.taxel=[];self.hs=[];self.pr=[];self.bs=[];self.gates=[];self.cache={};self.base=None;self.negativeGates={};self.rid=np.array([np.flatnonzero(self.rgroups==g)[0] for g in self.r['g']]);self.sid=np.array([np.flatnonzero(self.sgroups==g)[0] for g in self.s['g']]);self.coeff=np.zeros((1000,14))
  for i in range(4):self.coeff[:,i]=(self.boot['Real']==i).sum(1)/4
  for i in range(10):self.coeff[:,4+i]=-(self.boot['Sim']==i).sum(1)/10
  self.point=np.r_[np.full(4,.25),np.full(10,-.1)]
 def distribution(self,Y):
  # Exact weighted 1D CDF integrations on fixed1024directions; sameP48contract.
  P=Y@self.dirs;values=np.r_[self.Rp,P].T;order=np.argsort(values,axis=1,kind='stable');sv=np.take_along_axis(values,order,1);dt=np.diff(sv,axis=1);tag=np.r_[self.rid,4+self.sid][order];cdf=np.empty((14,1024*(len(self.R)+len(Y)-1)))
  for j in range(14):cdf[j]=((tag==j).cumsum(1)[:,:-1]/np.sum(np.r_[self.rid,4+self.sid]==j)).reshape(-1)
  d=dt.reshape(-1);swd=float((np.abs(self.point@cdf)*d).sum()/1024);swboot=np.empty(1000)
  for j in range(0,1000,8):swboot[j:j+8]=(np.abs(self.coeff[j:j+8]@cdf)@d)/1024
  del cdf
  Z=np.r_[self.R,Y];dd=cdist(Z,Z,'sqeuclidean');gram=np.mean([np.exp(-dd/(2*(self.sigma*a)**2)) for a in [.5,1,2]],axis=0);mmd=float(np.sqrt(max(np.r_[self.wr,-self.ws]@gram@np.r_[self.wr,-self.ws],0)));indices=[np.flatnonzero(self.rid==i) for i in range(4)]+[len(self.R)+np.flatnonzero(self.sid==i) for i in range(10)];gg=np.array([[gram[np.ix_(a,b)].mean() for b in indices] for a in indices]);mmboot=np.sqrt(np.maximum(np.einsum('bi,ij,bj->b',self.coeff,gg,self.coeff),0));return swd,mmd,swboot,mmboot
 def dependency(self,y,name):
  Y=transform(y,self.z['mean'],self.z['scale']);vals={};diffboot=None;primary=None;pairprimary=None
  for scale in [.5,1.,2.]:
   orig=[];shuffle=[]
   for g in self.sgroups:
    ix=np.flatnonzero(self.s['g']==g);xx,_=kernel(self.X[ix],scale);yy,_=kernel(Y[ix],scale);yp,_=kernel(Y[self.pair[ix]],scale);a=hsic_from_centered(center(xx),center(yy));b=hsic_from_centered(center(xx),center(yp));orig.append(a['normalized_HSIC']);shuffle.append(b['normalized_HSIC']);self.hs.append(dict(model=name,scope='Root',Root=g,scale=scale,HSIC=a['HSIC'],normalized_HSIC=a['normalized_HSIC'],shuffle_HSIC=b['normalized_HSIC']))
   orig=np.array(orig);shuffle=np.array(shuffle);delta=orig-shuffle;bb=delta[self.boot['Sim']].mean(1);self.hs.append(dict(model=name,scope='RootMacro',scale=scale,normalized_HSIC=float(orig.mean()),shuffle_HSIC=float(shuffle.mean()),difference=float(delta.mean()),difference_CI_low=float(np.quantile(bb,.025)),difference_CI_high=float(np.quantile(bb,.975)),physical_semantics_proven=False));vals[scale]=float(orig.mean())
   if scale==1.:primary=float(orig.mean());pairprimary=float(shuffle.mean());diffboot=bb
  return primary,pairprimary,float(np.quantile(diffboot,.025))
 def prdc(self,Y,name):
  dd=cdist(Y,Y);xy=cdist(self.R,Y);result={}
  for j,k in enumerate([3,5,10,20]):
   vals=[]
   for ii in self.samples:frad=neighbor_radii(dd[np.ix_(ii,ii)])[:,j];vals.append(prdc(xy[:,ii],self.realrad[:,j],frad,k))
   result[k]={key:float(np.mean([v[key] for v in vals])) for key in ['precision','recall','density','coverage']};self.pr.append(dict(model=name,k=k,**result[k],draws=100,MC_draws_not_independent=True,scope='sameP47uniformwindow reference andRootproportional samples;not mislabeledgroupweighted'))
  return result
 def evaluate(self,name,y,control=False):
  if name in self.cache:return self.cache[name]
  assert y.shape==(905,4,712) and np.isfinite(y).all() and (y>=0).all();t=time.time();Y=transform(y,self.z['mean'],self.z['scale']);swd,mmd,swb,mmb=self.distribution(Y);hsic,pairhs,difflo=self.dependency(y,name);pr=self.prdc(Y,name);stats,act,q=structure(y,self.s['g']);self.struct.append(dict(model=name,group='ALL',**stats));self.dist.append(dict(model=name,SWD=swd,SWD_CI_low=float(np.quantile(swb,.025)),SWD_CI_high=float(np.quantile(swb,.975)),MMD=mmd,MMD_CI_low=float(np.quantile(mmb,.025)),MMD_CI_high=float(np.quantile(mmb,.975))))
  for i in range(1000):self.bs.append(dict(route='distribution',model=name,replicate=i,SWD=float(swb[i]),MMD=float(mmb[i])))
  for g in self.sgroups:
   ix=self.s['g']==g;st,_,_=structure(y[ix],self.s['g'][ix]);self.struct.append(dict(model=name,group=g,task=self.s['task'][ix][0],**st))
  for stage in np.unique(self.s['s']):
   ix=self.s['s']==stage;st,_,_=structure(y[ix],self.s['g'][ix]);self.struct.append(dict(model=name,group='Stage:'+stage,**st,scope='Sim labelonly;not Real physicalstage pairing'))
  OFF=np.cumsum([0]+[32,32,56]*5+[112]);mass=np.stack([y[:,:,l:h].sum(-1) for l,h in zip(OFF[:-1],OFF[1:])],-1);source=np.stack([self.s['x'][:,:,l:h].sum(-1) for l,h in zip(OFF[:-1],OFF[1:])],-1);shares=mass/np.maximum(mass.sum(-1,keepdims=True),1e-12);ss=source/np.maximum(source.sum(-1,keepdims=True),1e-12);inputerror=float(self.ws@((shares-ss)**2).sum(-1).mean(-1));spatial=float(np.mean((act-self.trainact)**2));tail=float(np.mean(np.abs(np.log1p(q)-np.log1p(self.trainq))));temporal=abs(stats['lag1']-self.trainstats['lag1'])+abs(np.log1p(stats['frame_delta'])-np.log1p(self.trainstats['frame_delta']))+abs(stats['transition01']-self.trainstats['transition01'])+abs(stats['transition10']-self.trainstats['transition10']);zero=abs(stats['zero_fraction']-self.trainstats['zero_fraction'])
  for j,(l,h) in enumerate(zip(OFF[:-1],OFF[1:])):self.patch.append(dict(model=name,Patch=j,active_fraction=float(self.ws@(y[:,:,l:h]>0).mean((1,2))),empty_fraction=float(self.ws@(y[:,:,l:h].sum(-1)==0).mean(1)),mean_response=float(self.ws@y[:,:,l:h].mean((1,2))),physical_mapping='indexspace proxy only'))
  self.taxel.extend(dict(model=name,taxel=j,activation=float(a)) for j,a in enumerate(act))
  result=dict(model=name,SWD=swd,MMD=mmd,HSIC=hsic,shuffle_HSIC=pairhs,shuffle_difference_CI_low=difflo,PRDC=pr,structure=stats,tail_error=tail,temporal_error=temporal,spatial_error=spatial,input_patch_error=inputerror,zero_error=zero)
  if self.base is None:self.base=result
  b=self.base;gates=dict(distribution=swd<=b['SWD']*1.1+1e-8 and mmd<=b['MMD']*1.1+1e-8,PRDC=all(pr[k][v]>=b['PRDC'][k][v]-.02 for k in pr for v in ['precision','recall','density','coverage']),dependency=hsic>=b['HSIC']*.8 and difflo>0,diversity=stats['window_std']>=b['structure']['window_std']*.5 and stats['unique_windows']>=b['structure']['unique_windows']*.9,amplitude=tail<=b['tail_error']*1.1+1e-8 and stats['maximum']<=b['structure']['maximum']*1.5+1e-8,temporal=temporal<=b['temporal_error']*1.1+1e-8,spatial=spatial<=b['spatial_error']*1.1+1e-8 and inputerror<=b['input_patch_error']*1.1+1e-8,zero_structure=zero<=b['zero_error']*1.1+1e-8);result['gates']=gates;result['all_quality_pass']=all(gates.values());result['severe_collapse']=stats['window_std']<b['structure']['window_std']*.1 or hsic<b['HSIC']*.2;self.gates.append(dict(model=name,**gates,all_quality_pass=result['all_quality_pass'],severe_collapse=result['severe_collapse'],reference='fixedF0A_K1,predeclaredtolerances;notuniversalstandards'));self.cache[name]=result;self.flush();print('QUALITY',name,'SWD',swd,'MMD',mmd,'HSIC',hsic,'gates',gates,'seconds',round(time.time()-t,1),flush=True);return result
 def flush(self):
  table(O/'distribution_metrics.csv',self.dist);table(O/'structure_metrics.csv',self.struct);table(O/'HSIC_metrics.csv',self.hs);table(O/'PRDC_metrics.csv',self.pr);table(O/'patch_metrics.csv',self.patch);table(O/'taxel_metrics.csv',self.taxel);table(O/'quality_gates.csv',self.gates);table(O/'quality_bootstrap.csv',self.bs)
