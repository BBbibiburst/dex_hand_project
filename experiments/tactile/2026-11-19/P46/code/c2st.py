from common import *
import time,warnings
import torch
from scipy.stats import rankdata
class ConvergenceWarning(Warning):pass
def confusion_matrix(y,p,labels=None):return np.array([[((y==a)&(p==b)).sum() for b in [0,1]] for a in [0,1]])
def balanced_accuracy_score(y,p):return .5*np.mean(p[y==0]==0)+.5*np.mean(p[y==1]==1)
def roc_auc_score(y,p):
 n1=(y==1).sum();n0=(y==0).sum();return (rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def log_loss(y,p,labels=None):
 p=np.clip(p,1e-15,1-1e-15);return -(y*np.log(p)+(1-y)*np.log(1-p)).mean()
class LogisticRegression:
 def __init__(self,**kw):self.kw=kw
 def fit(self,x,y):
  torch.set_num_threads(1);xx=torch.tensor(x,dtype=torch.float64);yy=torch.tensor(y,dtype=torch.float64);w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True);b=torch.zeros((),dtype=torch.float64,requires_grad=True);weights=torch.tensor(np.where(y==1,.5/(y==1).sum(),.5/(y==0).sum()),dtype=torch.float64);opt=torch.optim.LBFGS([w,b],lr=1.,max_iter=100,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe');calls=[0]
  def closure():
   opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(xx@w+b,yy,reduction='none')*weights).sum()+w.square().sum()/(2*len(y));loss.backward();calls[0]+=1;return loss
  opt.step(closure);state=opt.state[w];self.n_iter_=[state['n_iter']];self.coef_=w.detach().numpy()[None];self.intercept_=b.detach().numpy()[None];self.closure_calls=calls[0]
  if self.n_iter_[0]>=100:warnings.warn('LBFGS reached iteration cap; convergence not certified',ConvergenceWarning)
  return self
 def predict_proba(self,x):
  z=x@self.coef_[0]+self.intercept_[0];p=np.exp(-np.logaddexp(0,-z));return np.c_[1-p,p]
assert (O/'prepare_complete.json').exists() and not (O/'c2st_complete.json').exists()
start=time.time();r=real('train');n=natural('train');rng=np.random.default_rng(4601);gs=rng.permutation(np.unique(r['g']));pseudo={g:int(i>=16) for i,g in enumerate(gs)};real_folds={g:(i%16)//4 for i,g in enumerate(gs)};sim_folds={}
for task in sorted(set(n['task'])):
 groups=rng.permutation(np.unique(n['g'][n['task']==task]))
 for fold,part in enumerate(np.array_split(groups,4)):
  for g in part:sim_folds[g]=fold
save(O/'c2st_group_folds.json',dict(Real={str(g):dict(fold=int(real_folds[g]),pseudo_domain=int(pseudo[g])) for g in gs},Sim={str(g):int(f) for g,f in sim_folds.items()},seed=4601,roles='Train only',preprocessing='Real fit-fold only'))
metrics=[];predrows=[];boots=[];pooled={}
def score(y,p):
 ba=float(balanced_accuracy_score(y,p>=.5));return dict(BA=ba,AUC=float(roc_auc_score(y,p)),D=1-2*abs(ba-.5),loss=float(log_loss(y,p,labels=[0,1])),confusion_matrix=json.dumps(confusion_matrix(y,p>=.5,labels=[0,1]).tolist()))
for fold in range(4):
 ri=np.array([real_folds[g]!=fold for g in r['g']]);si=np.array([sim_folds[g]!=fold for g in n['g']]);rrlabels=np.array([pseudo[g] for g in r['g']]);mu=np.log1p(r['x'][ri]).reshape(ri.sum(),-1).mean(0);sd=np.maximum(np.log1p(r['x'][ri]).reshape(ri.sum(),-1).std(0),.05);R=transform(r['x'],mu,sd);N=transform(n['x'],mu,sd)
 for name in ['Real_vs_Real','OriginalSim_vs_Real']:
  if name=='Real_vs_Real':x,y,g,fit=R,rrlabels,r['g'],ri
  else:x=np.concatenate([R,N]);y=np.r_[np.ones(len(R),int),np.zeros(len(N),int)];g=np.r_[np.char.add('Real:',r['g']),np.char.add('Sim:',n['g'])];fit=np.r_[ri,si]
  t=time.time();model=LogisticRegression(C=1.,solver='lbfgs',max_iter=100,class_weight='balanced',random_state=4601,tol=1e-4)
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');model.fit(x[fit],y[fit])
  pt=model.predict_proba(x[fit])[:,1];pv=model.predict_proba(x[~fit])[:,1];yy=y[~fit];gg=g[~fit];val=score(yy,pv);train=score(y[fit],pt);np.savez_compressed(O/f'c2st_{name}_fold{fold}.npz',coef=model.coef_,intercept=model.intercept_,mean=mu,scale=sd,held_labels=yy,held_scores=pv,held_groups=gg)
  metrics.append(dict(comparison=name,fold=fold,status='COMPLETED',fit_windows=int(fit.sum()),held_windows=int((~fit).sum()),fit_groups=len(set(g[fit])),held_groups=len(set(gg)),parameters=x.shape[1]+1,iterations=int(model.n_iter_[0]),optimizer_closure_calls=model.closure_calls,max_iter=100,convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in ws),seconds=time.time()-t,**val,**{'Train_'+k:v for k,v in train.items()}))
  for i,(label,pred,group) in enumerate(zip(yy,pv,gg)):predrows.append(dict(comparison=name,fold=fold,row=i,label=int(label),score=float(pred),group=str(group)))
  # Cluster bootstrap within each label and held fold; no window independence assumption.
  brng=np.random.default_rng(4601+fold);classes={c:np.unique(gg[yy==c]) for c in [0,1]};br=[]
  for b in range(300):
   ix=np.concatenate([np.concatenate([np.flatnonzero(gg==group) for group in brng.choice(classes[c],len(classes[c]),replace=True)]) for c in [0,1]]);v=score(yy[ix],pv[ix]);br.append([v[k] for k in ['BA','AUC','D']]);boots.append(dict(comparison=name,fold=fold,replicate=b,**{k:v[k] for k in ['BA','AUC','D']}))
  pooled.setdefault(name,[]).append(np.array(br));print(name,fold,val,'seconds',round(time.time()-t,2),flush=True)
for name,arr in pooled.items():
 means=np.mean(arr,axis=0);rows=[v for v in metrics if v['comparison']==name];out=dict(comparison=name,fold='mean_four_folds',status='COMPLETED',interpretation='Train-internal group CV diagnostic; not original C0; bootstrap conditional on fitted models')
 for j,k in enumerate(['BA','AUC','D']):out[k]=float(np.mean([v[k] for v in rows]));out[k+'_CI_low']=float(np.quantile(means[:,j],.025));out[k+'_CI_high']=float(np.quantile(means[:,j],.975))
 metrics.append(out)
for name in candidate_paths():metrics.append(dict(comparison=name+'_vs_Real',status='NOT_AVAILABLE',reason='No existing Translated Train outputs. Validation never used to fit C2ST. No new generator inference authorized in P46.'))
table(O/'c2st_metrics.csv',metrics);table(O/'c2st_held_predictions.csv',predrows);table(O/'c2st_bootstrap.csv',boots);assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'c2st_complete.json',dict(seconds=time.time()-start,fits=8,Train_only=True,original_C0_untouched=True));print('P46 C2ST COMPLETE',flush=True)
