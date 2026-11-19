from audit_utils import *
import torch,time,sys
from torch import nn
from torch.nn import functional as F
assert (O/'register_complete.json').exists()
# AST-import exact original architecture definitions, never importing original C0 fitting modules.
env=dict(torch=torch,nn=nn)
for file,classname in [('apps/evaluate_tactile_712d_gan.py','Probe'),('outputs/tactile_p35_deep_probe_2026-11-15/code/run.py','DeepProbe')]:
 tree=ast.parse(Path(file).read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname],type_ignores=[]),file,'exec'),env)
DeepProbe=env['DeepProbe'];torch.set_num_threads(2)
# Fixed synthetic timing only, no research-data selection of device/config.
bench=[]
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
 for kind in ['mlp','gru']:
  model=DeepProbe(kind).to(device);x=torch.randn(64,4,712,device=device);opt=torch.optim.AdamW(model.parameters(),lr=.001);t=time.time()
  for _ in range(20):opt.zero_grad();model(x).square().mean().backward();opt.step()
  if device=='cuda':torch.cuda.synchronize()
  bench.append(dict(device=device,kind=kind,seconds=time.time()-t));del model,x,opt
save(O/'probe_device_benchmark.json',bench);device=min(set(v['device'] for v in bench),key=lambda d:sum(v['seconds'] for v in bench if v['device']==d));print('PROBE_DEVICE',device,bench,flush=True)
# Wait-free dependency guard; orchestration launches after frozen outputs complete.
assert (O/'generate_complete.json').exists();start=time.time();reg=json.loads((O/'registration.json').read_text());r=real('train');rv=real('validation');s=natural('train');sv=natural('validation');yr=np.ones(len(r['x']));ys=np.zeros(len(s['x']));y=np.r_[yr,ys];yv=np.r_[np.ones(len(rv['x'])),np.zeros(len(sv['x']))];w=np.r_[.5*weights(r['g']),.5*weights(s['g'])];wv=np.r_[.5*weights(rv['g']),.5*weights(sv['g'])];N=len(y);trainrx=feature(r['x']);valrx=feature(rv['x']);audits=[];folder=O/'probes';folder.mkdir(exist_ok=True)
def predict(model,x):
 model.eval();vv=[]
 with torch.inference_mode():
  for i in range(0,len(x),256):vv.append(model(torch.from_numpy(x[i:i+256]).to(device)).cpu().numpy())
 return np.concatenate(vv).astype(float)
for name in reg['models']:
 x=np.concatenate([trainrx,feature(candidate('train',name))]);xv=np.concatenate([valrx,feature(candidate('validation',name))]);dest=folder/name;dest.mkdir(exist_ok=True)
 for kind,seed in [('logistic',0)]+[(kind,seed) for kind in ['mlp','gru'] for seed in [11,12,13]]:
  ident=kind if kind=='logistic' else f'{kind}_{seed}';p=dest/ident
  if (p/'complete.json').exists():continue
  if name!='Original':
   basefolder=folder/'Original'/ident;baseinfo=json.loads((basefolder/'complete.json').read_text());bp=np.load(basefolder/'predictions.npz');bl=bp['validation_logits'];by=bp['validation_labels'];bloss=np.logaddexp(0,bl)-by*bl;jb,jzero=bootstrap_scores(bloss[:780],bloss[780:],bloss[:780],bloss[780:],rv['g'],sv['g']);baseok=baseinfo['valid_for_JSD_claim'] and baseinfo['Validation']['JSD_proxy']>.01 and np.quantile(jzero,.025)>0
   if not baseok:
    p.mkdir(exist_ok=True);save(p/'complete.json',dict(model=name,instance=ident,status='NOT_TRAINED_BASELINE_INVALID',epochs=0,optimizer_steps=0,valid_for_JSD_claim=False,reason='predeclared Original validity gate; no candidate fit',baseline_source=str(basefolder/'complete.json')));continue
  p.mkdir(exist_ok=True);t=time.time();hist=[];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  if kind=='logistic':
   xx=torch.tensor(x.reshape(N,-1),dtype=torch.float64);yy=torch.tensor(y,dtype=torch.float64);ww=torch.tensor(w,dtype=torch.float64);weight=torch.zeros(2848,dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([weight,bias],max_iter=1000,max_eval=1500,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12);calls=[0]
   def closure():
    opt.zero_grad();loss=(F.binary_cross_entropy_with_logits(xx@weight+bias,yy,reduction='none')*ww).sum()+weight.square().sum()/(2*N);loss.backward();calls[0]+=1;return loss
   opt.step(closure);finalobjective=float(closure().detach());gn=float(max(weight.grad.abs().max(),bias.grad.abs()));steps=int(opt.state[weight]['n_iter']);converged=gn<=1e-6 and steps<1000
   tl=(xx.detach().numpy()@weight.detach().numpy()+bias.item());vl=xv.reshape(len(xv),-1).astype(float)@weight.detach().numpy()+bias.item();torch.save(dict(weight=weight.detach(),bias=bias.detach(),iterations=steps,gradient_max=gn),p/'checkpoint.pt');hist=[dict(epoch=0,optimizer_steps=steps,train_CE=evalstats(y,tl,w)['CE'],validation_CE=evalstats(yv,vl,wv)['CE'],objective=finalobjective,gradient_max=gn)];epochs=0;parameters=2849;del xx,yy,ww,weight,bias,opt
  else:
   model=DeepProbe(kind).to(device);parameters=sum(v.numel() for v in model.parameters());tx=torch.from_numpy(x).to(device);ty=torch.tensor(y,dtype=torch.float32,device=device);tw=torch.tensor(w*N,dtype=torch.float32,device=device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);steps=0
   for epoch in range(1,41):
    model.train();order=np.random.default_rng(seed+epoch).permutation(N);seen=np.zeros(N,bool)
    for offset in range(0,N,64):
     ix=order[offset:offset+64];seen[ix]=True;opt.zero_grad();loss=(F.binary_cross_entropy_with_logits(model(tx[ix]),ty[ix],reduction='none')*tw[ix]).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();steps+=1
    assert seen.all();tl=predict(model,x);vl=predict(model,xv);tr=evalstats(y,tl,w);va=evalstats(yv,vl,wv);hist.append(dict(epoch=epoch,optimizer_steps=steps,train_CE=tr['CE'],validation_CE=va['CE'],train_BA=tr['BA'],validation_BA=va['BA'],train_AUC=tr['AUC'],validation_AUC=va['AUC'],sample_visits=N,seconds=time.time()-t));table(p/'history.csv',hist)
   torch.save(dict(model=model.cpu().state_dict(),kind=kind,seed=seed,epoch=40),p/'checkpoint.pt');epochs=40;converged=np.ptp([h['train_CE'] for h in hist[-5:]])<=.02;gn=None;del model,opt,tx,ty,tw
   if torch.cuda.is_available():torch.cuda.empty_cache()
  assert np.isfinite(tl).all() and np.isfinite(vl).all();tr=evalstats(y,tl,w);va=evalstats(yv,vl,wv);mismatch=va['CE']-tr['CE']>.2;np.savez_compressed(p/'predictions.npz',train_logits=tl,validation_logits=vl,train_labels=y,validation_labels=yv,train_weights=w,validation_weights=wv,Real_validation_groups=rv['g'],Sim_validation_groups=sv['g']);table(p/'history.csv',hist);record=dict(model=name,instance=ident,kind=kind,seed=seed,epochs=epochs,optimizer_steps=steps,parameters=parameters,seconds=time.time()-t,Train_windows=N,RealTrain_Sessions=32,SimTrain_Roots=50,all_deep_epochs_full_coverage=True if epochs else None,device=device if epochs else 'CPUfloat64',converged=bool(converged),train_validation_mismatch=bool(mismatch),valid_for_JSD_claim=bool(converged and not mismatch),gradient_max=gn,Train=tr,Validation=va);save(p/'complete.json',record);audits.append(record);print('PROBE_COMPLETE',name,ident,'secs',round(time.time()-t,1),'TrainCE',round(tr['CE'],4),'ValCE',round(va['CE'],4),'valid',record['valid_for_JSD_claim'],flush=True)
save(O/'probes_complete.json',dict(seconds=time.time()-start,expected_instances=91,migration_epochs=0,migration_optimizer_steps=0,deep_epochs_each=40,deep_instances=78,Logistic_instances=13,validation_never_used_for_selection=True));print('P48 PROBES COMPLETE',flush=True)
