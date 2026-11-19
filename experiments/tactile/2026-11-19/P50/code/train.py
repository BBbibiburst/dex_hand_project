from audit_utils import *
from architecture import Transfer,MultiView,OFF
import torch,time,os
from torch.nn import functional as F
P49=Path('outputs/tactile_p49_round_frozen_adversarial_2026-11-19')
torch.set_num_threads(2);torch.manual_seed(5001);np.random.seed(5001);assert torch.cuda.is_available();device='cuda';reg=json.loads((O/'registration.json').read_text());(O/'checkpoints').mkdir(exist_ok=True);(O/'outputs').mkdir(exist_ok=True)
r=real('train');s=natural('train');rv=real('validation');sv=natural('validation');split=np.load(O/'inner_group_split.npz');rf=np.flatnonzero(np.isin(r['g'],split['Real_fit']));ri=np.flatnonzero(np.isin(r['g'],split['Real_inner']));sf=np.flatnonzero(np.isin(s['g'],split['Sim_fit']));si=np.flatnonzero(np.isin(s['g'],split['Sim_inner']));assert not set(r['g'][rf])&set(r['g'][ri]);assert not set(s['g'][sf])&set(s['g'][si]);assert not set(r['g'])&set(rv['g']);assert not set(s['g'])&set(sv['g'])
ck=torch.load(P45/'A/generator.pt',map_location='cpu',weights_only=False);G=Transfer(ck['scales']);G.base.load_state_dict(ck['model']);G.base.requires_grad_(False);G.base.field.requires_grad_(True);G.eval();Z={};baseline={}
with torch.inference_mode():
 for role,d in [('train',s),('validation',sv)]:
  z=torch.cat([G.base.sim.posterior_mean(torch.from_numpy(d['x'][i:i+64])) for i in range(0,len(d['x']),64)]);Z[role]=z
 l=torch.cat([G.base.real.posterior_mean(torch.from_numpy(r['x'][i:i+64])) for i in range(0,len(r['x']),64)])
scale=l.std((0,1),unbiased=False).clamp_min(.1).to(device);G.to(device);Z={k:v.to(device) for k,v in Z.items()};D=MultiView(json.loads((P44/'config.json').read_text())['D_log_scale_RealTrain']).to(device);dopt=torch.optim.Adam(D.parameters(),lr=1e-4,betas=(0.,.99));gp=list(G.base.field.parameters())+list(G.refiner.parameters());gopt=torch.optim.Adam(gp,lr=1e-4,betas=(0.,.99));used=0.;ds=gs=0;logs=[];gradientrows=[];done_phase='D';next_epoch=1;next_step=0
R=torch.from_numpy(r['x']).to(device);S=torch.from_numpy(s['x']).to(device);rw=torch.from_numpy(weights(r['g'][rf])*len(rf)).float().to(device);sw=torch.from_numpy(weights(s['g'][sf])*len(sf)).float().to(device);gw=torch.from_numpy(weights(s['g'])*len(s['x'])).float().to(device)
def shstate(m):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
vae_before={k:v.detach().cpu().clone() for k,v in G.base.state_dict().items() if k.startswith(('real.','sim.'))}
def generate(role):
 G.eval()
 with torch.inference_mode():y=torch.cat([G.from_latent(Z[role][i:i+64])[0].cpu() for i in range(0,len(Z[role]),64)]).numpy()
 assert np.isfinite(y).all() and (y>=0).all();return y
for role,d in [('train',s),('validation',sv)]:
 G.cpu();G.eval()
 with torch.inference_mode():baseline[role]=torch.cat([G.from_latent(Z[role][i:i+64].cpu())[0] for i in range(0,len(Z[role]),64)]).numpy()
 G.to(device);path=P49/f'outputs/F0_{role}.npz';old=np.load(path)['values'] if path.exists() else np.load(P45/'A/outputs/K1_validation.npz')['values'];err=float(np.abs(baseline[role]-old).max());assert np.array_equal(baseline[role],old)
 np.savez_compressed(O/f'outputs/F0_{role}.npz',values=baseline[role],groups=d['g'],stage=d['s'],units=d['u'],task=d['task']);print('F0 identity',role,err,flush=True)
# zero residual identity vs exact same base forward; each branch finite gradients to field AND refiner.
G.base.field.train();G.refiner.train();D.eval();D.requires_grad_(False);smoke=[]
for j,name in enumerate(['Global','Spatial','Temporal']):
 y,h,c,res=G.from_latent(Z['train'][:16],True);raw=G.base.from_latent(Z['train'][:16],1,True)[0];assert torch.equal(y,raw);y.retain_grad();loss=F.softplus(-D(y)[:,j]).mean();grads=torch.autograd.grad(loss,[y]+gp,allow_unused=True);nf=len(list(G.base.field.parameters()));fg=grads[1:1+nf];rg=grads[1+nf:];assert all(v is None or torch.isfinite(v).all() for v in grads);fn=sum(float(v.square().sum()) for v in fg if v is not None)**.5;rn=sum(float(v.square().sum()) for v in rg if v is not None)**.5;assert fn>0 and rn>0
 smoke.append(dict(branch=name,field_grad=fn,refiner_grad=rn,finite=True,active_input_gradient_channels=int((grads[0].abs().sum((0,1))>0).sum()),zero_residual_exact_identity=True))
save(O/'smoke.json',smoke);table(O/'parameter_inventory.csv',[dict(module=n,parameters=sum(p.numel() for p in m.parameters()),trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)) for n,m in [('base',G.base),('field',G.base.field),('refiner',G.refiner),('D50',D)]]);D.requires_grad_(True)
def state(phase,epoch,step):
 return dict(model=G.state_dict(),D=D.state_dict(),scales=ck['scales'],logscale=float(D.scale),G_optimizer=gopt.state_dict(),D_optimizer=dopt.state_dict(),used=used,D_steps=ds,G_steps=gs,phase=phase,epoch=epoch,step=step,logs=logs,gradients=gradientrows,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),K=1,seed=5001)
def checkpoint(name,phase,epoch,step):
 tmp=O/'checkpoints/.save.tmp';torch.save(state(phase,epoch,step),tmp);os.replace(tmp,O/'checkpoints'/name)
resume=O/'checkpoints/resume.pt'
if resume.exists():
 st=torch.load(resume,map_location=device,weights_only=False);G.load_state_dict(st['model']);D.load_state_dict(st['D']);gopt.load_state_dict(st['G_optimizer']);dopt.load_state_dict(st['D_optimizer']);used=st['used'];ds=st['D_steps'];gs=st['G_steps'];done_phase=st['phase'];next_epoch=st['epoch'];next_step=st['step'];logs=st['logs'];gradientrows=st['gradients'];torch.set_rng_state(st['torch_rng'].cpu());torch.cuda.set_rng_state(st['cuda_rng'].cpu());print('EXACT RESUME',done_phase,next_epoch,next_step,ds,gs,flush=True)
else:checkpoint('initial.pt','D',1,0)
def sync():torch.cuda.synchronize()
def start():
 if used>=1799:raise TimeoutError('1800sec cumulative budget')
 sync();return time.perf_counter()
def prediction(x):
 D.eval()
 with torch.inference_mode():return torch.cat([D(torch.from_numpy(x[i:i+128].copy()).to(device)).cpu() for i in range(0,len(x),128)]).numpy()
def dmetrics(y):
 rows=[];groups=[]
 for role,a,b in [('fit',rf,sf),('inner',ri,si)]:
  lr=prediction(r['x'][a]);ls=prediction(y[b]);np.savez_compressed(O/f'D50_{role}_probabilities.npz',real_logits=lr,sim_logits=ls,real_groups=r['g'][a],sim_groups=s['g'][b])
  for j,branch in enumerate(['Global','Spatial','Temporal']):
   mm=evalstats(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[lr[:,j],ls[:,j]],np.r_[.5*weights(r['g'][a]),.5*weights(s['g'][b])]);mm.update(role=role,branch=branch,Real_recall=float(weights(r['g'][a])@(lr[:,j]>=0)),Sim_recall=float(weights(s['g'][b])@(ls[:,j]<0)),confusion_matrix=json.dumps([[int((ls[:,j]<0).sum()),int((ls[:,j]>=0).sum())],[int((lr[:,j]<0).sum()),int((lr[:,j]>=0).sum())]]),prob_real_std=float(expit(lr[:,j]).std()),prob_sim_std=float(expit(ls[:,j]).std()))
   # Actual constant probability scores or single predicted class across BOTH domains are hard failures.
   if role=='inner':assert np.ptp(np.r_[lr[:,j],ls[:,j]])>1e-6 and 0<np.r_[lr[:,j]>=0,ls[:,j]>=0].mean()<1,'constant branch'
   rows.append(mm)
   for domain,gg,ll in [('Real',r['g'][a],lr[:,j]),('Sim',s['g'][b],ls[:,j])]:
    for g in np.unique(gg):v=ll[gg==g];groups.append(dict(role=role,branch=branch,domain=domain,group=g,acceptance=float((v>=0).mean()),recall=float(((v>=0) if domain=='Real' else (v<0)).mean()),prob_mean=float(expit(v).mean()),prob_P5=float(np.quantile(expit(v),.05)),prob_P50=float(np.median(expit(v))),prob_P95=float(np.quantile(expit(v),.95))))
 for row in rows:
  if row['role']=='inner':row['CE_gap']=row['CE']-next(v['CE'] for v in rows if v['role']=='fit' and v['branch']==row['branch'])
 table(O/'D50_metrics.csv',rows);table(O/'D50_group_metrics.csv',groups)
def preserve(y,x):
 masses=lambda a:torch.stack([a[:,:,l:h].sum(-1) for l,h in zip(OFF[:-1],OFF[1:])],-1)
 m,n=masses(y),masses(x);m=m/m.sum(-1,keepdim=True).clamp_min(1e-12);n=n/n.sum(-1,keepdim=True).clamp_min(1e-12);rel=lambda a:(a[:,1:]-a[:,:-1]).abs().sum(-1)/(a[:,1:].sum(-1)+a[:,:-1].sum(-1)).clamp_min(1e-12)
 return (m-n).square().sum(-1).mean(1)+(rel(y)-rel(x)).abs().mean(1)
try:
 if done_phase=='D':
  fake=torch.from_numpy(baseline['train']).to(device);D.train();D.requires_grad_(True)
  for epoch in range(next_epoch,11):
   rng=np.random.default_rng(5001+1000+epoch);steps=max(int(np.ceil(len(rf)/64)),int(np.ceil(len(sf)/64)));io=np.concatenate([rng.permutation(len(rf)) for _ in range(int(np.ceil(steps*64/len(rf))))])[:steps*64];jo=np.concatenate([rng.permutation(len(sf)) for _ in range(int(np.ceil(steps*64/len(sf))))])[:steps*64]
   for step in range(next_step if epoch==next_epoch else 0,steps):
    tic=start();a=io[step*64:(step+1)*64];b=jo[step*64:(step+1)*64];dopt.zero_grad(set_to_none=True);lazy=ds%16==0;rx=R[rf[a]].detach().requires_grad_(lazy);fl=D(fake[sf[b]]);rl=D(rx);ce=.5*((F.softplus(-rl)*rw[a,None]).mean()+(F.softplus(fl)*sw[b,None]).mean());pen=ce.new_zeros(())
    if lazy:
     for j in range(3):grad=torch.autograd.grad(rl[:,j].sum(),rx,create_graph=True,retain_graph=True)[0];pen=pen+(grad.square().flatten(1).sum(1)*rw[a]).mean()/3
    loss=ce+(.1/2)*16*pen if lazy else ce;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(D.parameters(),5);dopt.step();ds+=1;sync();used+=time.perf_counter()-tic;logs.append(dict(phase='D',epoch=epoch,step=step,loss=float(loss.detach()),CE=float(ce.detach()),global_CE=float(.5*(F.softplus(-rl[:,0]).mean()+F.softplus(fl[:,0]).mean())),spatial_CE=float(.5*(F.softplus(-rl[:,1]).mean()+F.softplus(fl[:,1]).mean())),temporal_CE=float(.5*(F.softplus(-rl[:,2]).mean()+F.softplus(fl[:,2]).mean())),optimization_seconds=used));checkpoint('resume.pt','D' if step+1<steps or epoch<10 else 'G',epoch if step+1<steps else (epoch+1 if epoch<10 else 1),step+1 if step+1<steps else 0)
   table(O/'training_log.csv',logs);print('D epoch',epoch,'steps',ds,'seconds',used,flush=True)
  checkpoint('D50.pt','G',1,0);next_epoch=1;next_step=0
 dmetrics(baseline['train']);D.eval();D.requires_grad_(False);dhash=shstate(D);save(O/'D_freeze_before.json',dict(state_sha=dhash));G.base.field.train();G.refiner.train()
 if done_phase!='complete':
  for epoch in range(next_epoch,21):
   order=np.random.default_rng(5001+2000+epoch).permutation(len(s['x']));steps=int(np.ceil(len(order)/64))
   for step in range(next_step if epoch==next_epoch else 0,steps):
    tic=start();ix=order[step*64:(step+1)*64];gopt.zero_grad(set_to_none=True);y,h,c,res=G.from_latent(Z['train'][ix],True);assert torch.isfinite(y).all() and (y>=0).all();logit=D(y);branches=(F.softplus(-logit)*gw[ix,None]).mean(0);adv=branches.mean();pre=(preserve(y,S[ix])*gw[ix]).mean();flow=((c[0]/scale).square().mean((1,2))*gw[ix]).mean();loss=adv+.5295653698327406*pre+.001*flow
    if step==0:
     nf=len(list(G.base.field.parameters()))
     for j,name in enumerate(['Global','Spatial','Temporal']):
      v=torch.autograd.grad(branches[j],gp,retain_graph=True,allow_unused=True);gradientrows.append(dict(epoch=epoch,branch=name,field_grad=sum(float(a.square().sum()) for a in v[:nf] if a is not None)**.5,refiner_grad=sum(float(a.square().sum()) for a in v[nf:] if a is not None)**.5,finite=all(a is None or torch.isfinite(a).all() for a in v)))
    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(gp,5);assert torch.isfinite(norm);gopt.step();gs+=1;sync();used+=time.perf_counter()-tic;logs.append(dict(phase='G',epoch=epoch,step=step,loss=float(loss.detach()),adv_global=float(branches[0].detach()),adv_spatial=float(branches[1].detach()),adv_temporal=float(branches[2].detach()),preserve=float(pre.detach()),flow=float(flow.detach()),residual_norm=float(res.detach().square().mean().sqrt()),output_window_std=float(y.detach().std(0).mean()),optimization_seconds=used));checkpoint('resume.pt','G' if step+1<steps or epoch<20 else 'complete',epoch if step+1<steps else epoch+1,step+1 if step+1<steps else 0)
   checkpoint(f'G_epoch_{epoch:02}.pt','G' if epoch<20 else 'complete',epoch+1,0);table(O/'training_log.csv',logs);table(O/'branch_gradients.csv',gradientrows);print('G epoch',epoch,'steps',gs,'seconds',used,flush=True)
 assert shstate(D)==dhash;assert all(torch.equal(v.cpu(),vae_before[k]) for k,v in G.base.state_dict().items() if k in vae_before);checkpoint('F50.pt','complete',21,0)
 y=generate('validation');np.savez_compressed(O/'outputs/F50_validation.npz',values=y,groups=sv['g'],stage=sv['s'],units=sv['u'],task=sv['task'])
 # Endpoint refiner-disabled ablation; no training or model choice.
 G.eval()
 with torch.inference_mode():raw=torch.cat([G.base.from_latent(Z['validation'][i:i+64],1,False)[0].cpu() for i in range(0,905,64)]).numpy()
 np.savez_compressed(O/'outputs/F50_without_refiner_validation.npz',values=raw,groups=sv['g'],stage=sv['s'],units=sv['u'],task=sv['task'])
 save(O/'complete.json',dict(D_steps=ds,G_steps=gs,optimization_seconds=used,GPU_peak_bytes=torch.cuda.max_memory_allocated(),D_frozen_hash=dhash,D_after_hash=shstate(D),VAE_bit_identical=True,seed=5001,rounds=1,endpoint=20))
except BaseException as e:
 save(O/'interruption.json',dict(error=str(e),D_steps=ds,G_steps=gs,optimization_seconds=used,resume='checkpoints/resume.pt stores NEXT step; completed steps not repeated'));raise
print('P50 TRAIN COMPLETE',used,flush=True)
