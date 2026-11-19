from c0_support import *
from model import Generator
from discriminator import AblationDiscriminator
from torch.nn import functional as F
import time,hashlib,copy,subprocess,platform
P44=Path('outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18');assert not (O/'config.yaml').exists();torch.set_num_threads(2);seed=4401;device='cuda:0' if torch.cuda.is_available() else 'cpu';cfg44=json.loads((P44/'config.json').read_text());summary44=json.loads((P44/'training_summary.json').read_text());assert summary44['completed_epochs']==20 and summary44['G_steps']==1152 and summary44['D_steps']==1280
paths=[P44/n for n in ['generator_initial.pt','discriminator_initial.pt','generator.pt','discriminator.pt','config.json','training_summary.json','C0_metrics.csv','D_metrics.csv','output_statistics.csv','code/model.py','code/vae.py']]+[Path('releases/tactile_c27_D090_v1/SHA256SUMS.json')];protected=dict(json.loads((C3/'c0/frozen.json').read_text())['files'])
for role in ['train','validation']:paths.extend([B/f'real_{role}/index.json',B/f'natural_sim_{role}/windows.npz'])
for p in paths:assert p.is_file(),p;protected[str(p)]=sha(p)
save(O/'protected_hashes_before.json',protected);assert sha(O/'code/model.py')==sha(P44/'code/model.py');assert sha(O/'code/vae.py')==sha(P44/'code/vae.py');initial=torch.load(P44/'generator_initial.pt',map_location='cpu',weights_only=False);dinitial=torch.load(P44/'discriminator_initial.pt',map_location='cpu',weights_only=False);final44=torch.load(P44/'generator.pt',map_location='cpu',weights_only=False);assert any(not torch.equal(v,final44['model'][k]) for k,v in initial['model'].items() if k.startswith('field.'));assert all(torch.equal(v,final44['model'][k]) for k,v in initial['model'].items() if k.startswith(('real.','sim.')));initG=Generator(**initial['scales']);initG.load_state_dict(initial['model']);initG.eval();initG.requires_grad_(False);r=real('train');n=natural('train');positive=r['x'][r['x']>0];epsilon=0.;tau=float(np.median(positive));assert tau>0;zs=[];zr=[];teacher=[]
with torch.inference_mode():
 for i in range(0,4049,64):
  z=initG.sim.posterior_mean(torch.from_numpy(n['x'][i:i+64]));zs.append(z.numpy());teacher.append(initG.real.decoder.deterministic_decode(initG.field.translator(z)).numpy())
 for i in range(0,5153,64):zr.append(initG.real.posterior_mean(torch.from_numpy(r['x'][i:i+64])).numpy())
data=dict(z=torch.from_numpy(np.concatenate(zs)).to(device),source=torch.from_numpy(n['x']).to(device),real=torch.from_numpy(r['x']).to(device),teacher=torch.from_numpy(np.concatenate(teacher)).to(device));latent_scale=torch.from_numpy(np.maximum(np.concatenate(zr).std((0,1)),.1)).to(device);logscale=dinitial['logscale'];OFF=np.cumsum([0]+[32,32,56]*5+[112]);np.savez(O/'training_loss_scales.npz',latent_scale=latent_scale.cpu().numpy(),D_logscale=float(logscale),epsilon=epsilon,tau=tau)
def statehash(state):
 h=hashlib.sha256()
 for k in sorted(state):h.update(k.encode());h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def make(group):
 torch.manual_seed(seed);G=Generator(**initial['scales']);G.load_state_dict(initial['model']);G.to(device);G.real.requires_grad_(False);G.sim.requires_grad_(False);G.real.eval();G.sim.eval();D=AblationDiscriminator(group,logscale,epsilon,tau);D.load_common_P44(dinitial['model']);D.to(device);return G,D
parameter_rows=[];initial_audits=[];gradrows=[]
def gradient_norm(t):return float(torch.linalg.vector_norm(t.detach()).cpu())
def grad_audit(group,G,D):
 G.eval();G.field.train();D.eval();z=data['z'][:8];y,h,_=G.from_latent(z,8,True);hard=G.decode(h,False);assert torch.equal(y,hard);features,m,a=D.features(y,True);score=D.net(features).squeeze(-1);loss=F.softplus(-score).mean();df=torch.autograd.grad(loss,features,retain_graph=True)[0]
 if group=='A':ga=torch.autograd.grad(a,y,grad_outputs=df,retain_graph=True)[0];gm=torch.zeros_like(y)
 else:
  ga=torch.autograd.grad(a,y,grad_outputs=df[:,2848:],retain_graph=True)[0]
  gm=torch.autograd.grad(m,y,grad_outputs=df[:,:2848],retain_graph=True)[0] if m.requires_grad else torch.zeros_like(y)
 total=ga+gm;params=list(G.field.parameters())
 for branch,v in [('continuous',ga),('Mask',gm),('combined',total)]:
  pgr=torch.autograd.grad(y,params,grad_outputs=v,retain_graph=True,allow_unused=True);pn=float(torch.sqrt(sum(g.detach().double().square().sum() for g in pgr if g is not None)).cpu());row=dict(group=group,branch=branch,input_output_gradient_L2=gradient_norm(v),field_parameter_gradient_L2=pn,finite=bool(torch.isfinite(v).all()) and all(torch.isfinite(g).all() for g in pgr if g is not None));assert row['finite'];gradrows.append(row)
  if group=='C' and branch=='Mask':assert pn>0 and gradient_norm(v)>0
  if group=='A' and branch=='combined':assert pn>0 and gradient_norm(v)>0
  if group=='B' and branch=='Mask':assert pn==0 and gradient_norm(v)==0
 if group=='C':assert torch.equal(D.mask(y,True),(y>0).float())
 # Preserve P44 R1 computation path exactly for real data (no surrogate occupancy in R1).
 rx=data['real'][:8].detach().requires_grad_(True);lo=D(rx,mask_backward=False);gr=torch.autograd.grad(lo.sum(),rx,create_graph=True)[0];pen=gr.square().flatten(1).sum(1).mean();dp=torch.autograd.grad(F.softplus(-lo).mean()+.05*pen,list(D.parameters()));assert all(torch.isfinite(g).all() for g in dp)
for group in ['A','B','C']:
 G,D=make(group);g_hash=statehash(G.state_dict());assert g_hash==statehash(initial['model']);audit=[]
 for k,v in dinitial['model'].items():
  vv=D.state_dict()[k];expected=v[:,2848:] if group=='A' and k=='net.0.weight' else v;assert torch.equal(vv.cpu(),expected);audit.append(k)
 initial_audits.append(dict(group=group,Generator_initial_state_hash=g_hash,source_initial_file_sha256=sha(P44/'generator_initial.pt'),D_common_tensors_exact=True,D_first_layer='original amplitude columns2848:5696' if group=='A' else 'original full5696columns',Generator_parameters=sum(p.numel() for p in G.parameters()),field_trainable_parameters=sum(p.numel() for p in G.field.parameters()),D_parameters=sum(p.numel() for p in D.parameters())))
 for name,p in D.named_parameters():parameter_rows.append(dict(group=group,model='D',name=name,shape=str(list(p.shape)),numel=p.numel(),sha256=statehash({name:p}),trainable=True))
 for name,p in G.named_parameters():parameter_rows.append(dict(group=group,model='G',name=name,shape=str(list(p.shape)),numel=p.numel(),sha256=statehash({name:p}),trainable=p.requires_grad))
 grad_audit(group,G,D)
 # Same forward C/B and common layers; A is exactly original D with Mask feature block zeroed.
 with torch.no_grad():
  a=data['real'][:8];old=module('p45_original_P44_model',P44/'code/model.py').Discriminator(logscale).to(device);old.load_state_dict(dinitial['model']);old.eval();fx=torch.cat([(a>0).float().flatten(1),(torch.log1p(a)/old.log_scale).flatten(1)],1)
  if group=='A':fx[:,:2848]=0;expected=old.net(fx).squeeze(-1)
  else:expected=old(a)
  assert torch.allclose(D(a),expected,rtol=1e-5,atol=1e-6)
 del G,D
save(O/'initialization_audit.json',initial_audits);table(O/'gradient_audit.csv',gradrows);table(O/'parameter_audit.csv',parameter_rows);config=dict(date='2026-11-18',seed=seed,groups=['A_Continuous_only','B_P44_historical','C_STE_Mask'],B_retrained=False,Generator_initial_checkpoint=str(P44/'generator_initial.pt'),Discriminator_initial_checkpoint=str(P44/'discriminator_initial.pt'),parent_config_sha256=sha(P44/'config.json'),training_inherited=cfg44,epsilon=epsilon,tau=tau,tau_rule='median of all positive Real TRAIN processed tactile values, no clipping/search',hard_threshold=0.,C_forward='hard+(soft-soft.detach()), bit-exact hard x>0',C_R1='explicit mask_backward=False on real R1 path, preserving P44 regularizer; STE only supplies additional G-side occupancy gradient',A_initial_mapping='first layer copies amplitude-half columns of original D; every common-layer tensor copied exactly',no_generator_or_loss_change=True,selected_checkpoint_rule='same as P44, fixed epoch20 endpoint',per_group_max_epochs=20,per_group_max_seconds=1800,same_batches_and_K=True,Test_Holdout=False,C0_not_used_for_gradients_or_selection=True,autostart_P46=False);save(O/'config.json',config);(O/'config.yaml').write_text(json.dumps(config,indent=2,ensure_ascii=False));save(O/'code_hashes_before_training.json',{str(p):sha(p) for p in (O/'code').glob('*.py')});save(O/'smoke.json',dict(all_group_gradients_finite=True,A_continuous_to_field_nonzero=True,B_Mask_to_field_zero=True,C_Mask_to_field_nonzero=True,K8_hard_forward_bit_identical=True,R1_finite=True,no_optimizer_steps_during_audit=True));print('P45 AUDIT PASS',json.dumps(initial_audits),flush=True)
def shares(x):
 m=torch.stack([x[:,:,l:h].sum(-1) for l,h in zip(OFF[:-1],OFF[1:])],-1);return m/m.sum(-1,keepdim=True).clamp_min(1e-12)
def relchange(x):return (x[:,1:]-x[:,:-1]).abs().sum(-1)/(x[:,1:].sum(-1)+x[:,:-1].sum(-1)).clamp_min(1e-12)
def preserve(y,x):return (shares(y)-shares(x)).square().sum(-1).mean()+(relchange(y)-relchange(x)).abs().mean()
metrics=module('p45_metrics','outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py').metrics
@torch.no_grad()
def monitor(G,D):
 G.eval();D.eval();ni=np.linspace(0,4048,256).round().astype(int);ri=np.linspace(0,5152,256).round().astype(int);fake,_,_=G.from_latent(data['z'][ni],4,False);prob=torch.cat([D(fake).sigmoid(),D(data['real'][ri]).sigmoid()]).cpu().numpy();m=metrics(np.r_[np.zeros(256,int),np.ones(256,int)],prob);G.field.train();D.train();return m
for group in ['A','C']:
 folder=O/group;folder.mkdir();G,D=make(group);G.field.train();D.train();gopt=torch.optim.Adam(G.field.parameters(),lr=1e-4,betas=(0.,.99));dopt=torch.optim.Adam(D.parameters(),lr=1e-4,betas=(0.,.99));trainstart=time.time();history=[];nvis=np.zeros(4049,int);rvis=np.zeros(5153,int);gsteps=dsteps=0;kcounts={k:0 for k in [1,2,4,8]};plans=[];torch.cuda.reset_peak_memory_stats() if device.startswith('cuda') else None;peak_smoke=None
 for epoch in range(1,21):
  assert time.time()-trainstart<1800,'P45 group budget exhausted, stop no extension';rng=np.random.default_rng(seed+epoch);ni=rng.permutation(4049);ri=rng.permutation(5153);schedule=np.array([1,2,4,8]*16);rng.shuffle(schedule);plans.append(dict(epoch=epoch,Natural_order_hash=sha_bytes(ni) if False else hashlib.sha256(ni.tobytes()).hexdigest(),Real_order_hash=hashlib.sha256(ri.tobytes()).hexdigest(),K_order_hash=hashlib.sha256(schedule.tobytes()).hexdigest()));warm=epoch<=2;dlrows=[];glrows=[]
  for step in range(64):
   assert time.time()-trainstart<1800;ix=ni[step*64:(step+1)*64];ir=ri[step*81:min((step+1)*81,5153)];K=int(schedule[step]);kcounts[K]+=1;nvis[ix]+=1;rvis[ir]+=1;z=data['z'][ix];source=data['source'][ix];realx=data['real'][ir];D.requires_grad_(True);dopt.zero_grad(set_to_none=True)
   with torch.no_grad():fake,_,_=G.from_latent(z,K,False)
   if not torch.isfinite(fake).all() or fake.max()>1e6:raise FloatingPointError('same P44 range guard, no clipping')
   lazy=(dsteps%16==0);rx=realx.detach().requires_grad_(lazy);rlog=D(rx,mask_backward=False);flog=D(fake.detach(),mask_backward=False);dl=.5*(F.softplus(-rlog).mean()+F.softplus(flog).mean());pen=dl.new_zeros(())
   if lazy:gg=torch.autograd.grad(rlog.sum(),rx,create_graph=True)[0];pen=gg.square().flatten(1).sum(1).mean();dl=dl+(.1/2)*16*pen
   assert torch.isfinite(dl);dl.backward();dn=torch.nn.utils.clip_grad_norm_(D.parameters(),5);assert torch.isfinite(dn);dopt.step();dsteps+=1;dlrows.append([float(dl.detach()),float(pen.detach())])
   if not warm:
    D.requires_grad_(False);gopt.zero_grad(set_to_none=True);y,h,c=G.from_latent(z,K,True);adv=F.softplus(-D(y,mask_backward=True)).mean();pre=preserve(y,source);flow=(c/latent_scale).square().mean();teacher=((torch.log1p(y)-torch.log1p(data['teacher'][ix]))/logscale).square().mean();tw=max(0.,1-(epoch-3)/5);loss=adv+cfg44['lambda_preserve']*pre+.001*flow+tw*teacher;assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(G.field.parameters(),5);assert torch.isfinite(gn);gopt.step();gsteps+=1;glrows.append([float(v.detach()) for v in [loss,adv,pre,flow,teacher]]+[tw]);D.requires_grad_(True)
  mm=monitor(G,D);row=dict(group=group,epoch=epoch,D_steps=dsteps,G_steps=gsteps,D_loss_mean=float(np.mean([v[0] for v in dlrows])),R1_mean=float(np.mean([v[1] for v in dlrows])),monitor_Train_AUC=mm['roc_auc'],monitor_Train_BA=mm['balanced_accuracy'],seconds=time.time()-trainstart)
  for j,k in enumerate(['G_total','G_adv','G_preserve','G_flow','G_teacher','teacher_weight']):row[k]=float(np.mean([v[j] for v in glrows])) if glrows else None
  history.append(row);table(folder/'training_history.csv',history);print('EPOCH',json.dumps(row),flush=True)
  with (O/f'training_log_{group}.txt').open('a') as f:f.write(json.dumps(row)+'\n')
 assert gsteps==1152 and dsteps==1280 and np.all(nvis==20) and np.all(rvis==20);assert all(torch.equal(v.cpu(),initial['model'][k]) for k,v in G.state_dict().items() if k.startswith(('real.','sim.')));torch.save(dict(model={k:v.detach().cpu().clone() for k,v in G.state_dict().items()},scales=initial['scales'],epoch=20,group=group,G_steps=gsteps),folder/'generator.pt');torch.save(dict(model={k:v.detach().cpu().clone() for k,v in D.state_dict().items()},logscale=logscale,epsilon=epsilon,tau=tau,mode=group,epoch=20,D_steps=dsteps),folder/'discriminator.pt');np.savez(folder/'coverage.npz',Natural=nvis,Real=rvis);save(folder/'sampling_orders.json',plans);save(folder/'training_summary.json',dict(epochs=20,G_steps=gsteps,D_steps=dsteps,K_step_counts=kcounts,train_seconds=time.time()-trainstart,GPU_peak_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,all_Train_windows_each_epoch=True,VAE_bit_identical=True,field_initial_hash=statehash(initial['model']),no_checkpoint_search=True));save(folder/'frozen.json',dict(generator_sha256=sha(folder/'generator.pt'),D_sha256=sha(folder/'discriminator.pt'),time=time.time(),C0_not_yet_evaluated=True));print('GROUP FROZEN',group,flush=True);del G,D,gopt,dopt
assert json.loads((O/'A/sampling_orders.json').read_text())==json.loads((O/'C/sampling_orders.json').read_text());assert all(sha(p)==h for p,h in protected.items());save(O/'training_complete.json',dict(groups=['A','C'],B_reused=True,models_frozen_before_any_C0=True,same_sampling_orders=True,steps_match_P44=True,no_further_training=True));print('P45 TRAINING COMPLETE NO_P46',flush=True)
