"""Frozen final-gradient and target-domain reconstruction checks; no optimizer or C0 fitting."""
from c0_support import *
from model import Generator
from discriminator import AblationDiscriminator
from torch.nn import functional as F
import hashlib
P44=Path('outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18');assert (O/'training_complete.json').exists();assert not (O/'diagnostics_complete.json').exists();torch.set_num_threads(2);cfg=json.loads((O/'config.json').read_text());epsilon,tau=cfg['epsilon'],cfg['tau'];device='cuda:0' if torch.cuda.is_available() else 'cpu';initial=torch.load(P44/'generator_initial.pt',map_location='cpu',weights_only=False);base=Generator(**initial['scales']);base.load_state_dict(initial['model']);base.eval();base.requires_grad_(False);train=natural('train');val=natural('validation');rr=real('validation');metrics=module('p45_diag_metrics','outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py').metrics
with torch.inference_mode():sourcez=base.sim.posterior_mean(torch.from_numpy(train['x'][:64])).numpy()[:8];rec=np.concatenate([base.real.decoder.deterministic_decode(base.real.posterior_mean(torch.from_numpy(rr['x'][i:i+64]))).numpy() for i in range(0,780,64)])
np.savez_compressed(O/'Real_validation_C3_reconstruction.npz',values=rec,groups=rr['g'],stage=rr['s'],units=rr['u']);rows=[dict(phase='initial',**r) for r in csv.DictReader((O/'gradient_audit.csv').open())];table(O/'gradient_audit_initial.csv',rows);readout=[];memory=[];sharing=[]
for group in ['A','B','C']:
 folder=P44 if group=='B' else O/group;ck=torch.load(folder/'generator.pt',map_location='cpu',weights_only=False);dc=torch.load(folder/'discriminator.pt',map_location='cpu',weights_only=False);G=Generator(**ck['scales']);G.load_state_dict(ck['model']);G.to(device);G.eval();G.field.train();D=AblationDiscriminator(group,dc['logscale'],epsilon,tau);D.load_common_P44(dc['model']) if group=='B' else D.load_state_dict(dc['model']);D.to(device);D.eval();D.requires_grad_(False);torch.cuda.reset_peak_memory_stats() if device.startswith('cuda') else None;z=torch.from_numpy(sourcez).to(device);y,h,_=G.from_latent(z,8,True);assert torch.equal(y,G.decode(h,False));features,m,a=D.features(y,True);lo=F.softplus(-D.net(features).squeeze(-1)).mean();df=torch.autograd.grad(lo,features,retain_graph=True)[0]
 if group=='A':ga=torch.autograd.grad(a,y,grad_outputs=df,retain_graph=True)[0];gm=torch.zeros_like(y)
 else:
  ga=torch.autograd.grad(a,y,grad_outputs=df[:,2848:],retain_graph=True)[0];gm=torch.autograd.grad(m,y,grad_outputs=df[:,:2848],retain_graph=True)[0] if m.requires_grad else torch.zeros_like(y)
 for branch,v in [('continuous',ga),('Mask',gm),('combined',ga+gm)]:
  grads=torch.autograd.grad(y,list(G.field.parameters()),grad_outputs=v,retain_graph=True,allow_unused=True);pn=float(torch.sqrt(sum(g.detach().double().square().sum() for g in grads if g is not None)).cpu());rows.append(dict(phase='final',group=group,branch=branch,input_output_gradient_L2=float(torch.linalg.vector_norm(v).detach().cpu()),field_parameter_gradient_L2=pn,finite=bool(torch.isfinite(v).all()) and all(torch.isfinite(g).all() for g in grads if g is not None)))
 memory.append(dict(group=group,kind='post-training frozen-weight K8 backward smoke, no optimizer',peak_GPU_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,all_finite=True));G.eval();D.eval();pp=[]
 with torch.inference_mode():
  for x in [rec,rr['x']]:
   for i in range(0,len(x),128):pp.append(D(torch.from_numpy(x[i:i+128]).to(device)).sigmoid().cpu().numpy())
 labels=np.r_[np.zeros(780,int),np.ones(780,int)];readout.append(dict(group=group,scope='Real processed vs own C3 reconstruction, BOTH sourceReal; NOT Natural transfer',**metrics(labels,np.concatenate(pp))))
 if group=='B':
  (O/'B/outputs').mkdir(parents=True,exist_ok=True);take=np.concatenate([np.flatnonzero(val['g']==g)[np.linspace(0,sum(val['g']==g)-1,4).round().astype(int)] for g in sorted(set(val['g']))]);z=G.sim.posterior_mean(torch.from_numpy(val['x'][take]).to(device))
  with torch.no_grad():
   for K in [1,2,4,8]:
    h=G.initial_state.expand(len(z),-1,-1);states=[h.cpu().numpy().copy()];vel=[]
    for k in range(K):v,c=G.field(h,z,k/K);old=h;h=h+v/K;states.append(h.cpu().numpy().copy());vel.append(v.cpu().numpy());sharing.append(dict(group=group,K=K,step=k,time=k/K,state_norm_mean=float(torch.linalg.vector_norm(h.flatten(1),dim=1).mean()),velocity_norm_mean=float(torch.linalg.vector_norm(v.flatten(1),dim=1).mean()),step_delta_norm=float(torch.linalg.vector_norm((h-old).flatten(1),dim=1).mean()),finite=True))
    np.savez_compressed(O/f'B/outputs/K{K}_trajectory.npz',states=np.stack(states,1),velocities=np.stack(vel,1),units=val['u'][take],groups=val['g'][take],source='P44 final frozen G, no retraining')
 del G,D
assert all(str(r['finite']) in ['True','true'] for r in rows);table(O/'gradient_audit.csv',rows);table(O/'post_training_memory_smoke.csv',memory);table(O/'Real_reconstruction_D_readout.csv',readout);table(O/'B_iteration_trajectory.csv',sharing)
# Reconstruct P44 sampling algorithm, compare all20 epochs and coverage, no new training.
expected=[]
for ep in range(1,21):
 rng=np.random.default_rng(4401+ep);ni=rng.permutation(4049);ri=rng.permutation(5153);K=np.array([1,2,4,8]*16);rng.shuffle(K);expected.append(dict(epoch=ep,Natural_order_hash=hashlib.sha256(ni.tobytes()).hexdigest(),Real_order_hash=hashlib.sha256(ri.tobytes()).hexdigest(),K_order_hash=hashlib.sha256(K.tobytes()).hexdigest()))
for group in ['A','C']:
 assert json.loads((O/group/'sampling_orders.json').read_text())==expected;a=np.load(O/group/'coverage.npz');b=np.load(P44/'sample_visits.npz');assert np.array_equal(a['Natural'],b['Natural']) and np.array_equal(a['Real'],b['Real'])
save(O/'sampling_audit.json',dict(A_C_full20epoch_order_equal=True,P44_same_seed_and_explicit_sampling_algorithm=True,full_visitation_counts_equal_P44=True,source_P44_train_sha256=sha(P44/'code/train.py'),same_Real5153_Natural4049_each_epoch=True));assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'diagnostics_complete.json',dict(no_training=True,all_initial_final_gradients_finite=True,readout_scope='target self-reconstruction diagnostic only',B_trajectory_inference_only=True,no_new_C0_evaluation=True));print('P45 FROZEN DIAGNOSTICS COMPLETE',flush=True)
