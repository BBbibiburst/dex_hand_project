"""Frozen-C0 baseline sensitivity only; no optimization, acceptance changes, or converter graph."""
from common import *
import time
assert json.loads((O/'baseline_reproduced.json').read_text())['passed'];torch.set_num_threads(2);start=time.time();r=real('validation');n=natural('validation');gen=np.load(O/'baseline/translated_validation.npz')['values'];ri=np.concatenate([np.flatnonzero(r['g']==g)[np.linspace(0,sum(r['g']==g)-1,10).round().astype(int)] for g in sorted(set(r['g']))]);ni=np.concatenate([np.flatnonzero(n['g']==g)[np.linspace(0,sum(n['g']==g)-1,4).round().astype(int)] for g in sorted(set(n['g']))]);sc=np.load(C3/'c0/scaler.npz');x=((np.log1p(np.concatenate([gen[ni],r['x'][ri]]))-sc['mean'])/sc['scale']).astype('float32');variants=[('original',x)];OFF=np.cumsum([0]+[32,32,56]*5+[112])
for frame in range(4):a=x.copy();a[:,frame]=0;variants.append((f'frame_{frame}',a))
for p,(l,h) in enumerate(zip(OFF[:-1],OFF[1:])):a=x.copy();a[:,:,l:h]=0;variants.append((f'patch_{p}',a))
allx=np.concatenate([a for _,a in variants]);Deep=extract('outputs/tactile_p35_deep_probe_2026-11-15/code/run.py','DeepProbe',dict(env,Probe=extract('apps/evaluate_tactile_712d_gan.py','Probe',env)));rows=[];frozen=json.loads((C3/'c0/frozen.json').read_text())
for e in frozen['instances']:
 name,kind=e['name'],e['kind'];p=C3/'c0'/name;parts=[]
 with torch.inference_mode():
  if kind=='logistic':ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
  elif kind=='knn':ck=np.load(p/'checkpoint.npz');ref=torch.from_numpy(ck['values'].reshape(len(ck['labels']),-1)).double();lab=torch.from_numpy(ck['labels']).double();rn=(ref*ref).sum(1)
  else:ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False);net=Deep(kind);net.load_state_dict(ck['model']);net.eval();net.requires_grad_(False)
  for i in range(0,len(allx),128):
   q=torch.from_numpy(allx[i:i+128])
   if kind=='logistic':pp=(q.flatten(1).double()@ck['weight']+ck['bias']).sigmoid()
   elif kind=='knn':q=q.flatten(1).double();dd=(q*q).sum(1)[:,None]+rn[None]-2*q@ref.T;pp=lab[dd.topk(5,largest=False,dim=1).indices].mean(1)
   else:pp=net(q).sigmoid()
   parts.append(pp.numpy())
 pr=np.concatenate(parts).reshape(len(variants),80);base=pr[0]
 for j,(label,_) in enumerate(variants[1:],1):rows.append(dict(instance=name,removed_block=label,normalized_replacement='original Train mean (0 in original C0 normalized coordinates)',mean_abs_Real_probability_change=float(abs(pr[j]-base).mean()),mean_signed_change=float((pr[j]-base).mean()),Sim_half_mean_abs_change=float(abs(pr[j,:40]-base[:40]).mean()),Real_half_mean_abs_change=float(abs(pr[j,40:]-base[40:]).mean()),acceptance_use=False))
 print('SENSITIVITY',name,flush=True)
table(O/'baseline_channel_frame_sensitivity.csv',rows);save(O/'sensitivity_scope.json',dict(sample_Real_units=r['u'][ri].tolist(),sample_Natural_units=n['u'][ni].tolist(),selection='metadata-only equally spaced4 perNaturalRoot,10 perRealSession',frozen_models=True,optimizer_steps=0,no_gradient_or_converter_backprop=True,acceptance_use=False,warning='occlusion may be OOD; block sensitivity not causal attribution nor physical Taxel importance; saturation can hide sensitivity',seconds=time.time()-start));print('SENSITIVITY COMPLETE',flush=True)
