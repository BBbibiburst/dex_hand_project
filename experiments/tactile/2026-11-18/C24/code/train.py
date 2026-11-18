"""C24 one fixed Real-only continuation; no classifier feedback and no alternate weight."""
import ast,json,csv,time,hashlib,importlib.util,sys,zipfile,subprocess
from pathlib import Path
import numpy as np
import torch
O=Path(__file__).resolve().parents[1];P=Path('outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18');B=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16');C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');C23=Path('outputs/tactile_c23_real_reconstruction_c0_2026-11-18')
def save(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def table(n,rs):
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert json.loads((O/'stage0.json').read_text())['training_can_proceed'];assert not (O/'registration.json').exists();start=time.time();torch.set_num_threads(2);device='cuda:0' if torch.cuda.is_available() else 'cpu';seed=4313;torch.manual_seed(seed)
 env=dict(np=np,zipfile=zipfile);tree=ast.parse(Path('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='npz_rows'],type_ignores=[]),'getrows','exec'),env);get=env['npz_rows']
 spec=importlib.util.spec_from_file_location('c24_existing_vae',C3/'code/models.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
 ck=torch.load(P/'checkpoint.pt',map_location='cpu',weights_only=False);R=M.DomainVAE(ck['encoder_scale'],ck['positive_scale']).to(device);R.load_state_dict(ck['model']);R.requires_grad_(False);R.eval();D=R.decoder
 for p in list(D.current.parameters())+list(D.amplitude.parameters()):p.requires_grad_(True)
 params=[p for p in R.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==594512;initial={k:v.detach().cpu().clone() for k,v in R.state_dict().items()};torch.save(ck,O/'initial_checkpoint.pt')
 protected=dict(json.loads((C3/'c0/frozen.json').read_text())['files'])
 for p in [P/'checkpoint.pt',P/'train_outputs.npz',P/'validation_outputs.npz',P/'code/run.py',C3/'code/models.py',C23/'metrics.csv',C23/'code/evaluate.py']+[Path('outputs/tactile_p43_c4_root_cause_audit_2026-11-16/latents')/f'Real_{r}.npz' for r in ['train','validation']]+[B/f'real_{role}/index.json' for role in ['train','validation']]:protected[str(p)]=sha(p)
 assert all(sha(p)==h for p,h in protected.items());save('protected_hashes_before.json',protected);data={};cache={}
 if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
 for role in ['train','validation']:
  idx=json.loads((B/f'real_{role}/index.json').read_text());x=get(idx['source'],'real',idx['indices']); stored_mu=np.load('outputs/tactile_p43_c4_root_cause_audit_2026-11-16/latents/Real_'+role+'.npz')['mu']; g=get(idx['source'],'real_group',idx['indices']);s=get(idx['source'],'real_stage',idx['indices']);u=get(idx['source'],'real_unit',idx['indices']);assert len(x)==(5153 if role=='train' else 780);data[role]=dict(x=x,g=g,s=s,u=u);hs=[];ms=[];mus=[]
  with torch.no_grad():
   for i in range(0,len(x),128):
    recomputed=R.encoder.cpu()(torch.from_numpy(x[i:i+128]))[0];z=torch.from_numpy(stored_mu[i:i+128]).to(device); assert torch.allclose(z.cpu(),recomputed,rtol=1e-5,atol=1e-5);mus.append(z);h=z.new_zeros((len(z),128));prev=z.new_zeros((len(z),712));hh=[];mm=[]
    for t in range(4):
     context=D.bos[None].expand(len(z),-1) if t==0 else D.previous(prev);h=D.cell(torch.cat([z[:,t],context],1),h);prev=(D.mask(h)>=0).float();hh.append(h);mm.append(prev)
    hs.append(torch.stack(hh,1));ms.append(torch.stack(mm,1))
  cache[role]=dict(h=torch.cat(hs),mask=torch.cat(ms),mu=torch.cat(mus),target=torch.from_numpy(x).to(device));assert np.array_equal(cache[role]['mask'].cpu().numpy()>0,np.load(P/f'{role}_outputs.npz')['values']>0)
 def forward(role,ix):
  c=cache[role];n=len(ix);mix,k,rate=tuple(v.reshape(n,4,712,2) for v in D.parameters_for(c['h'][ix].reshape(-1,128),c['mask'][ix].reshape(-1,712)));target=c['target'][ix];active=target>0;scale=D.positive_scale.double();x=torch.where(active,target,torch.ones_like(target)).double()/scale;rr=rate.double();kk=k.double();logp=kk*rr.log()-torch.lgamma(kk)+(kk-1)*x[...,None].log()-rr*x[...,None];logp=torch.logsumexp(mix.double().log_softmax(-1)+logp,-1)-scale.log();nll=-(logp*active).sum((1,2))/active.sum((1,2)).clamp_min(1)
  # Deterministic expectation in ORIGINAL nonnegative response units, with frozen free-run Mask.
  amp=(mix.softmax(-1)*k/rate).sum(-1)*D.positive_scale;y=amp*c['mask'][ix];dy=torch.diff(torch.log1p(y.double()),dim=1);dt=torch.diff(torch.log1p(target.double()),dim=1);temp=(dy-dt).abs().mean((1,2));return nll,temp,y
 def ev(role):
  n=[];t=[]
  with torch.no_grad():
   for i in range(0,len(data[role]['x']),64):
    a,b,_=forward(role,np.arange(i,min(i+64,len(data[role]['x']))));n.append(a.cpu().numpy());t.append(b.cpu().numpy())
  n=np.concatenate(n);t=np.concatenate(t);return dict(nll=float(n.mean()),absolute_per_window_nll=float(np.abs(n).mean()),temporal=float(t.mean()))
 initialscale=ev('train');lam=.1*initialscale['absolute_per_window_nll']/initialscale['temporal'];assert np.isfinite(lam) and lam>0
 reg=dict(seed=seed,train_windows=5153,train_Sessions=32,validation_windows=780,validation_Sessions=4,epochs_fixed=18,epoch_basis='C13 actual18 epochs; fixed18, no early stopping or extension',batch_size=64,learning_rate=.001,weight_decay=.0001,gradient_clip=5,optimizer='AdamW new state as C13; continuation from C13 best model',trainable_prefixes=['decoder.current.','decoder.amplitude.'],trainable_parameters=594512,all_parameters=1094232,loss='original per-window positive Gamma mixture NLL + lambda mean3x712 absolute log1p increment discrepancy',lambda_temporal=lam,lambda_rule='0.1 * mean(abs(initial per-window Real TRAIN NLL)) / initial Real TRAIN temporal loss; exactly one weight, no C0/Validation used',lambda_initial_train=initialscale,checkpoint_rule='unchanged C13: minimum Real Validation POSITIVE NLL, improvement tolerance1e-5; no C0-based selection; final endpoint also saved',budget_seconds=900,max_new_bytes=1_000_000_000,latent_context='reuse exact C13 cached Real posterior means; original CPU Encoder recomputation checked, not substituted, avoids roundoff-driven Mask changes',units='original uncalibrated processed tactile response units, no claim of Newton/Pa; decoder positive_scale restored before log1p; log1p applied once',initial_checkpoint_sha256=sha(P/'checkpoint.pt'),Test_Holdout_read=False,Natural_read=False,epochs_or_seed_search=False,autostart_next=False,git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 save('registration.json',reg);save('parameter_list.json',[dict(name=k,shape=list(v.shape),numel=v.numel(),trainable=v.requires_grad) for k,v in R.named_parameters()]);save('code_hashes_at_registration.json',{str(p):sha(p) for p in (O/'code').glob('*.py')});print('REGISTERED',json.dumps(reg),flush=True)
 initialval=ev('validation');save('initial_loss.json',dict(train=initialscale,validation=initialval));opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001);best=float('inf');bestep=0;hist=[];visits=np.zeros(5153,int);steps=0;trainstart=time.time()
 for epoch in range(1,19):
  assert time.time()-start<900;order=np.random.default_rng(seed+epoch).permutation(5153);R.eval();D.current.train();D.amplitude.train();online=[]
  for i in range(0,5153,64):
   assert time.time()-start<900;ix=order[i:i+64];visits[ix]+=1;opt.zero_grad(set_to_none=True);n,t,_=forward('train',ix);loss=(n+lam*t).mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,5);assert torch.isfinite(norm);opt.step();steps+=1;online.append((float(n.mean().detach()),float(t.mean().detach()),len(ix)))
  tr=ev('train');va=ev('validation');state=dict(model={k:v.detach().cpu().clone() for k,v in R.state_dict().items()},encoder_scale=ck['encoder_scale'],positive_scale=ck['positive_scale'],epoch=epoch,optimizer_steps=steps,seed=seed,validation_positive_NLL=va['nll'],kind='C24_single_temporal_amplitude_repair',lambda_temporal=lam)
  if va['nll']<best-1e-5:best=va['nll'];bestep=epoch;torch.save(state,O/'checkpoint.pt')
  row=dict(epoch=epoch,optimizer_steps=steps,total_sample_visits=int(visits.sum()),train_NLL=tr['nll'],train_temporal=tr['temporal'],train_total=tr['nll']+lam*tr['temporal'],validation_NLL=va['nll'],validation_temporal=va['temporal'],validation_total=va['nll']+lam*va['temporal'],best_epoch=bestep,seconds=time.time()-trainstart);hist.append(row);table('training_history.csv',hist);print('EPOCH',json.dumps(row),flush=True)
 torch.save(state,O/'endpoint_checkpoint.pt');selected=torch.load(O/'checkpoint.pt',map_location='cpu',weights_only=False);R.load_state_dict(selected['model']);R.requires_grad_(False);R.eval();changed=[k for k,v in R.state_dict().items() if not torch.equal(v.cpu(),initial[k])];assert all(k.startswith(('decoder.current.','decoder.amplitude.')) for k in changed);save('parameter_change_audit.json',dict(changed_keys=changed,all_frozen_bit_identical=True,Mask_and_Encoder_bit_identical=True));np.savez(O/'sample_visits.npz',visits=visits);out=[]
 with torch.no_grad():
  for i in range(0,780,64):
   ix=np.arange(i,min(i+64,780));_,_,y=forward('validation',ix);out.append(y.cpu().numpy())
 y=np.concatenate(out);assert np.isfinite(y).all() and (y>=0).all();old=np.load(P/'validation_outputs.npz')['values'];assert np.array_equal(y>0,old>0)
 # Audit cached differentiable readout against unchanged architecture's actual free-running decode.
 check=[]
 with torch.no_grad():
  for i in range(0,780,64):check.append(D.deterministic_decode(cache['validation']['mu'][i:i+64]).cpu().numpy())
 actual=np.concatenate(check);assert np.allclose(actual,y,rtol=1e-5,atol=1e-5);np.savez_compressed(O/'validation_outputs.npz',values=y,groups=data['validation']['g'],stage=data['validation']['s']);save('decode_equivalence.json',dict(max_abs_diff=float(np.abs(actual-y).max()),Mask_bit_identical_C13=True,allclose_rtol=1e-5,atol=1e-5))
 assert np.all(visits==18);assert all(sha(p)==h for p,h in protected.items());save('training_summary.json',dict(epochs=18,best_epoch=bestep,optimizer_steps=steps,total_sample_visits=int(visits.sum()),all_5153_windows_covered_every_epoch=True,seconds=time.time()-trainstart,device=device,GPU_name=torch.cuda.get_device_name() if device.startswith('cuda') else None,GPU_peak_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,new_probe_fits=0,protected_unchanged=True,only_one_lambda=True,C0_not_accessed_before_freeze=True));save('training_freeze.json',dict(checkpoint_sha256=sha(O/'checkpoint.pt'),outputs_sha256=sha(O/'validation_outputs.npz'),frozen_before_C0_evaluation_unix=time.time(),no_more_model_training=True));print('TRAINING COMPLETE selected',bestep,flush=True)
if __name__=='__main__':main()
