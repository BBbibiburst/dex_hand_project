from common import *
import torch,time,importlib.util,sys
assert (O/'register_complete.json').exists() and not (O/'generate_complete.json').exists();start=time.time();torch.set_num_threads(2)
# Exact copied source; no monkey-patching/training. Load original dependency by controlled module name.
spec=importlib.util.spec_from_file_location('vae',P44/'code/vae.py');v=importlib.util.module_from_spec(spec);sys.modules['vae']=v;spec.loader.exec_module(v)
spec=importlib.util.spec_from_file_location('p48_frozen_original',P44/'code/model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[];audit=[];folder=O/'outputs';folder.mkdir(exist_ok=True);paths=O/'paths';paths.mkdir(exist_ok=True)
for group in ['A','B','C']:
 ckpath=P44/'generator.pt' if group=='B' else P45/group/'generator.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);G=m.Generator(**ck['scales']);G.load_state_dict(ck['model'],strict=True);G.eval();G.requires_grad_(False);sharedhash=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in G.field.state_dict().values())).hexdigest()
 for role in ['train','validation']:
  d=natural(role);x=d['x'];assert x.dtype==np.float32
  with torch.inference_mode():Z=torch.cat([G.sim.posterior_mean(torch.from_numpy(x[i:i+64])) for i in range(0,len(x),64)])
  for K in [1,2,4,8]:
   arrays=[];hidden=[]
   with torch.inference_mode():
    for i in range(0,len(x),64):
     z=Z[i:i+64];h=G.initial_state.expand(len(z),-1,-1);ys=[G.decode(h)];hs=[h]
     for step in range(K):vv,c=G.field(h,z,step/K);h=h+vv/K;ys.append(G.decode(h));hs.append(h)
     ys=torch.stack(ys,1);hs=torch.stack(hs,1);original=G.from_latent(z,K)[0];assert torch.equal(ys[:,-1],original);arrays.append(ys.numpy());hidden.append(hs.numpy())
   Y=np.concatenate(arrays);H=np.concatenate(hidden);assert Y.shape==(len(x),K+1,4,712) and np.isfinite(Y).all() and (Y>=0).all()
   name=f'{group}_K{K}';dest=folder/f'{name}_{role}.npz';np.savez_compressed(dest,values=Y[:,-1],groups=d['g'],stage=d['s'],units=d['u']);rows.append(dict(model=name,role=role,path=str(dest),sha256=sha(dest),shape=str(Y[:,-1].shape),input_role=role,frozen_checkpoint=str(ckpath),checkpoint_sha=sha(ckpath),optimizer_steps=0))
   if role=='validation':
    oldpath=candidate_paths()[name];old=np.load(oldpath)['values'];err=float(np.max(np.abs(Y[:,-1]-old)));assert np.allclose(Y[:,-1],old,rtol=1e-5,atol=1e-6),f'Frozen inference mismatch {name} {err}';audit.append(dict(model=name,K=K,field_hash=sharedhash,field_parameter_ids_shared=True,interval='0to1',dt=1/K,source_latent='same frozen Sim posterior mean',internal_h0='Real-coordinate zeros, NOT Original Sim',decoder='same frozen RealDecoder',max_difference_old_validation=err,bit_equal_old=np.array_equal(Y[:,-1],old),K_is_training_epochs=False));np.savez_compressed(paths/f'{name}.npz',decoded=Y,latent=H,source_latent=Z.numpy(),groups=d['g'],stage=d['s'],units=d['u'],times=np.linspace(0,1,K+1))
   print('FROZEN_INFERENCE',name,role,'seconds',round(time.time()-start,1),flush=True)
  del Z
 assert sharedhash==hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in G.field.state_dict().values())).hexdigest()
 del G
# Missing Train outputs generated legitimately; archived Validation arrays retained fordistribution metrics.
table(O/'output_manifest.csv',rows);table(O/'K_semantics_audit.csv',audit);assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'generate_complete.json',dict(seconds=time.time()-start,models=3,outputs=24,migration_epochs=0,migration_optimizer_steps=0,K_verified=True));print('P48 GENERATE COMPLETE',flush=True)
