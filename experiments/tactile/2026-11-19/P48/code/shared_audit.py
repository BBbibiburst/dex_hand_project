from audit_utils import *
import torch,sys,importlib.util
spec=importlib.util.spec_from_file_location('vae',P44/'code/vae.py');vae=importlib.util.module_from_spec(spec);sys.modules['vae']=vae;spec.loader.exec_module(vae);spec=importlib.util.spec_from_file_location('p48_shared_audit',P44/'code/model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);torch.set_num_threads(1);rows=[];calls=[]
for group in ['A','B','C']:
 source=P44/'generator.pt' if group=='B' else P45/group/'generator.pt';ck=torch.load(source,map_location='cpu',weights_only=False);G=m.Generator(**ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);before={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in G.field.state_dict().items()};ids={name:id(param) for name,param in G.field.named_parameters()};z=torch.from_numpy(np.load(O/f'paths/{group}_K8.npz')['source_latent'][:2]);counter=[]
 def hook(module,args):
  current={name:id(param) for name,param in module.named_parameters()};assert current==ids;counter.append(dict(module_id=id(module),parameter_ids=current,time=float(args[2])))
 handle=G.field.register_forward_pre_hook(hook)
 for K in [1,2,4,8]:
  counter.clear()
  with torch.inference_mode():G.integrate(z,K)
  assert len(counter)==K and len({v['module_id'] for v in counter})==1;calls.append(dict(group=group,K=K,actual_calls=counter.copy(),same_parameter_objects=True))
  for name,value in G.field.state_dict().items():
   h=hashlib.sha256(value.numpy().tobytes()).hexdigest();assert h==before[name];rows.append(dict(group=group,K=K,parameter=name,shape=str(tuple(value.shape)),sha256=h,unchanged=True))
 handle.remove()
table(O/'shared_parameter_hashes.csv',rows);save(O/'shared_parameter_call_trace.json',calls);print('SHARED PARAMETER AUDIT PASS',flush=True)
