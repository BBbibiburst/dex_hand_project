"""Read only registered development roles. No converter/Probe/C0 inference at import."""
import ast,csv,json,zipfile,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
O=Path(__file__).resolve().parents[1];B=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16');P44=Path('outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18');P45=Path('outputs/tactile_p45_mask_ablation_2026-11-18')
def _json_scalar(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,np.ndarray):return value.tolist()
 raise TypeError(f'Unsupported JSON value type:{type(value)}')
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=_json_scalar))
def table(p,rs):
 if not rs:return
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
env=dict(np=np,zipfile=zipfile);tree=ast.parse(Path('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='npz_rows'],type_ignores=[]),'original selective rows','exec'),env);rows=env['npz_rows']
def real(role):
 assert role in ['train','validation'];i=json.loads((B/f'real_{role}/index.json').read_text());return {k:rows(i['source'],'real'+s,i['indices']) for k,s in [('x',''),('g','_group'),('s','_stage'),('u','_unit')]}
def natural(role):
 assert role in ['train','validation'];z=np.load(B/f'natural_sim_{role}/windows.npz');m={r['window_id']:r for r in csv.DictReader((B/f'natural_sim_{role}/windows_manifest.csv').open())};return dict(x=z['values'],g=z['groups'],s=z['stage'],u=z['units'],task=np.array([m[str(u)]['task'] for u in z['units']]))
def candidate_paths():
 return {**{f'B_K{k}':P44/f'outputs/K{k}_validation.npz' for k in [1,2,4,8]},**{f'{g}_K{k}':P45/f'{g}/outputs/K{k}_validation.npz' for g in ['A','C'] for k in [1,2,4,8]}}
def inputs():
 n=natural('validation');result={'Original':n['x']}
 for name,p in candidate_paths().items():
  z=np.load(p);assert np.array_equal(z['groups'],n['g']) and np.array_equal(z['stage'],n['s']) and np.array_equal(z['units'],n['u']);result[name]=z['values']
 return n,result

def transform(x,mean,sd):return ((np.log1p(x).astype('float64').reshape(len(x),-1)-mean)/sd)
def distances(x,y,sigma):
 aa=cdist(x,x,'sqeuclidean');bb=cdist(y,y,'sqeuclidean');ab=cdist(x,y,'sqeuclidean');n,m=len(x),len(y);out={};kernels=[]
 for scale in [.5,1.,2.]:
  kxx=np.exp(-aa/(2*(sigma*scale)**2));kyy=np.exp(-bb/(2*(sigma*scale)**2));kxy=np.exp(-ab/(2*(sigma*scale)**2));v=float(kxx.mean()+kyy.mean()-2*kxy.mean());u=float((kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(m*(m-1))-2*kxy.mean()) if min(n,m)>1 else None;kernels.append((v,u));out[f'MMD_rbf_{scale:g}']=np.sqrt(max(v,0));out[f'MMD2V_rbf_{scale:g}']=v;out[f'MMD2U_rbf_{scale:g}']=u
 v=float(np.mean([z[0] for z in kernels]));u=float(np.mean([z[1] for z in kernels])) if kernels[0][1] is not None else None;energy=float(2*np.sqrt(ab).mean()-np.sqrt(aa).mean()-np.sqrt(bb).mean());out.update(MMD=np.sqrt(max(v,0)),MMD2V=v,MMD2U=u,Energy=np.sqrt(max(energy,0)),Energy_V=energy);return out
METRICS=['MMD','MMD_rbf_0.5','MMD_rbf_1','MMD_rbf_2','Energy']
def group_seed(name):return 4601+int(hashlib.sha256(str(name).encode()).hexdigest()[:8],16)
