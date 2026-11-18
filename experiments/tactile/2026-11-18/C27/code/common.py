import ast,csv,json,hashlib,importlib.util,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
O=Path(__file__).resolve().parents[1];C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');B=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16')
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False))
def table(p,rs):
 if not rs:return
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
M=module('c27_original_models',C3/'code/models.py')
def extract(path,name,env):
 tr=ast.parse(Path(path).read_text());nodes=[n for n in tr.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name];assert len(nodes)==1;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);return env[name]
env=dict(np=np,torch=torch,nn=nn,zipfile=zipfile);get=extract('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py','npz_rows',env)
def real(role):
 idx=json.loads((B/f'real_{role}/index.json').read_text());return {k:get(idx['source'],'real'+suffix,idx['indices']) for k,suffix in [('x',''),('g','_group'),('s','_stage'),('u','_unit')]}
def natural(role):
 z=np.load(B/f'natural_sim_{role}/windows.npz');meta={r['window_id']:r for r in csv.DictReader((B/f'natural_sim_{role}/windows_manifest.csv').open())};return dict(x=z['values'],g=z['groups'],s=z['stage'],u=z['units'],tasks=np.array([meta[str(u)]['task'] for u in z['units']]))
def load():
 nets=[]
 for name in ['real','sim']:
  ck=torch.load(C3/f'{name}_vae/checkpoint.pt',map_location='cpu',weights_only=False);n=M.DomainVAE(ck['encoder_scale'],ck['positive_scale']);n.load_state_dict(ck['model']);n.eval();n.requires_grad_(False);nets.append(n)
 ck=torch.load(C3/'translator/checkpoint.pt',map_location='cpu',weights_only=False);t=M.Translator();t.load_state_dict(ck['model']);t.eval();return *nets,t
@torch.inference_mode()
def convert(R,S,T,x):
 return np.concatenate([R.decoder.deterministic_decode(T(S.posterior_mean(torch.from_numpy(x[i:i+64])))).numpy() for i in range(0,len(x),64)])
def evaluate(dest,gen,reference,nat):
 dest=Path(dest);dest.mkdir(exist_ok=True);assert not (dest/'metrics.csv').exists();Probe=extract('apps/evaluate_tactile_712d_gan.py','Probe',env);Deep=extract('outputs/tactile_p35_deep_probe_2026-11-15/code/run.py','DeepProbe',env);weighted=extract('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/code/common11.py','weighted_metrics',env);boot=extract('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/code/common11.py','bootstrap_weights',env);metrics=module('c27_metrics','outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py').metrics;sc=np.load(C3/'c0/scaler.npz');assert str(sc['transform'])=='log1p';x=((np.log1p(np.concatenate([gen,reference['x']]))-sc['mean'])/sc['scale']).astype('float32');y=np.r_[np.zeros(len(gen),int),np.ones(len(reference['x']),int)];g=np.r_[np.char.add('Sim:',nat['g']),np.char.add('Real:',reference['g'])];w,contract=boot(y,g);save(dest/'bootstrap_contract.json',contract);frozen=json.loads((C3/'c0/frozen.json').read_text());rows=[];rootrows=[];taskrows=[];(dest/'predictions').mkdir(exist_ok=True)
 for e in frozen['instances']:
  name,kind=e['name'],e['kind'];p=C3/'c0'/name
  with torch.inference_mode():
   if kind=='logistic':ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
   elif kind=='knn':ck=np.load(p/'checkpoint.npz');ref=torch.from_numpy(ck['values'].reshape(len(ck['labels']),-1)).double();lab=torch.from_numpy(ck['labels']).double();rn=(ref*ref).sum(1);assert int(ck['k'])==5
   else:ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False);net=Deep(kind);net.load_state_dict(ck['model']);net.eval();net.requires_grad_(False)
   parts=[]
   for i in range(0,len(x),128):
    q=torch.from_numpy(x[i:i+128])
    if kind=='logistic':pr=(q.flatten(1).double()@ck['weight']+ck['bias']).sigmoid()
    elif kind=='knn':q=q.flatten(1).double();dd=(q*q).sum(1)[:,None]+rn[None]-2*q@ref.T;pr=lab[dd.topk(5,largest=False,dim=1).indices].mean(1)
    else:pr=net(q).sigmoid()
    parts.append(pr.numpy())
   prob=np.concatenate(parts)
  mm=metrics(y,prob);bs=weighted(y,prob,w);r=dict(instance=name,**mm,passed_joint=.45<=mm['balanced_accuracy']<=.55 and .475<=mm['roc_auc']<=.525 and mm['domain_confusion']>=.9)
  for k,a in zip(['BA','AUC','D'],bs):r[k+'_CI_low'],r[k+'_CI_high']=map(float,np.quantile(a,[.025,.975]))
  rows.append(r);np.savez_compressed(dest/'predictions'/f'{name}.npz',probabilities=prob,labels=y,groups=g)
  for vals,target in [(nat['g'],rootrows),(nat['tasks'],taskrows)]:
   for v in set(vals):
    ix=np.r_[vals==v,np.ones(len(reference['x']),bool)];target.append(dict(instance=name,group=str(v),reference='all780 Real Validation, NOT matched physical condition',**metrics(y[ix],prob[ix])))
  print(dest.name,name,round(mm['roc_auc'],5),r['passed_joint'],flush=True)
 table(dest/'metrics.csv',rows);table(dest/'per_root_metrics.csv',rootrows);table(dest/'per_task_metrics.csv',taskrows);assert all(sha(p)==h for p,h in frozen['files'].items());save(dest/'complete.json',dict(joint_pass_count=sum(r['passed_joint'] for r in rows),instances=11,C0_not_refit=True));return rows
