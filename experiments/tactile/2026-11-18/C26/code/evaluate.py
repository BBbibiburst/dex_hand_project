"""Original frozen11C0 once, only after C26 response screen and output freeze."""
import ast,json,csv,hashlib,zipfile,time,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
O=Path(__file__).resolve().parents[1];C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');B=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16')
def save(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(n,rs):
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def extract(path,name,env):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name];assert len(nodes)==1;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);return env[name]
def main():
 assert not (O/'C0_complete.json').exists();screen=json.loads((O/'StageD_screen.json').read_text());assert screen['C0_allowed'] and not screen['hard_stop'];freeze=json.loads((O/'inference_freeze.json').read_text());assert sha(O/'Natural_validation_outputs.npz')==freeze['Natural_outputs_sha256'];assert sha(O/'Real_validation_outputs.npz')==freeze['Real_outputs_sha256'];torch.set_num_threads(2)
 env=dict(np=np,torch=torch,nn=nn,zipfile=zipfile);Deep=extract('outputs/tactile_p35_deep_probe_2026-11-15/code/run.py','DeepProbe',env);get=extract('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py','npz_rows',env);weighted=extract('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/code/common11.py','weighted_metrics',env);boot=extract('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/code/common11.py','bootstrap_weights',env)
 spec=importlib.util.spec_from_file_location('c26_original_metrics','outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py');met=importlib.util.module_from_spec(spec);spec.loader.exec_module(met)
 idx=json.loads((B/'real_validation/index.json').read_text());real=get(idx['source'],'real',idx['indices']);rg=get(idx['source'],'real_group',idx['indices']);rs=get(idx['source'],'real_stage',idx['indices']);nz=np.load(B/'natural_sim_validation/windows.npz');natural=nz['values'];ng=nz['groups'];ns=nz['stage'];gen=np.load(O/'Natural_validation_outputs.npz')['values'];rec=np.load(O/'Real_validation_outputs.npz')['values'];assert len(real)==780 and len(gen)==905
 sc=np.load(C3/'c0/scaler.npz');assert str(sc['transform'])=='log1p';candidates={'A_original_Natural':natural,'B_C26_zero_shot_Natural':gen,'C_Real_self_reconstruction_reference':rec};data={}
 for label,x0 in candidates.items():
  N=len(x0);x=((np.log1p(np.concatenate([x0,real]))-sc['mean'])/sc['scale']).astype('float32');y=np.r_[np.zeros(N,int),np.ones(780,int)];g=np.r_[rg,rg] if label.startswith('C_') else np.r_[np.char.add('Natural:',ng),np.char.add('Real:',rg)];st=np.r_[rs,rs] if label.startswith('C_') else np.r_[ns,rs];w,ga=boot(y,g);data[label]=dict(x=x,y=y,g=g,s=st,w=w);save(label+'_bootstrap_contract.json',ga)
 frozen=json.loads((C3/'c0/frozen.json').read_text());assert all(sha(p)==h for p,h in frozen['files'].items());rows=[];stagerows=[];(O/'predictions').mkdir(exist_ok=True)
 for item in frozen['instances']:
  name,kind=item['name'],item['kind'];p=C3/'c0'/name
  with torch.inference_mode():
   if kind=='logistic':ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
   elif kind=='knn':ck=np.load(p/'checkpoint.npz');ref=torch.from_numpy(ck['values'].reshape(len(ck['labels']),-1)).double();lab=torch.from_numpy(ck['labels']).double();rn=(ref*ref).sum(1);assert int(ck['k'])==5
   else:ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False);model=Deep(kind);model.load_state_dict(ck['model']);model.requires_grad_(False);model.eval()
   for label,d in data.items():
    x,y=d['x'],d['y'];parts=[]
    for i in range(0,len(x),128):
     q=torch.from_numpy(x[i:i+128])
     if kind=='logistic':out=(q.flatten(1).double()@ck['weight']+ck['bias']).sigmoid()
     elif kind=='knn':q=q.flatten(1).double();dist=(q*q).sum(1)[:,None]+rn[None,:]-2*q@ref.T;out=lab[dist.topk(5,largest=False,dim=1).indices].mean(1)
     else:out=model(q).sigmoid()
     parts.append(out.numpy())
    prob=np.concatenate(parts);mm=met.metrics(y,prob);bs=weighted(y,prob,d['w']);row=dict(candidate=label,instance=name,**mm,passed_joint=.45<=mm['balanced_accuracy']<=.55 and .475<=mm['roc_auc']<=.525 and mm['domain_confusion']>=.9)
    for k,a in zip(['BA','AUC','D'],bs):row[k+'_CI_low'],row[k+'_CI_high']=map(float,np.quantile(a,[.025,.975]))
    rows.append(row);np.savez_compressed(O/'predictions'/f'{label}_{name}.npz',probabilities=prob,labels=y,groups=d['g'],stage=d['s']);print(label,name,mm['roc_auc'],row['passed_joint'],flush=True)
    # Same literal stage labels stratify; NOT same physical condition certification.
    for stage in sorted(set(rs)&set(ns if not label.startswith('C_') else rs)):
     ix=d['s']==stage;stagerows.append(dict(candidate=label,instance=name,stage=str(stage),same_physical_stage_certified=False,**met.metrics(y[ix],prob[ix])))
  table('C0_metrics.csv',rows)
 table('C0_stage_metrics.csv',stagerows);counts={label:sum(r['passed_joint'] for r in rows if r['candidate']==label) for label in data};save('C0_complete.json',dict(joint_pass_counts=counts,all_11_pass_B=counts['B_C26_zero_shot_Natural']==11,new_probe_fits=0,weights_scaler_log1p_order_label_threshold_unchanged=True,scope='Natural PRIMARY zero-shot baseline; Real self reconstruction reference not transfer',finished_unix=time.time()));assert all(sha(p)==h for p,h in frozen['files'].items());assert sha(O/'Natural_validation_outputs.npz')==freeze['Natural_outputs_sha256'];print('C26 C0 COMPLETE',counts,flush=True)
if __name__=='__main__':main()
