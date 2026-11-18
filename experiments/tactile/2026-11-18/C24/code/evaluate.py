"""C24 only: once frozen selected Real reconstruction + frozen C0; no Natural signal access or fitting."""
import ast,csv,json,time,hashlib,zipfile,sys,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
O=Path(__file__).resolve().parents[1];C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');C13=Path('outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18');BASE=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16')
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rs):
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def extract(path,name,env):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name];assert len(nodes)==1;exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path), 'exec'),env);return env[name]
def main():
 assert not (O/'evaluation_registration.json').exists(),'New result directory required; do not overwrite prior run'
 start=time.time();torch.set_num_threads(2);save('evaluation_registration.json',dict(scope='C23 target-side Real reconstruction readout diagnostic, NOT Natural Sim PRIMARY',inputs=[str(O/'validation_outputs.npz'),str(BASE/'real_validation/index.json')],candidate='unique C24 selected frozen Encoder/Decoder reconstruction; generated before any C0 evaluation',classifier_registry=str(C3/'c0/frozen.json'),instances=11,preprocessing='unchanged original log1p/Scaler/channel order/0.5 threshold',labels='Original Real reference=1, generated/reconstructed comparison side=0, exactly as C11 A. Both raw source domains remain Real; reconstruction is NOT relabelled Natural Sim.',criteria=dict(BA=[.45,.55],AUC=[.475,.525],D_min=.9,D='1-2*abs(BA-.5)',BA_D_redundant=True),history='C11 A Real-reconstruction / F Natural historical METRICS ONLY; C13 Natural historical METRICS ONLY; no Natural data/array inference',kNN_reference='Required original frozen C0 Train reference embedded in checkpoint, no new Natural files opened',budget_seconds=300,budget_bytes=200000000,bootstrap_replicates=300,new_probe_training=False,new_GMM_Translator_GAN_Diffusion_VAE=False,Test_Holdout_read=False,autostart_next=False))
 def budget():
  assert time.time()-start<300,'C23 5min budget'
  assert sum(p.stat().st_size for p in O.rglob('*') if p.is_file())<200000000,'C23 200MB budget'
 frozen=json.loads((C3/'c0/frozen.json').read_text());protected=dict(frozen['files']);files=[C13/'checkpoint.pt',C13/'validation_outputs.npz',C13/'integrity.json',C13/'protocol_clarification.json',BASE/'real_validation/index.json',Path('outputs/tactile_p35_deep_probe_2026-11-15/code/run.py'),Path('apps/evaluate_tactile_712d_gan.py'),Path('outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py'),Path('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/code/common11.py')]
 for p in files:protected[str(p)]=sha(p)
 assert all(sha(p)==h for p,h in frozen['files'].items());original=json.loads((C13/'integrity.json').read_text())['files'];assert sha(C13/'validation_outputs.npz')==original['validation_outputs.npz'];assert sha(C13/'checkpoint.pt')==original['checkpoint.pt'];save('evaluation_protected_hashes_before.json',protected); frozen24=json.loads((O/'training_freeze.json').read_text()); assert sha(O/'checkpoint.pt')==frozen24['checkpoint_sha256']; assert sha(O/'validation_outputs.npz')==frozen24['outputs_sha256']
 # Extract definitions without importing GAN/VAE/Translator or any training dispatch.
 env=dict(np=np,torch=torch,nn=nn,zipfile=zipfile)
 Probe=extract('apps/evaluate_tactile_712d_gan.py','Probe',env);DeepProbe=extract('outputs/tactile_p35_deep_probe_2026-11-15/code/run.py','DeepProbe',env)
 getrows=extract('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py','npz_rows',env)
 weighted=extract('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/code/common11.py','weighted_metrics',env);boot=extract('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/code/common11.py','bootstrap_weights',env)
 spec=importlib.util.spec_from_file_location('c23_original_metrics','outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py');metricsmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(metricsmod)
 idx=json.loads((BASE/'real_validation/index.json').read_text());ix=np.asarray(idx['indices'],int);assert len(ix)==780
 real=getrows(idx['source'],'real',ix);groups=getrows(idx['source'],'real_group',ix);stages=getrows(idx['source'],'real_stage',ix);units=getrows(idx['source'],'real_unit',ix)
 z=np.load(O/'validation_outputs.npz');gen=z['values'];assert gen.shape==real.shape==(780,4,712);assert np.array_equal(z['groups'],groups) and np.array_equal(z['stage'],stages);assert np.isfinite(real).all() and np.isfinite(gen).all() and (gen>=0).all() and (real>=0).all()
 save('data_contract.json',dict(Real_windows=780,reconstructed_windows=780,shape=list(real.shape),Session_count=len(set(groups)),Trial_count=len(set('/'.join(str(u).split('/')[:2]) for u in units)),numeric_loading='only exact registered780 Real Validation row indices streamed from source NPZ; other numeric rows/keys not materialized',source_npz=idx['source'],source_groups_matched=True,no_Natural_dataset_opened=True,no_Test_Holdout_rows_selected=True,generated_model_sha256=sha(O/'checkpoint.pt'),generated_values_sha256=sha(O/'validation_outputs.npz')))
 sc=np.load(C3/'c0/scaler.npz');assert str(sc['transform'])=='log1p';x=((np.log1p(np.concatenate([gen,real]))-sc['mean'])/sc['scale']).astype('float32');y=np.r_[np.zeros(780,int),np.ones(780,int)];jointgroups=np.r_[groups,groups];jointstages=np.r_[stages,stages];w,ga=boot(y,jointgroups);assert ga['shared_Real_source_groups']==4 and ga['Real_only_groups']==0 and ga['generated_only_groups']==0;save('bootstrap_group_audit.json',ga)
 (O/'predictions').mkdir(exist_ok=True);rows=[];stage_rows=[]
 for e in frozen['instances']:
  budget();name,kind=e['name'],e['kind'];p=C3/'c0'/name
  with torch.inference_mode():
   if kind=='logistic':
    ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False);prob=(torch.from_numpy(x.reshape(len(x),-1)).double()@ck['weight']+ck['bias']).sigmoid().numpy()
   else:
    if kind=='knn':
     ck=np.load(p/'checkpoint.npz');assert int(ck['k'])==5;ref=torch.from_numpy(ck['values'].reshape(len(ck['labels']),-1)).double();lab=torch.from_numpy(ck['labels']).double();rn=(ref*ref).sum(1)
    else:
     ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False);model=DeepProbe(kind);model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
    parts=[]
    for i in range(0,len(x),128):
     q=torch.from_numpy(x[i:i+128])
     if kind=='knn':q=q.reshape(len(q),-1).double();dist=(q*q).sum(1)[:,None]+rn[None,:]-2*q@ref.T;parts.append(lab[dist.topk(5,largest=False,dim=1).indices].mean(1).numpy())
     else:parts.append(model(q).sigmoid().numpy())
    prob=np.concatenate(parts)
  m=metricsmod.metrics(y,prob);check=weighted(y,prob,np.ones((1,len(y))));assert abs(check[0][0]-m['balanced_accuracy'])<1e-10 and abs(check[1][0]-m['roc_auc'])<1e-10
  bs=weighted(y,prob,w);row=dict(instance=name,**m,BA_pass=.45<=m['balanced_accuracy']<=.55,AUC_pass=.475<=m['roc_auc']<=.525,D_pass=m['domain_confusion']>=.9,passed_joint=.45<=m['balanced_accuracy']<=.55 and .475<=m['roc_auc']<=.525 and m['domain_confusion']>=.9,reconstruction_predicted_Real_fraction=float((prob[:780]>=.5).mean()),reference_predicted_Real_fraction=float((prob[780:]>=.5).mean()),reconstruction_score_mean=float(prob[:780].mean()),reference_score_mean=float(prob[780:].mean()))
  for key,arr in zip(['BA','AUC','D'],bs):row[key+'_CI95_low'],row[key+'_CI95_high']=map(float,np.quantile(arr,[.025,.975]))
  rows.append(row);np.savez_compressed(O/'predictions'/f'{name}.npz',probabilities=prob,comparison_labels=y,source_domain=np.full(len(y),'Real'),groups=jointgroups,stages=jointstages)
  for st in sorted(set(stages)):
   mask=jointstages==st;stage_rows.append(dict(instance=name,stage=str(st),**metricsmod.metrics(y[mask],prob[mask])))
  table('metrics.csv',rows);table('stage_metrics.csv',stage_rows);print(name,m,'pass',row['passed_joint'],flush=True)
 # Output diagnostics reuse the already-read paired Real arrays; never alter C0 input.
 stats=[];weak=[];patch=[];frame=[];session=[]
 def summary(a):
  pos=a[a>0];m=a>0;prev=m[:,:-1];nxt=m[:,1:];u=a[:,:-1].astype('float64').ravel();v=a[:,1:].astype('float64').ravel();pair=(u>0)&(v>0)
  return dict(windows=len(a),finite=bool(np.isfinite(a).all()),nonnegative=bool((a>=0).all()),zero_fraction=float((a==0).mean()),positive_count=len(pos),positive_mean=float(pos.mean()),**{f'positive_P{q}':float(np.quantile(pos,q/100)) for q in [1,10,50,90,95,99]},maximum=float(a.max()),transition_01=float((~prev&nxt).sum()/max((~prev).sum(),1)),transition_10=float((prev&~nxt).sum()/max(prev.sum(),1)),mean_abs_frame_delta=float(np.abs(v-u).mean()),lag1_correlation=float(np.corrcoef(u,v)[0,1]),both_positive_mean_abs_frame_delta=float(np.abs(v[pair]-u[pair]).mean()),unique_numeric_windows=len({v.tobytes() for v in a}))
 err=(gen.astype('float64')-real.astype('float64'))**2;rm=real>0;gm=gen>0;inter=(rm&gm).sum((1,2));union=(rm|gm).sum((1,2));ious=np.divide(inter,union,out=np.ones(len(real)),where=union>0);total=err.sum();masks={'both_positive':rm&gm,'missed_positive':rm&~gm,'generated_only_positive':~rm&gm,'both_zero':~rm&~gm};decomp=[dict(component=k,squared_error_sum=float(err[v].sum()),share_of_full_squared_error=float(err[v].sum()/total),fraction_of_all_Taxels=float(v.mean())) for k,v in masks.items()];table('reconstruction_error_decomposition.csv',decomp)
 save('reconstruction.json',dict(MSE=float(err.mean()),Mask_IoU=float(ious.mean()),finite=True,nonnegative=True,alignment='same Real window/Taxel/frame',no_amplitude_clipping=True,interpretation='decomposition descriptive; positive-only errors conditional on both masks are not isolated amplitude-head causal errors'))
 for label,a in [('Real',real),('C13_reconstructed_Real',gen)]:
  stats.append(dict(domain=label,**summary(a)))
  for fam,t in [('absolute',v) for v in [0,1e-8,1e-6,1e-4,.001,.01]]+[('relative_frame_peak',v) for v in [.001,.01,.05]]:
   th=t if fam=='absolute' else t*a.max(-1,keepdims=True);strong=a>th;active=np.stack([strong[:,:,p0:p1].any(-1) for p0,p1 in [(0,32),(32,64),(64,120),(120,152),(152,184),(184,240),(240,272),(272,304),(304,360),(360,392),(392,424),(424,480),(480,512),(512,544),(544,600),(600,712)]],-1)
   weak.append(dict(domain=label,family=fam,threshold=t,positive_below_threshold_fraction=float(((a>0)&~strong).sum()/max((a>0).sum(),1)),active_Taxel_fraction=float(strong.mean()),mean_active_Patches=float(active.sum(-1).mean()),used_for_C0=False))
 for t in range(4):frame.append(dict(frame=t,paired_MSE=float(err[:,t].mean()),paired_Mask_IoU=float(np.divide((rm[:,t]&gm[:,t]).sum(1),(rm[:,t]|gm[:,t]).sum(1),out=np.ones(780),where=(rm[:,t]|gm[:,t]).sum(1)>0).mean())))
 for g in sorted(set(groups)):
  ix0=groups==g;session.append(dict(Session=str(g),windows=int(ix0.sum()),paired_MSE=float(err[ix0].mean()),paired_Mask_IoU=float(ious[ix0].mean())))
 table('output_statistics.csv',stats);table('weak_response_statistics.csv',weak);table('frame_reconstruction.csv',frame);table('Session_reconstruction.csv',session)
 # Comparison only against existing C23 scalar metrics.
 old={r['instance']:r for r in csv.DictReader(Path('outputs/tactile_c23_real_reconstruction_c0_2026-11-18/metrics.csv').open())};comparisons=[]
 for r in rows:
  p=old[r['instance']];assert abs(float(p['recall_real'])-r['recall_real'])<1e-12
  cr=dict(instance=r['instance'])
  for label,rr in [('C23',p),('C24',r)]:
   for f in ['balanced_accuracy','roc_auc','domain_confusion','passed_joint']:cr[label+'_'+f]=rr[f]
  cr['AUC_distance_to_half_change']=abs(r['roc_auc']-.5)-abs(float(p['roc_auc'])-.5);comparisons.append(cr)
 table('C23_C24_comparison.csv',comparisons);budget();assert all(sha(p)==h for p,h in protected.items());assert sha(O/'checkpoint.pt')==frozen24['checkpoint_sha256'];save('evaluation_integrity.json',dict(all_protected_unchanged=True,files=protected,C0_not_refit=True,scaler_not_refit=True,Test_Holdout_not_accessed=True,Natural_tactile_files_not_accessed=True,selected_C24_model_frozen_before_C0=True))
 save('evaluation_complete.json',dict(joint_pass_count=sum(r['passed_joint'] for r in rows),instances=11,all_11_joint_passed=all(r['passed_joint'] for r in rows),scope='REAL_RECONSTRUCTION_READOUT_DIAGNOSTIC_ONLY_NOT_SIM_PRIMARY',BA_pass_count=sum(r['BA_pass'] for r in rows),AUC_pass_count=sum(r['AUC_pass'] for r in rows),D_pass_count=sum(r['D_pass'] for r in rows),wall_seconds=time.time()-start,new_Probe_fits=0,stopped_after_C24=True))
 print('C24 EVALUATION COMPLETE',sum(r['passed_joint'] for r in rows),'/11',flush=True)
if __name__=='__main__':main()
