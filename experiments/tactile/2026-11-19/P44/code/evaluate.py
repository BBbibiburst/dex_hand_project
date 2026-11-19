from c0_support import *
from model import Generator,Discriminator
from scipy.stats import spearmanr
import time
assert not (O/'evaluation_complete.json').exists();start=time.time();torch.set_num_threads(2);frozen=json.loads((O/'frozen.json').read_text());assert sha(O/'generator.pt')==frozen['generator_sha256'];assert sha(O/'discriminator.pt')==frozen['D_sha256'];ck=torch.load(O/'generator.pt',map_location='cpu',weights_only=False);G=Generator(**ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);dc=torch.load(O/'discriminator.pt',map_location='cpu',weights_only=False);D=Discriminator(dc['logscale']);D.load_state_dict(dc['model']);D.eval();D.requires_grad_(False);r=real('validation');n=natural('validation');(O/'outputs').mkdir(exist_ok=True);(O/'evaluation').mkdir(exist_ok=True);sharing=[];out={};sensitivity=[]
with torch.inference_mode():
 for K in [1,2,4,8]:
  calls=[];hook=G.field.register_forward_hook(lambda m,inp,out:calls.append(id(m)));one=G(torch.from_numpy(n['x'][:2]),K);hook.remove();assert len(calls)==K and len(set(calls))==1;sharing.append(dict(K=K,field_calls=K,unique_field_instances=1,field_id=calls[0],generator_checkpoint_sha256=sha(O/'generator.pt'),D_checkpoint_sha256=sha(O/'discriminator.pt')));ys=[]
  for i in range(0,len(n['x']),64):ys.append(G(torch.from_numpy(n['x'][i:i+64]),K).numpy())
  y=np.concatenate(ys);assert y.shape==(905,4,712) and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all();out[f'K{K}']=y;np.savez_compressed(O/'outputs'/f'K{K}_validation.npz',values=y,groups=n['g'],stage=n['s'],units=n['u'],tasks=n['tasks']);print('OUTPUT',K,y.shape,'max',float(y.max()),flush=True)
  # Fixed complete-window input perturbation diagnostic, not acceptance inputs; no extra model fitting.
  take=np.unique(np.linspace(0,len(n['x'])-1,64).round().astype(int));yh=[]
  for i in range(0,len(take),64):yh.append(G(torch.from_numpy(n['x'][take[i:i+64]]*.5),K).numpy())
  yh=np.concatenate(yh);sensitivity.append(dict(K=K,diagnostic_only=True,input_halving_output_RMSE=float(np.sqrt(((y[take].astype('float64')-yh)**2).mean())),windows=len(take),nonphysical_gain_test=True))
save(O/'parameter_sharing_final.json',sharing);table(O/'input_halving_diagnostic.csv',sensitivity)
# Reuse exact archived original baselines, no retraining or checkpoint selection.
C27=Path('outputs/tactile_c27_c3_route_improvement_2026-11-18');out['Natural_raw']=n['x'];out['C3']=np.load(C27/'baseline/translated_validation.npz')['values'];out['C27']=np.load(C27/'candidate/translated_validation.npz')['values'];allrows=[];rootrows=[];taskrows=[]
for label,y in out.items():
 rows=evaluate(O/'evaluation'/label,y,r,n)
 for row in rows:allrows.append(dict(candidate=label,**row))
 for filename,target in [('per_root_metrics.csv',rootrows),('per_task_metrics.csv',taskrows)]:
  for row in csv.DictReader((O/'evaluation'/label/filename).open()):target.append(dict(candidate=label,**row))
 table(O/'C0_metrics.csv',allrows);print('C0_CANDIDATE_DONE',label,flush=True)
table(O/'per_root_metrics.csv',rootrows);table(O/'per_task_metrics.csv',taskrows)
# Final shared D: full Train and Validation evaluation, allK, and raw-source positive-control diagnostic.
metrics=module('p44_original_metrics','outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py').metrics;drows=[]
for role in ['train','validation']:
 rr=real(role);nn=natural(role)
 for K in [0,1,2,4,8]:
  if K==0:fake=nn['x']
  elif role=='validation':fake=out[f'K{K}']
  else:
   yy=[]
   with torch.inference_mode():
    for i in range(0,len(nn['x']),128):yy.append(G(torch.from_numpy(nn['x'][i:i+128]),K).numpy())
   fake=np.concatenate(yy)
  probs=[];target=np.r_[np.zeros(len(fake),int),np.ones(len(rr['x']),int)]
  with torch.inference_mode():
   for a in [fake,rr['x']]:
    for i in range(0,len(a),128):probs.append(D(torch.from_numpy(a[i:i+128])).sigmoid().numpy())
  prob=np.concatenate(probs);mm=metrics(target,prob);bce=float(-(target*np.log(np.maximum(prob,1e-12))+(1-target)*np.log(np.maximum(1-prob,1e-12))).mean());drows.append(dict(role=role,K=K,**mm,Loss_BCE=bce,train_windows_Real=len(rr['x']),generated_windows=len(fake),source='same one final D; NOT original C0'))
  print('D_EVAL',role,K,mm['roc_auc'],flush=True)
 table(O/'D_metrics.csv',drows)
# Full output quality and perRoot/Task, with same index mapping caveat as historical experiments.
OFF=np.cumsum([0]+[32,32,56]*5+[112])
def summary(a,source):
 z=a.astype('float64');p=z[z>0];mask=z>0;u=mask[:,:-1];v=mask[:,1:];src=source>0;inter=(mask&src).sum((1,2));union=(mask|src).sum((1,2));mass=np.stack([z[:,:,l:h].sum(-1) for l,h in zip(OFF[:-1],OFF[1:])],-1);return dict(N=len(a),finite_fraction=float(np.isfinite(z).mean()),nonnegative_fraction=float((z>=0).mean()),zero_fraction=float((z==0).mean()),**{f'positive_P{q}':float(np.quantile(p,q/100)) if len(p) else None for q in [1,10,50,90,95,99]},maximum=float(z.max()),index_Mask_IoU=float(np.divide(inter,union,out=np.ones(len(z)),where=union>0).mean()),mean_active_Patches=float((mass>0).sum(-1).mean()),absolute_frame_delta=float(np.abs(np.diff(z,axis=1)).mean()),lag1_correlation=float(np.corrcoef(z[:,:-1].ravel(),z[:,1:].ravel())[0,1]),transition01=float((~u&v).sum()/max((~u).sum(),1)),transition10=float((u&~v).sum()/max(u.sum(),1)),mean_window_std=float(z.std(0).mean()),unique_numeric_windows=len({v.tobytes() for v in a}))
qrows=[];grows=[];response=[];patch=[]
for label,y in dict(out,Real=r['x']).items():
 source=r['x'] if label=='Real' else n['x'];qrows.append(dict(candidate=label,**summary(y,source)))
 if label!='Real':
  for key in ['g','tasks']:
   for val in sorted(set(n[key])):ix=n[key]==val;grows.append(dict(candidate=label,stratum=key,value=str(val),**summary(y[ix],source[ix])))
  for name,a,b in [('total_response',source.sum((1,2)),y.sum((1,2))),('positive_fraction',(source>0).mean((1,2)),(y>0).mean((1,2))),('frame_variation',np.abs(np.diff(source,axis=1)).mean((1,2)),np.abs(np.diff(y,axis=1)).mean((1,2)))]:rho=spearmanr(a,b).statistic;response.append(dict(candidate=label,feature=name,Spearman=float(rho) if np.isfinite(rho) else None,physical_correspondence=False))
 for p,(l,h) in enumerate(zip(OFF[:-1],OFF[1:])):a=y[:,:,l:h];patch.append(dict(candidate=label,patch=p,activation_fraction=float((a>0).mean()),mean_response=float(a.mean()),empty_patch_frames=float((a==0).all(-1).mean())))
table(O/'output_statistics.csv',qrows);table(O/'output_group_statistics.csv',grows);table(O/'input_response.csv',response);table(O/'patch_statistics.csv',patch)
for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items():assert sha(p)==h
assert sha(O/'generator.pt')==frozen['generator_sha256'];assert sha(O/'discriminator.pt')==frozen['D_sha256'];save(O/'evaluation_complete.json',dict(all_K_evaluated=True,shared_weights_preserved=True,no_model_or_C0_training_during_evaluation=True,C0_required_each=11,full_validations_retained=True,seconds=time.time()-start,Test_Holdout=False));print('P44 EVALUATION COMPLETE',flush=True)
