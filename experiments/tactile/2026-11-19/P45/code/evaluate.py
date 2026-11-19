from c0_support import *
from model import Generator
from discriminator import AblationDiscriminator
from scipy.stats import spearmanr
import time
P44=Path('outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18');assert (O/'training_complete.json').exists();assert not (O/'evaluation_complete.json').exists();torch.set_num_threads(2);start=time.time();r=real('validation');n=natural('validation');metrics=module('p45_frozen_metrics','outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py').metrics;allrows=[];drows=[];rootrows=[];taskrows=[];sharing=[];trajrows=[];outputs={};cfg=json.loads((O/'config.json').read_text());epsilon=cfg['epsilon'];tau=cfg['tau']
# B strictly reuses P44 scores/outputs, never retrained or relabelled as new evaluation.
for row in csv.DictReader((P44/'C0_metrics.csv').open()):
 if row['candidate'] in ['K1','K2','K4','K8']:allrows.append(dict(group='B',**row))
for row in csv.DictReader((P44/'D_metrics.csv').open()):
 if row['K']!='0':drows.append(dict(group='B',**row))
for file,target in [('per_root_metrics.csv',rootrows),('per_task_metrics.csv',taskrows)]:
 for row in csv.DictReader((P44/file).open()):
  if row['candidate'] in ['K1','K2','K4','K8']:value=row.pop('group');target.append(dict(group='B',stratum_value=value,**row))
for K in [1,2,4,8]:outputs['B',K]=np.load(P44/f'outputs/K{K}_validation.npz')['values']
take=np.concatenate([np.flatnonzero(n['g']==g)[np.linspace(0,sum(n['g']==g)-1,4).round().astype(int)] for g in sorted(set(n['g']))])
for group in ['A','C']:
 folder=O/group;(folder/'outputs').mkdir();(folder/'evaluation').mkdir();freeze=json.loads((folder/'frozen.json').read_text());assert sha(folder/'generator.pt')==freeze['generator_sha256'];assert sha(folder/'discriminator.pt')==freeze['D_sha256'];ck=torch.load(folder/'generator.pt',map_location='cpu',weights_only=False);G=Generator(**ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);dc=torch.load(folder/'discriminator.pt',map_location='cpu',weights_only=False);D=AblationDiscriminator(group,dc['logscale'],dc['epsilon'],dc['tau']);D.load_state_dict(dc['model']);D.eval();D.requires_grad_(False)
 with torch.inference_mode():
  zfixed=G.sim.posterior_mean(torch.from_numpy(n['x'][take]))
  for K in [1,2,4,8]:
   ys=[];calls=[];hook=G.field.register_forward_hook(lambda m,inp,out:calls.append(id(m)));G(torch.from_numpy(n['x'][:2]),K);hook.remove();assert len(calls)==K and len(set(calls))==1;sharing.append(dict(group=group,K=K,calls=K,one_field_instance=True,generator_sha256=freeze['generator_sha256'],D_sha256=freeze['D_sha256']))
   for i in range(0,905,64):ys.append(G(torch.from_numpy(n['x'][i:i+64]),K).numpy())
   y=np.concatenate(ys);assert y.shape==(905,4,712) and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all();outputs[group,K]=y;np.savez_compressed(folder/'outputs'/f'K{K}_validation.npz',values=y,groups=n['g'],stage=n['s'],units=n['u'],tasks=n['tasks']);print('OUTPUT',group,K,float(y.max()),flush=True)
   h=G.initial_state.expand(len(zfixed),-1,-1);states=[h.numpy().copy()];vel=[]
   for k in range(K):v,c=G.field(h,zfixed,k/K);old=h;h=h+v/K;states.append(h.numpy().copy());vel.append(v.numpy());trajrows.append(dict(group=group,K=K,step=k,time=k/K,state_norm_mean=float(torch.linalg.vector_norm(h.flatten(1),dim=1).mean()),velocity_norm_mean=float(torch.linalg.vector_norm(v.flatten(1),dim=1).mean()),step_delta_norm=float(torch.linalg.vector_norm((h-old).flatten(1),dim=1).mean()),finite=bool(torch.isfinite(h).all())))
   np.savez_compressed(folder/'outputs'/f'K{K}_trajectory.npz',states=np.stack(states,1),velocities=np.stack(vel,1),units=n['u'][take],groups=n['g'][take])
 for K in [1,2,4,8]:
  rows=evaluate(folder/'evaluation'/f'K{K}',outputs[group,K],r,n)
  for row in rows:allrows.append(dict(group=group,candidate=f'K{K}',**row))
  for file,target in [('per_root_metrics.csv',rootrows),('per_task_metrics.csv',taskrows)]:
   for row in csv.DictReader((folder/'evaluation'/f'K{K}'/file).open()):value=row.pop('group');target.append(dict(group=group,candidate=f'K{K}',stratum_value=value,**row))
  table(O/'C0_metrics.csv',allrows)
 # Full new-D evaluations, exact same source split, no refitting.
 for role in ['train','validation']:
  nr=natural(role);rr=real(role)
  for K in [1,2,4,8]:
   if role=='validation':fake=outputs[group,K]
   else:
    yy=[]
    with torch.inference_mode():
     for i in range(0,len(nr['x']),64):yy.append(G(torch.from_numpy(nr['x'][i:i+64]),K).numpy())
    fake=np.concatenate(yy)
   pp=[]
   with torch.inference_mode():
    for x in [fake,rr['x']]:
     for i in range(0,len(x),128):pp.append(D(torch.from_numpy(x[i:i+128])).sigmoid().numpy())
   prob=np.concatenate(pp);labels=np.r_[np.zeros(len(fake),int),np.ones(len(rr['x']),int)];mm=metrics(labels,prob);bce=float(-(labels*np.log(np.maximum(prob,1e-12))+(1-labels)*np.log(np.maximum(1-prob,1e-12))).mean());drows.append(dict(group=group,role=role,K=K,**mm,Loss_BCE=bce,train_windows_Real=len(rr['x']),generated_windows=len(fake),source='one final groupD, notC0'));print('D_EVAL',group,role,K,mm['roc_auc'],flush=True)
  table(O/'D_metrics.csv',drows)
 assert sha(folder/'generator.pt')==freeze['generator_sha256'];assert sha(folder/'discriminator.pt')==freeze['D_sha256']
table(O/'C0_metrics.csv',allrows);table(O/'per_root_metrics.csv',rootrows);table(O/'per_task_metrics.csv',taskrows);save(O/'parameter_sharing.json',sharing);table(O/'iteration_trajectory.csv',trajrows)
OFF=np.cumsum([0]+[32,32,56]*5+[112])
def stats(a,source):
 x=a.astype('float64');p=x[x>0];m=x>0;u=m[:,:-1];v=m[:,1:];b=source>0;inter=(m&b).sum((1,2));union=(m|b).sum((1,2));pa=np.stack([x[:,:,l:h].sum(-1) for l,h in zip(OFF[:-1],OFF[1:])],-1);return dict(N=len(x),shape=str(list(x.shape)),finite_fraction=float(np.isfinite(x).mean()),nonnegative_fraction=float((x>=0).mean()),zero_fraction=float((x==0).mean()),positive_mean=float(p.mean()) if len(p) else None,**{f'positive_P{q}':float(np.quantile(p,q/100)) if len(p) else None for q in [1,10,50,90,95,99]},maximum=float(x.max()),index_Mask_IoU=float(np.divide(inter,union,out=np.ones(len(x)),where=union>0).mean()),active_Patch_mean=float((pa>0).sum(-1).mean()),absolute_frame_delta=float(np.abs(np.diff(x,axis=1)).mean()),lag1_correlation=float(np.corrcoef(x[:,:-1].ravel(),x[:,1:].ravel())[0,1]),transition01=float((~u&v).sum()/max((~u).sum(),1)),transition10=float((u&~v).sum()/max(u.sum(),1)),mean_window_std=float(x.std(0).mean()),unique_numeric_windows=len({z.tobytes() for z in a}))
quality=[];groupquality=[];response=[];taxels=[];patches=[]
for (group,K),y in outputs.items():
 quality.append(dict(group=group,K=K,**stats(y,n['x'])))
 for key in ['g','tasks']:
  for val in sorted(set(n[key])):ix=n[key]==val;groupquality.append(dict(group=group,K=K,stratum=key,value=str(val),**stats(y[ix],n['x'][ix])))
 for name,a,b in [('total_response',n['x'].sum((1,2)),y.sum((1,2))),('positive_fraction',(n['x']>0).mean((1,2)),(y>0).mean((1,2))),('frame_variation',np.abs(np.diff(n['x'],axis=1)).mean((1,2)),np.abs(np.diff(y,axis=1)).mean((1,2)))]:rho=spearmanr(a,b).statistic;response.append(dict(group=group,K=K,feature=name,Spearman=float(rho) if np.isfinite(rho) else None,physical_mapping_certified=False))
 for j in range(712):taxels.append(dict(group=group,K=K,Taxel=j,activation_fraction=float((y[:,:,j]>0).mean()),mean_response=float(y[:,:,j].mean())))
 for p,(l,h) in enumerate(zip(OFF[:-1],OFF[1:])):z=y[:,:,l:h];patches.append(dict(group=group,K=K,Patch=p,activation_fraction=float((z>0).mean()),empty_frames=float((z==0).all(-1).mean()),mean_response=float(z.mean())))
quality.extend([dict(group='Real',K=0,**stats(r['x'],r['x'])),dict(group='Natural',K=0,**stats(n['x'],n['x']))]);table(O/'output_statistics.csv',quality);table(O/'output_group_statistics.csv',groupquality);table(O/'input_response.csv',response);table(O/'taxel_statistics.csv',taxels);table(O/'patch_statistics.csv',patches)
assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'evaluation_complete.json',dict(groups=['A','B','C'],C0_rows=len(allrows),B='original P44 results reused, not retrained',all_K_same_group_weights=True,no_model_or_probe_training=True,no_Test_Holdout=True,seconds=time.time()-start));print('P45 EVALUATION COMPLETE NO_P46',flush=True)
