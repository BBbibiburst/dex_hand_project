"""C26 shared frozen C13 inference only. No fitting, OT, GMM, or Test access."""
import ast,csv,json,time,hashlib,zipfile,subprocess,importlib.util
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
O=Path(__file__).resolve().parents[1];P=Path('outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18');C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');B=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16');C25=Path('outputs/tactile_c25_latent_gmm_ot_feasibility_2026-11-18');P41=Path('outputs/tactile_p41_real_informed_gmm_forward_2026-11-15');P40=Path('outputs/tactile_p40_shared_features_2026-11-15')
def save(n,d):(O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def table(n,rs):
 if not rs:return
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 assert not (O/'registration.json').exists();start=time.time();torch.set_num_threads(2);reg=dict(date='2026-11-18',roles=['train','validation'],shared_checkpoint=str(P/'checkpoint.pt'),Encoder='original frozen C13 posterior mean, CPU batch64 for all domains',Decoder='original frozen C13 deterministic_decode, same GPU/CPU device batch64 for both validation roles',numeric_scope='unchanged canonical INDEX space; physical channel correspondence UNKNOWN is retained, not certified',Stage0_hard_stops=['nonfinite or negative input','shape/order metadata mismatch','nonincreasing timestamps','source indices misalignment'],StageD_hard_stops=['nonfinite or negative output','constant output: <2 unique windows or mean window-std<1e-8 despite varying input','>=95% of nonzero-input windows decoded completely zero'],StageD_other_diagnostics='Do not claim semantic/physical retention from numeric gates or correlation alone; severe retention problems may still invalidate result',no_model_training=True,no_scaler_fitting=True,no_GMM=True,no_OT=True,no_C0_fitting=True,no_input_edit=True,no_Test_Holdout=True,no_server_change=True,bootstrap=300,seed=2601,budget_seconds=600,budget_bytes=500000000,primary_selection='single candidate only, original11C0 once if A-D numeric gates pass',auto_C27=False,git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());save('registration.json',reg)
 protected=dict(json.loads((C3/'c0/frozen.json').read_text())['files']);files=[P/'checkpoint.pt',P/'validation_outputs.npz',C3/'code/models.py',P41/'real_sim_patch_mapping.csv',P41/'real_sim_taxel_mapping.csv',P41/'calibration_audit.json',P40/'selected_model/checkpoint.pt',P40/'code/model.py',P40/'p40_report.md',Path('apps/train_tactile_712d_gan.py'),Path('source/sensors/tactile/fitting/layout.py'),Path('source/sensors/tactile/signal_processing.py')]
 for role in ['train','validation']:files.extend([B/f'real_{role}/index.json',B/f'natural_sim_{role}/windows.npz',B/f'natural_sim_{role}/windows_manifest.csv'])
 for p in files:protected[str(p)]=sha(p)
 assert all(sha(p)==h for p,h in protected.items());save('protected_hashes_before.json',protected)
 env=dict(np=np,zipfile=zipfile);tree=ast.parse(Path('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='npz_rows'],type_ignores=[]),'rows','exec'),env);get=env['npz_rows'];sets={};contract=[];timeaudit=[]
 for role in ['train','validation']:
  idx=json.loads((B/f'real_{role}/index.json').read_text());d=dict(x=get(idx['source'],'real',idx['indices']),g=get(idx['source'],'real_group',idx['indices']),s=get(idx['source'],'real_stage',idx['indices']),u=get(idx['source'],'real_unit',idx['indices']),tasks=np.full(len(idx['indices']),'UNKNOWN_REAL_TASK'))
  ct=np.load(Path('outputs/tactile_c24_temporal_amplitude_repair_2026-11-18')/f'{role}_time_contract.npz');assert np.array_equal(ct['units'],d['u']);times=ct['timestamps'];assert np.all(np.diff(times,axis=1)>0);timeaudit.append(dict(domain='Real',role=role,dt_min=float(np.diff(times,axis=1).min()),dt_P50=float(np.median(np.diff(times,axis=1))),dt_max=float(np.diff(times,axis=1).max()),source='C24 raw processed/sync_id exact audit',fixed_20Hz=False));sets['Real_'+role]=d
  z=np.load(B/f'natural_sim_{role}/windows.npz');d=dict(x=z['values'],g=z['groups'],s=z['stage'],u=z['units']);meta=list(csv.DictReader((B/f'natural_sim_{role}/windows_manifest.csv').open()));byid={r['window_id']:r for r in meta};tt=[];dts=[]
  for i,u in enumerate(d['u']):
   r=byid[str(u)];assert r['tactile_sha256']==hashlib.sha256(d['x'][i].tobytes()).hexdigest();assert int(r['last_frame'])-int(r['first_frame'])==3;span=float(r['timestamp_last'])-float(r['timestamp_first']);assert span>0;dts.append(span/3);tt.append(r['task'])
  d['tasks']=np.array(tt);sets['Natural_'+role]=d;timeaudit.append(dict(domain='Natural',role=role,dt_min=float(np.min(dts)),dt_P50=float(np.median(dts)),dt_max=float(np.max(dts)),source='registered4 consecutive source frames + timestamp span',fixed_20Hz=True))
 for name,d in sets.items():
  a=d['x'];assert a.ndim==3 and a.shape[1:]==(4,712) and np.isfinite(a).all() and (a>=0).all();pos=a[a>0];contract.append(dict(dataset=name,shape=str(list(a.shape)),dtype=str(a.dtype),finite=True,nonnegative=True,zero_fraction=float((a==0).mean()),minimum=float(a.min()),maximum=float(a.max()),positive_P01=float(np.quantile(pos,.01)),positive_P50=float(np.quantile(pos,.5)),positive_P99=float(np.quantile(pos,.99)),windows=len(a),groups=len(set(d['g'])),group_unit='Session' if name.startswith('Real') else 'Root',exact_unique_numeric_windows=len({w.tobytes() for w in a})))
 table('input_contract.csv',contract);table('time_contract.csv',timeaudit);mapping=list(csv.DictReader((P41/'real_sim_patch_mapping.csv').open()));table('physical_patch_mapping_evidence.csv',mapping);save('Stage0_gate.json',dict(numeric_contract_passed=True,shape_channel_block_sizes_match=True,channel_permutation_unchanged=True,physical_same_Taxel_position='UNKNOWN',finger_middle_root_conflict='conditional320Taxels if historical user anatomical table is correct; not confirmed driver mapping',row_col_orientation='UNKNOWN',physical_sign_and_units='UNKNOWN: both stored nonnegative response, not verified same physical quantity',Real_processing='Calibration/driver missing; gate flag mixed and execution unverified',Sim_processing='native touch force -> project deadzone/saturation/power/lowpass/crosstalk; not certified same real transfer function',actual_incompatibility_proven=False,proceed_scope='index-space zero-shot diagnostic permitted by StageD physical-UNKNOWN branch; no physical correspondence claim',not_uniform_shared_temporal_sampling=True))
 ck=torch.load(P/'checkpoint.pt',map_location='cpu',weights_only=False);M=module('c26_existing_C13_architecture',C3/'code/models.py');R=M.DomainVAE(ck['encoder_scale'],ck['positive_scale']);R.load_state_dict(ck['model']);R.eval();R.requires_grad_(False);paramhash=sha(P/'checkpoint.pt');dim=[];summary=[];groupstats=[];geometry={};covariances={}
 def stats(z):
  cc=np.cov(z,rowvar=False);ev=np.maximum(np.linalg.eigvalsh(cc)[::-1],0);pp=ev/max(ev.sum(),1e-300);nonzero=pp>0;norm=np.linalg.norm(z,axis=1);return dict(windows=len(z),effective_rank=float(np.exp(-(pp[nonzero]*np.log(pp[nonzero])).sum())),PCs_90pct=int(np.searchsorted(np.cumsum(pp),.9)+1),top10_explained_variance=float(pp[:10].sum()),total_variance=float(ev.sum()),norm_P50=float(np.median(norm)),norm_P95=float(np.quantile(norm,.95)),norm_max=float(norm.max()),mean_coordinate_std=float(z.std(0).mean())),cc,ev
 for name,d in sets.items():
  mu=[];lv=[]
  with torch.inference_mode():
   for i in range(0,len(d['x']),64):
    a,b=R.encoder(torch.from_numpy(d['x'][i:i+64]));mu.append(a.numpy());lv.append(b.numpy())
  mu=np.concatenate(mu);lv=np.concatenate(lv);assert np.isfinite(mu).all() and np.isfinite(lv).all();d['mu']=mu;d['lv']=lv;np.savez_compressed(O/f'{name}_latents.npz',mu=mu,logvar=lv,latent=mu,groups=d['g'],stage=d['s'],units=d['u'],tasks=d['tasks']);flat=mu.reshape(len(mu),-1).astype('float64');d['flat']=flat;ss,cv,ev=stats(flat);summary.append(dict(dataset=name,**ss,posterior_variance_P50=float(np.median(np.exp(lv))),posterior_variance_P95=float(np.quantile(np.exp(lv),.95)),same_Encoder=True));covariances[name]=cv;np.savez_compressed(O/f'{name}_latent_statistics.npz',mean=flat.mean(0),std=flat.std(0),covariance=cv,eigenvalues=ev)
  for j in range(256):dim.append(dict(dataset=name,frame=j//64,dimension=j%64,mean=float(flat[:,j].mean()),std=float(flat[:,j].std()),posterior_variance_mean=float(np.exp(lv[:,j//64,j%64]).mean())))
  for key in ['g','s','tasks']:
   for v in sorted(set(d[key])):
    zz=flat[d[key]==v]
    if len(zz)>1:ss,_,_=stats(zz);groupstats.append(dict(dataset=name,stratum=key,value=str(v),**ss))
  if name.startswith('Real'):
   cache=np.load(C25/f'{name}_native_latents.npz');assert np.allclose(mu,cache['mu'],rtol=1e-5,atol=1e-5)
  print('SHARED_ENCODED',name,len(mu),flush=True)
 table('latent_summary.csv',summary);table('latent_dimensions.csv',dim);table('latent_group_statistics.csv',groupstats)
 ref=sets['Real_train']['flat'];mean=ref.mean(0);sd=np.maximum(ref.std(0),1e-6);ev,Q=np.linalg.eigh(covariances['Real_train']);floor=max(float(ev.max())*1e-4,1e-8);W=Q@np.diag(1/np.sqrt(np.maximum(ev,floor)))@Q.T;basis=Q[:,np.argsort(ev)[::-1]];np.savez(O/'Real_Train_only_geometry.npz',mean=mean,std=sd,whitening=W,PCAbasis=basis,eigenfloor=floor);pca={};support=[];distrows=[];envelopes=[];nnout={}
 for name,d in sets.items():pca[name]=(d['flat']-mean)@basis[:,:10]
 np.savez_compressed(O/'shared_coordinate_PCA.npz',**pca)
 for metric,transform in [('raw_Euclidean',lambda x:x),('RealTrain_standardized',lambda x:(x-mean)/sd),('RealTrain_whitened',lambda x:(x-mean)@W),('cosine',lambda x:x)]:
  A=transform(ref);mode='cosine' if metric=='cosine' else 'euclidean';rr=cdist(A,A,mode);np.fill_diagonal(rr,np.inf);selfnn=rr.min(1);rr[sets['Real_train']['g'][:,None]==sets['Real_train']['g'][None,:]]=np.inf;cross=rr.min(1);cut=float(np.quantile(cross,.95));envelopes.append(dict(metric=metric,source='RealTrain crossSession nearest-neighbor P95 only',P95=cut))
  for label,v in [('RealTrain_exclude_self',selfnn),('RealTrain_crossSession',cross)]:distrows.append(dict(metric=metric,dataset=label,P05=float(np.quantile(v,.05)),P50=float(np.median(v)),P95=float(np.quantile(v,.95))))
  for name in ['Real_validation','Natural_train','Natural_validation']:
   query=transform(sets[name]['flat']);dd=cdist(query,A,mode);nearest=dd.min(1);arg=dd.argmin(1);nnout[name+'_'+metric]=nearest;inside=nearest<=cut;distrows.append(dict(metric=metric,dataset=name+'_to_RealTrain',P05=float(np.quantile(nearest,.05)),P50=float(np.median(nearest)),P95=float(np.quantile(nearest,.95))));support.append(dict(metric=metric,dataset=name,windows=len(nearest),within_RealTrain_crossSession_P95=float(inside.mean()),equal_group_fraction=float(np.mean([inside[sets[name]['g']==g].mean() for g in set(sets[name]['g'])])),unique_target_nearest_Real_Sessions=len(set(sets['Real_train']['g'][arg])),interpretation='numeric Train reference envelope, not semantic coverage or calibrated OOD probability'))
   for key in ['g','s','tasks']:
    for v in sorted(set(sets[name][key])):
     ix=sets[name][key]==v;support.append(dict(metric=metric,dataset=name,stratum=key,value=str(v),windows=int(ix.sum()),within_RealTrain_crossSession_P95=float(inside[ix].mean()),interpretation='same fixedTrain envelope'))
  if metric=='raw_Euclidean':
   n=sets['Natural_train'];nn=cdist(n['flat'],n['flat']);np.fill_diagonal(nn,np.inf);v=nn.min(1);distrows.append(dict(metric=metric,dataset='NaturalTrain_exclude_self',P05=float(np.quantile(v,.05)),P50=float(np.median(v)),P95=float(np.quantile(v,.95))));nn[n['g'][:,None]==n['g'][None,:]]=np.inf;v=nn.min(1);distrows.append(dict(metric=metric,dataset='NaturalTrain_crossRoot',P05=float(np.quantile(v,.05)),P50=float(np.median(v)),P95=float(np.quantile(v,.95))))
  print('SUPPORT',metric,flush=True)
 np.savez_compressed(O/'nearest_neighbor_distances.npz',**nnout);table('support_reference_envelopes.csv',envelopes);table('support_by_group_stage_task.csv',support);table('latent_distances.csv',distrows)
 # Marginal latent OOD reference only fixed by Real Train; not filter samples.
 lo=np.quantile(ref,.005,axis=0);hi=np.quantile(ref,.995,axis=0);ood=[]
 for name,d in sets.items():
  outside=(d['flat']<lo)|(d['flat']>hi);ood.append(dict(dataset=name,fraction_coordinate_values_outside_RealTrain_P005_P995=float(outside.mean()),fraction_windows_with_any_coordinate_outside=float(outside.any(1).mean()),mean_fraction_outside_per_window=float(outside.mean(1).mean()),high_dimensional_warning='any-coordinate exceedance not calibrated OOD, no sample filtering'))
 table('latent_marginal_outliers.csv',ood)
 # Original unchanged decoder interface for both roles; no teacher forcing.
 dev='cuda:0' if torch.cuda.is_available() else 'cpu';D=R.decoder.to(dev);D.eval();generated={};readout=[]
 for name in ['Real_validation','Natural_validation']:
  arr=[]
  with torch.inference_mode():
   for i in range(0,len(sets[name]['mu']),64):arr.append(D.deterministic_decode(torch.from_numpy(sets[name]['mu'][i:i+64]).to(dev)).cpu().numpy())
  y=np.concatenate(arr);generated[name]=y;d=sets[name];assert np.isfinite(y).all() and (y>=0).all();np.savez_compressed(O/f'{name}_outputs.npz',values=y,groups=d['g'],stage=d['s'],units=d['u'],tasks=d['tasks']);print('DECODED',name,len(y),flush=True)
 original=np.load(P/'validation_outputs.npz')['values'];realgen=generated['Real_validation'];maxerr=float(np.abs(realgen-original).max());assert np.allclose(realgen,original,rtol=1e-5,atol=1e-5);assert np.array_equal(realgen>0,original>0);save('C13_Real_reconstruction_replay.json',dict(max_absolute_difference=maxerr,allclose=True,rtol=1e-5,atol=1e-5,Mask_bit_identical=True,inference=dev))
 def outputstats(a):
  p=a[a>0];z=a.astype('float64');mask=a>0;prev=mask[:,:-1];nxt=mask[:,1:];tot=z.sum(-1);sq=(z*z).sum(-1);conc=np.divide(tot*tot,712*sq,out=np.zeros_like(tot),where=sq>0);return dict(N=len(a),finite=bool(np.isfinite(a).all()),nonnegative=bool((a>=0).all()),zero_fraction=float((a==0).mean()),**{f'positive_P{q}':float(np.quantile(p,q/100)) if len(p) else None for q in [1,10,50,90,95,99]},maximum=float(a.max()),total_response_mean=float(tot.mean()),concentration=float(conc.mean()),absolute_frame_delta=float(np.abs(np.diff(z,axis=1)).mean()),log_frame_delta=float(np.abs(np.diff(np.log1p(z),axis=1)).mean()),lag1_correlation=float(np.corrcoef(z[:,:-1].ravel(),z[:,1:].ravel())[0,1]),transition01=float((~prev&nxt).sum()/max((~prev).sum(),1)),transition10=float((prev&~nxt).sum()/max(prev.sum(),1)),unique_numeric_windows=len({w.tobytes() for w in a}),mean_window_std=float(z.std(0).mean()),all_zero_windows=int((a==0).all((1,2)).sum()))
 statrows=[];outgroup=[]
 for name,a in [('Real_original',sets['Real_validation']['x']),('Real_reconstruction',generated['Real_validation']),('Natural_original',sets['Natural_validation']['x']),('Natural_zero_shot',generated['Natural_validation'])]:
  statrows.append(dict(source=name,**outputstats(a)));g=sets['Real_validation' if name.startswith('Real') else 'Natural_validation']['g'];task=sets['Real_validation' if name.startswith('Real') else 'Natural_validation']['tasks']
  for key,vals in [('group',g),('task',task)]:
   for v in set(vals):outgroup.append(dict(source=name,stratum=key,value=str(v),**outputstats(a[vals==v])))
 table('output_statistics.csv',statrows);table('output_group_task_statistics.csv',outgroup)
 x=sets['Natural_validation']['x'].astype('float64');y=generated['Natural_validation'].astype('float64');response=[]
 def numeric_features(a):
  mass=a.sum(-1);sq=(a*a).sum(-1);return dict(total_response=mass.mean(1),response_max=a.max((1,2)),index_positive_fraction=(a>0).mean((1,2)),concentration=np.divide(mass*mass,712*sq,out=np.zeros_like(mass),where=sq>0).mean(1),absolute_frame_delta=np.abs(np.diff(a,axis=1)).mean((1,2)))
 fin=numeric_features(x);fout=numeric_features(y)
 for key,a in fin.items():
  b=fout[key];rr=spearmanr(a,b);response.append(dict(feature=key,Spearman=float(rr.statistic) if np.isfinite(rr.statistic) else None,Pearson=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None,interpretation='association across unchanged windows, NOT intervention or physical correspondence'))
 table('input_output_response.csv',response);np.savez_compressed(O/'window_response_features.npz',**{'input_'+k:v for k,v in fin.items()},**{'output_'+k:v for k,v in fout.items()},groups=sets['Natural_validation']['g'],tasks=sets['Natural_validation']['tasks'])
 # Exact index overlap descriptive only; no assertion that index overlap equals physical position retention.
 xm=x>0;ym=y>0;inter=(xm&ym).sum((1,2));union=(xm|ym).sum((1,2));iou=np.divide(inter,union,out=np.ones(len(x)),where=union>0);save('index_structure_only.json',dict(mean_index_Mask_IoU=float(iou.mean()),not_physical_location_preservation=True,spatial_coordinate_mapping='UNKNOWN',no_input_modified=True))
 nr=next(r for r in statrows if r['source']=='Natural_zero_shot');nonzero=(x>0).any((1,2));lost=(y==0).all((1,2));frac=float(lost[nonzero].mean());stop=nr['unique_numeric_windows']<2 or nr['mean_window_std']<1e-8 or frac>=.95;save('StageD_screen.json',dict(hard_stop=stop,zero_output_fraction_among_nonzero_inputs=frac,unique_outputs=nr['unique_numeric_windows'],mean_window_std=nr['mean_window_std'],numeric_nonconstant_screen_only=True,physical_semantics_retention='UNKNOWN',C0_allowed=not stop,all_samples_retained=True))
 # P40 contract audit, no inference or neural fitting.
 pck=torch.load(P40/'selected_model/checkpoint.pt',map_location='cpu',weights_only=False);save('P40_shared_encoder_audit.json',dict(checkpoint=str(P40/'selected_model/checkpoint.pt'),sha256=sha(P40/'selected_model/checkpoint.pt'),checkpoint_keys=list(pck),shared_same_function=True,latent_shape='4x64 feature Z, not VAE posterior',posterior_mean_logvar=False,train_Real=5153,train_Sim=1824,Sim_source='P37 transformed Sim, not current Natural5task',preprocessing='Mask branch and log1p(x)/fixed common amplitude_scale branch',Decoder_present=False,C13_Decoder_coordinate_compatible=False,may_be_separate_representation_candidate=True,no_mixing_with_C26=True,source=str(P40/'code/model.py'),evidence=str(P40/'p40_report.md')))
 assert all(sha(p)==h for p,h in protected.items());assert sha(P/'checkpoint.pt')==paramhash;save('inference_freeze.json',dict(time=time.time(),checkpoint_sha256=paramhash,Real_outputs_sha256=sha(O/'Real_validation_outputs.npz'),Natural_outputs_sha256=sha(O/'Natural_validation_outputs.npz'),all_protected_unchanged=True,no_model_or_scaler_fitting=True,C0_evaluation_not_yet_started=True,seconds=time.time()-start));print('C26 INFERENCE COMPLETE C0_ALLOWED',not stop,flush=True)
if __name__=='__main__':main()
