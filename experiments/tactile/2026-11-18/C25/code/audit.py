"""C25 bounded audit: no cross-independent-latent metric, neural fitting, or C0 inference."""
import ast,csv,json,time,hashlib,importlib.util,zipfile,subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
O=Path(__file__).resolve().parents[1];C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');C4=Path('outputs/tactile_p43_c4_root_cause_audit_2026-11-16');C13=Path('outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18');BASE=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16');P40=Path('outputs/tactile_p40_shared_features_2026-11-15');OLD=Path('outputs/tactile_gmm_dual_vae_2026-11-16')
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
def statehash(state,prefix):
 h=hashlib.sha256()
 for k in sorted(state):
  if k.startswith(prefix):h.update(k.encode());h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def getmodule(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 assert not (O/'registration.json').exists();start=time.time();torch.set_num_threads(2);save('registration.json',dict(date='2026-11-18',stage_A_gate='native independent64D encoders require verified common cost coordinates; otherwise cross-domain distances,GMM,OT and sampling skipped',roles=['train','validation'],bootstrap_draws=300,seed=2501,max_seconds=600,max_new_bytes=500000000,no_neural_training=True,no_C0_evaluation=True,no_712D_generation=True,no_Test_Holdout=True,K_if_and_only_if_gate_passes=[2,4,8,16],condition_thresholds='exact0; relative per-frame peak .001,.01,.05 (descriptive only)',no_auto_C26=True,git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
 protected=dict(json.loads((C3/'c0/frozen.json').read_text())['files']);inputs=[C3/'real_vae/checkpoint.pt',C3/'sim_vae/checkpoint.pt',C3/'translator/checkpoint.pt',C3/'code/models.py',C3/'model_training_registration.json',C13/'checkpoint.pt',C13/'protocol_clarification.json',Path('outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/checkpoint.pt'),P40/'selected_model/checkpoint.pt',P40/'code/model.py',P40/'p40_report.md',OLD/'code/dual_vae.py',OLD/'code/gmm.py',OLD/'vae_config.json']
 for name in ['C_dual_vae','D_full']:inputs.extend([OLD/name/'checkpoint.pt',OLD/'amplitude_repair'/name/'checkpoint.pt'])
 for role in ['train','validation']:inputs.extend([BASE/f'real_{role}/index.json',BASE/f'natural_sim_{role}/windows.npz',BASE/f'natural_sim_{role}/windows_manifest.csv',C4/f'latents/Real_{role}.npz',C4/f'latents/Natural_{role}.npz'])
 for p in inputs:protected[str(p)]=sha(p)
 assert all(sha(p)==h for p,h in protected.items());save('protected_hashes_before.json',protected)
 ckR=torch.load(C13/'checkpoint.pt',map_location='cpu',weights_only=False);origR=torch.load(C3/'real_vae/checkpoint.pt',map_location='cpu',weights_only=False);ckS=torch.load(C3/'sim_vae/checkpoint.pt',map_location='cpu',weights_only=False);assert all(torch.equal(v,origR['model'][k]) for k,v in ckR['model'].items() if k.startswith('encoder.'));assert any(not torch.equal(v,ckS['model'][k]) for k,v in ckR['model'].items() if k.startswith('encoder.'))
 model=getmodule('c25_native_vae',C3/'code/models.py');nets={}
 for domain,ck in [('Real',ckR),('Natural',ckS)]:
  net=model.DomainVAE(ck['encoder_scale'],ck['positive_scale']);net.load_state_dict(ck['model']);net.eval();net.requires_grad_(False);nets[domain]=net
 inventory=[]
 for name,ck,path,domain in [('C13_Real',ckR,C13/'checkpoint.pt','Real'),('C3_Real_original',origR,C3/'real_vae/checkpoint.pt','Real'),('C3_Sim',ckS,C3/'sim_vae/checkpoint.pt','Natural+GMM_Paired_Sim')]:inventory.append(dict(model=name,source=str(path),checkpoint_sha256=sha(path),encoder_state_sha256=statehash(ck['model'],'encoder.'),input_shape='4x712',latent_shape='4x64',posterior_mean_logvar=True,training_domain=domain,weight_shared_with_other_domain=False,cross_domain_joint_training=False,common_coordinates_verified=False,preprocess='concat exact-positive Mask, log1p(x)/fixed encoder.scale; domain-specific scale',encoder_scale=float(ck['encoder_scale']),positive_scale=float(ck['positive_scale']),C13_Decoder_compatible=name!='C3_Sim',evidence=str(C3/'code/models.py')+':7-12; model_training_registration.json'))
 for name in ['C_dual_vae','D_full']:
  cp=OLD/name/'checkpoint.pt';ck=torch.load(cp,map_location='cpu',weights_only=False);sd=ck.get('model',ck.get('state_dict',{}));prefixes=sorted(set(k.split('.')[0] for k in sd));save(name+'_checkpoint_contract.json',dict(keys=list(ck),state_prefixes=prefixes))
  for enc,dom in [('ER','Real'),('ES','Sim processed'+(' after index-GMM' if name=='D_full' else ''))]:inventory.append(dict(model='legacy_'+name+'_'+enc,source=str(cp),checkpoint_sha256=sha(cp),encoder_state_sha256=statehash(sd,enc+'.'),input_shape='4x712',latent_shape='4x32',posterior_mean_logvar=True,training_domain=dom,weight_shared_with_other_domain=False,cross_domain_joint_training=True,common_coordinates_verified='nominal shared heads/alignment objective, semantic correspondence NOT certified',preprocess='domain-specific scale: Mask + log1p/scale',C13_Decoder_compatible=False,evidence=str(OLD/'code/dual_vae.py')+':18-48; vae_config.json'))
 cp=P40/'selected_model/checkpoint.pt';ck=torch.load(cp,map_location='cpu',weights_only=False);save('P40_checkpoint_contract.json',dict(keys=list(ck)));inventory.append(dict(model='P40_shared_feature_encoder',source=str(cp),checkpoint_sha256=sha(cp),encoder_state_sha256='see checkpoint hash; no inference in C25',input_shape='4x712',latent_shape='4x64 feature Z, NOT C13 VAE posterior',posterior_mean_logvar=False,training_domain='Real P37 + transformed P37 Sim; not current5task Natural',weight_shared_with_other_domain=True,cross_domain_joint_training=True,common_coordinates_verified='same function structurally; Natural semantic metric/Decoder compatibility not verified',preprocess='same Mask + log1p/amplitude_scale',C13_Decoder_compatible=False,evidence=str(P40/'code/model.py')+':17-24; p40_report.md'))
 inventory.extend([dict(model='P41/index_GMM',source=str(OLD/'code/gmm.py'),checkpoint_sha256='NOT_APPLICABLE',input_shape='4x712 patches',latent_shape='2D fixed patch-index response centers, NOT neural latent',posterior_mean_logvar=False,training_domain='Real spatial proxy prior',weight_shared_with_other_domain='NOT_APPLICABLE',cross_domain_joint_training=False,common_coordinates_verified='physical row/col cross-domain map UNVERIFIED',C13_Decoder_compatible=False,evidence='GMMRenderer/grid; original P41 stopped preflight, later index-space GMM separate'),dict(model='C3_GRU/C10/C10R1_latent_mappers',source=str(C3/'code/models.py'),checkpoint_sha256=sha(C3/'translator/checkpoint.pt'),input_shape='Sim4x64',latent_shape='predicted Real4x64',posterior_mean_logvar=False,training_domain='statistical pseudo-paired Sim/Real latents',weight_shared_with_other_domain=False,cross_domain_joint_training=False,common_coordinates_verified='learned correspondence candidate, NOT encoder-coordinate certification',C13_Decoder_compatible='output interface only; correctness not established',evidence='Translator at models.py:34; C4 audit'),dict(model='C17-C22_Mask_or_condition_networks',source='outputs/tactile_c13_c22_research_decision_2026-11-18/REPORT.md',checkpoint_sha256='reused C3/C13 Encoder hashes above',input_shape='proxy conditions and/or original mapped latent',latent_shape='tokens/conditional Mask state, not shared VAE coordinates',posterior_mean_logvar=False,training_domain='see historical reports',weight_shared_with_other_domain=False,cross_domain_joint_training='not a new shared Real-Sim VAE',common_coordinates_verified=False,C13_Decoder_compatible='amplitude integration only',evidence='C20/code/integrate.py explicitly reuses C13 + originalGRU'),dict(model='hierarchical_stochastic_Mask',source='outputs/tactile_p43_autoregressive_mask_2026-11-16/hierarchical_v2/code/model.py',checkpoint_sha256='NOT_LOADED_NOT_PRIMARY',input_shape='mapped4x64 plus recurrentMask context',latent_shape='global8+16patch8 conditional random variables perframe',posterior_mean_logvar=True,training_domain='Real Mask posterior/conditional prior',weight_shared_with_other_domain='NOT_APPLICABLE',cross_domain_joint_training=False,common_coordinates_verified=False,C13_Decoder_compatible=False,evidence='Mask-specific latent; not the C3 independent4x64 posterior')])
 table('encoder_inventory.csv',inventory);save('coordinate_gate.json',dict(native_R_S_shared_weights=False,Real_C13_Encoder_equals_C3_Real=True,encoder_R_hash=statehash(ckR['model'],'encoder.'),encoder_S_hash=statehash(ckS['model'],'encoder.'),native_shared_coordinate_verified=False,cross_native_distances_forbidden=True,GMM_OT_blocked=True,P40_shared_encoder_exists=True,P40_not_C13_coordinate=True,legacy_joint_VAE_exists=True,legacy_latent4x32_not_C13_4x64=True,why='same dimension/standard Gaussian KL does not identify orientation/semantic axes across independently fitted models; Translator is a candidate mapping not common-coordinate proof'))
 env=dict(np=np,zipfile=zipfile);tree=ast.parse(Path('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='npz_rows'],type_ignores=[]),'getrows','exec'),env);get=env['npz_rows'];sets={};source_rows=[]
 for role in ['train','validation']:
  idx=json.loads((BASE/f'real_{role}/index.json').read_text());x=get(idx['source'],'real',idx['indices']);g=get(idx['source'],'real_group',idx['indices']);s=get(idx['source'],'real_stage',idx['indices']);u=get(idx['source'],'real_unit',idx['indices']);sets['Real_'+role]=dict(x=x,g=g,s=s,u=u,tasks=np.full(len(x),'UNKNOWN_REAL_TASK_NOT_FIVE_TASK_LABEL'),domain='Real',role=role)
  z=np.load(BASE/f'natural_sim_{role}/windows.npz');d=dict(x=z['values'],g=z['groups'],s=z['stage'],u=z['units'],domain='Natural',role=role);meta=list(csv.DictReader((BASE/f'natural_sim_{role}/windows_manifest.csv').open()));byhash={r['tactile_sha256']:r for r in meta if r['primary_unique_numeric_window']=='True'};tt=[]
  for w in d['x']:tt.append(byhash[hashlib.sha256(w.tobytes()).hexdigest()]['task'])
  d['tasks']=np.array(tt);sets['Natural_'+role]=d
 for name,d in sets.items():
  assert d['x'].shape[1:]==(4,712);cache=np.load(C4/f'latents/{name}.npz');mu=cache['mu'];lv=cache['logvar'];assert np.array_equal(d['g'],cache['g']) and np.array_equal(d['s'],cache['s']);assert np.array_equal(d['u'],cache['u'])
  with torch.inference_mode():a,b=nets[d['domain']].encoder(torch.from_numpy(d['x'][:64]));assert np.allclose(a.numpy(),mu[:64],rtol=1e-5,atol=1e-5);assert np.allclose(b.numpy(),lv[:64],rtol=1e-5,atol=1e-5)
  d['mu']=mu.astype('float64');d['lv']=lv;np.savez_compressed(O/f'{name}_native_latents.npz',mu=mu,logvar=lv,groups=d['g'],stage=d['s'],units=d['u'],tasks=d['tasks']);source_rows.append(dict(dataset=name,windows=len(d['x']),groups=len(set(d['g'])),trials_or_episode_units=len(set('/'.join(str(u).split('/')[:2]) for u in d['u'])) if d['domain']=='Real' else len(set(str(u).split('/')[0] for u in d['u'])),group_type='Session' if d['domain']=='Real' else 'Root',unique_numeric_windows=len({w.tobytes() for w in d['x']}),latent_shape=str(list(mu.shape)),cached_encoder_checked=True,common_coordinate=False))
 table('data_inventory.csv',source_rows)
 # All distribution geometry below is WITHIN one encoder's coordinates only.
 latentrows=[];dimrows=[];eigrows=[];stratarows=[];distrows=[];stability=[];views={}
 def spectrum(z):
  cov=np.cov(z,rowvar=False);v=np.linalg.eigvalsh(cov)[::-1];v=np.maximum(v,0);p=v/max(v.sum(),1e-300);nz=p>0;rank=float(np.exp(-(p[nz]*np.log(p[nz])).sum()));return cov,v,np.cumsum(p),rank
 def desc(z):
  _,v,c,r=spectrum(z);norm=np.linalg.norm(z,axis=1);return dict(windows=len(z),effective_rank=r,top10_variance=float(c[min(9,len(c)-1)]),PCs_for90pct=int(np.searchsorted(c,.9)+1),norm_P10=float(np.quantile(norm,.1)),norm_P50=float(np.median(norm)),norm_P90=float(np.quantile(norm,.9)),norm_max=float(norm.max()),total_variance=float(v.sum()))
 for name,d in sets.items():
  z=d['mu'].reshape(len(d['mu']),-1);cov,eig,cs,rank=spectrum(z);latentrows.append(dict(dataset=name,**desc(z)));np.savez_compressed(O/f'{name}_covariance.npz',covariance=cov,mean=z.mean(0),std=z.std(0),eigenvalues=eig,cumulative_PCA=cs)
  for j in range(z.shape[1]):dimrows.append(dict(dataset=name,frame=j//64,dimension=j%64,mean=float(z[:,j].mean()),std=float(z[:,j].std()),posterior_logvar_mean=float(d['lv'][:,j//64,j%64].mean())))
  for j,(e,c) in enumerate(zip(eig,cs)):eigrows.append(dict(dataset=name,PC=j+1,eigenvalue=float(e),cumulative_variance=float(c)))
  for key in ['g','s','tasks']:
   for val in sorted(set(d[key])):
    zz=z[d[key]==val];stratarows.append(dict(dataset=name,stratum={'g':'Session_or_Root','s':'stage','tasks':'task'}[key],value=str(val),**(desc(zz) if len(zz)>1 else {'windows':len(zz)})))
 for dom in ['Real','Natural']:
  tr=sets[dom+'_train'];va=sets[dom+'_validation'];ref=tr['mu'].reshape(len(tr['mu']),-1);query=va['mu'].reshape(len(va['mu']),-1);mean=ref.mean(0);sd=np.maximum(ref.std(0),1e-6);cov=np.cov(ref,rowvar=False);ev,Q=np.linalg.eigh(cov);ridge=max(float(ev.max())*1e-4,1e-8);wh=Q@np.diag(1/np.sqrt(np.maximum(ev,ridge)))@Q.T;vv,Qp=np.linalg.eigh(cov);basis=Qp[:,np.argsort(vv)[::-1]][:,:10];views[dom]=dict(train=(ref-mean)@basis,validation=(query-mean)@basis)
  np.savez_compressed(O/f'{dom}_train_only_geometry.npz',mean=mean,std=sd,whitening=wh,PCA_basis=basis,whitening_eigenfloor=ridge)
  for metric,transform in [('raw_native_Euclidean',lambda x:x),('Train_only_standardized',lambda x:(x-mean)/sd),('Train_only_whitened',lambda x:(x-mean)@wh),('native_cosine',lambda x:x)]:
   A=transform(ref);V=transform(query);mode='cosine' if metric=='native_cosine' else 'euclidean';TT=cdist(A,A,mode);TV=cdist(V,A,mode);TT=np.nan_to_num(TT,nan=1.,posinf=np.inf);TV=np.nan_to_num(TV,nan=1.,posinf=np.inf);np.fill_diagonal(TT,np.inf);selfnn=TT.min(1);TT[tr['g'][:,None]==tr['g'][None,:]]=np.inf;cross=TT.min(1);vnn=TV.min(1);threshold=float(np.quantile(cross,.95));stability.append(dict(domain=dom,metric=metric,Train_cross_group_NN_P95=threshold,validation_fraction_within_Train_cross_group_P95=float((vnn<=threshold).mean()),interpretation='within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee'))
   for label,a in [('Train_exclude_self',selfnn),('Train_cross_group',cross),('Validation_to_Train',vnn)]:distrows.append(dict(domain=dom,metric=metric,comparison=label,N=len(a),P05=float(np.quantile(a,.05)),P50=float(np.median(a)),P95=float(np.quantile(a,.95))))
   print('WITHIN_GEOMETRY',dom,metric,flush=True)
 table('latent_summary.csv',latentrows);table('latent_dimensions.csv',dimrows);table('latent_eigenspectrum.csv',eigrows);table('latent_group_stage_task.csv',stratarows);table('within_domain_distances.csv',distrows);table('within_domain_support_stability.csv',stability)
 np.savez_compressed(O/'separate_native_PCA_views.npz',**{dom+'_'+role:a for dom,dd in views.items() for role,a in dd.items()})
 # Input-derived weak statistical descriptors, not certified common physical conditions.
 OFF=np.cumsum([0]+[32,32,56]*5+[112]);features={};conditional=[];groupstats=[];sensitivity=[];boost=[];rng=np.random.default_rng(2501)
 def feats(x):
  a=x.astype('float64');mass=a.sum(-1);sq=(a*a).sum(-1);patch=np.stack([a[:,:,l:r].sum(-1) for l,r in zip(OFF[:-1],OFF[1:])],-1);p=np.divide(patch,mass[...,None],out=np.zeros_like(patch),where=mass[...,None]>0);ent=-(p*np.log(np.maximum(p,1e-300))).sum(-1);part=np.divide(mass*mass,712*sq,out=np.zeros_like(mass),where=sq>0);active=(a>0);p01=(~active[:,:-1]&active[:,1:]).sum((1,2))/np.maximum((~active[:,:-1]).sum((1,2)),1);p10=(active[:,:-1]&~active[:,1:]).sum((1,2))/np.maximum(active[:,:-1].sum((1,2)),1)
  return dict(total_response=mass.mean(1),positive_log_mean=np.log1p(a).sum((1,2))/np.maximum(active.sum((1,2)),1),taxel_participation=part.mean(1),patch_response_entropy=ent.mean(1),effective_patch_number=np.divide(1,(p*p).sum(-1),out=np.zeros_like(mass),where=(p*p).sum(-1)>0).mean(1),log_frame_delta=np.abs(np.diff(np.log1p(a),axis=1)).mean((1,2)),scale_free_frame_delta=np.divide(np.abs(np.diff(a,axis=1)).sum(-1),mass[:,:-1]+mass[:,1:],out=np.zeros_like(mass[:,:-1]),where=(mass[:,:-1]+mass[:,1:])>0).mean(1),transition01=p01,transition10=p10,exact_active_patch=np.stack([active[:,:,l:r].any(-1) for l,r in zip(OFF[:-1],OFF[1:])],-1).sum(-1).mean(1))
 for name,d in sets.items():
  f=feats(d['x']);features[name]=f;np.savez_compressed(O/f'{name}_weak_descriptors.npz',**f,groups=d['g'],stage=d['s'],tasks=d['tasks'])
  for k,a in f.items():
   conditional.append(dict(dataset=name,feature=k,N=len(a),mean=float(a.mean()),P10=float(np.quantile(a,.1)),P50=float(np.median(a)),P90=float(np.quantile(a,.9))))
   for key in ['g','s','tasks']:
    for v in sorted(set(d[key])):
     ix=d[key]==v;groupstats.append(dict(dataset=name,stratum={'g':'Session_or_Root','s':'stage','tasks':'task'}[key],group=str(v),feature=k,N=int(ix.sum()),mean=float(a[ix].mean()),P50=float(np.median(a[ix]))))
   gm=np.array([a[d['g']==g].mean() for g in sorted(set(d['g']))]);b=gm[rng.integers(0,len(gm),size=(300,len(gm)))].mean(1);boost.append(dict(dataset=name,feature=k,unit='Session' if d['domain']=='Real' else 'Root',independent_groups=len(gm),equal_group_mean=float(gm.mean()),CI_low=float(np.quantile(b,.025)),CI_high=float(np.quantile(b,.975)),draws=300,not_final_inference=True))
  a=d['x'];peak=a.max(-1,keepdims=True)
  for th in [0,.001,.01,.05]:
   mask=a>peak*th;patchactive=np.stack([mask[:,:,l:r].any(-1) for l,r in zip(OFF[:-1],OFF[1:])],-1).sum(-1).mean(1)
   sensitivity.append(dict(dataset=name,threshold_type='relative_frame_peak',threshold=th,mean_active_patch=float(patchactive.mean()),equal_group_mean=float(np.mean([patchactive[d['g']==g].mean() for g in set(d['g'])])),no_output_threshold_change=True))
 table('condition_distributions.csv',conditional);table('condition_group_stage_task.csv',groupstats);table('condition_group_bootstrap.csv',boost);table('threshold_sensitivity.csv',sensitivity)
 # Group-aware descriptor support in SAME defined response features; never native-latent OT.
 weak=['taxel_participation','effective_patch_number','scale_free_frame_delta'];rr=np.stack([features['Real_train'][k] for k in weak],1);ss=np.stack([features['Natural_train'][k] for k in weak],1);lo=np.quantile(rr,.05,axis=0);hi=np.quantile(rr,.95,axis=0);overlap=[]
 for name,f in features.items():
  arr=np.stack([f[k] for k in weak],1);inside=(arr>=lo)&(arr<=hi);overlap.append(dict(dataset=name,features='|'.join(weak),fraction_in_all_Real_Train_marginal_P05_P95=float(inside.all(1).mean()),interpretation='descriptive axis-aligned weak-feature overlap; not common latent/physical correspondence proof'))
 table('weak_descriptor_overlap.csv',overlap)
 table('cross_domain_latent_distances.csv',[dict(comparison='Natural_native_mu vs Real_native_mu',Euclidean='NOT_COMPUTED',Wasserstein='NOT_COMPUTED',support_overlap='UNKNOWN',common_low_dimensional_semantics='UNKNOWN',reason='independent coordinate gauges; same4x64 shape not certified common metric')])
 save('stage_status.json',dict(A='FAILED_FOR_DEFAULT_NATIVE_LATENT_COST; legacy alternatives explicitly inventoried',B='WITHIN_DOMAIN_ONLY_COMPLETE; CROSS_DOMAIN_UNDEFINED',C='NOT_RUN_GATE_A_FAILED',D='WEAK_PROXY_STATISTICS_COMPLETE; NO_PHYSICAL_CONDITION_CERTIFIED',E='NOT_RUN_GATES_A_D_FAILED',F='NOT_RUN_NO_VALID_OT_PLAN',neural_training_epochs=0,optimizer_steps=0,new_GMM_fits=0,OT_solves=0,C0_evaluations=0,new_712D_outputs=0))
 assert all(sha(p)==h for p,h in protected.items());assert sum(p.stat().st_size for p in O.rglob('*') if p.is_file())<500000000;save('integrity.json',dict(all_protected_hashes_unchanged=True,files=protected,Real_Sim_original_unmodified=True,Test_Holdout_not_accessed=True,no_training=True,no_C0_inference=True,no_server_change=True,elapsed_seconds=time.time()-start));print('C25 AUDIT COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
