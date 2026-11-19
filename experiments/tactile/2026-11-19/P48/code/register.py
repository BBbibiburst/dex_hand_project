from common import *
import subprocess,platform
P47=Path('outputs/tactile_p47_conditional_transfer_evaluation_2026-11-18');P46=Path('outputs/tactile_p46_real_variability_envelope_audit_2026-11-18')
assert not (O/'registration.json').exists()
config=dict(date='2026-11-19',seed=4801,models=['Original']+[f'{a}_K{k}' for a in ['A','B','C'] for k in [1,2,4,8]],K_interpretation='same shared field Euler on t[0,1],dt1/K,h0zero;K not adversarial epochs',preprocessing=str(P46/'RealTrain_preprocessing.npz'),weights='equal domains .5 each;equal Session/Root totalweight,uniform windows within group;no stage reweight;window sensitivity separate',bootstrap=1000,bootstrap_seed=4802,bootstrap_sim='taskstratified2Roots perValtask;paired counts allmethods',JSD=dict(instances=['logistic','mlp_11','mlp_12','mlp_13','gru_11','gru_12','gru_13'],deep_epochs=40,batch_size=64,optimizer='AdamW',lr=.001,weight_decay=.001,gradient_clip=5.,checkpoint='fixedepoch40 endpoint;NOValidation earlystopping/selection',scaler='frozenP46 same allcandidates',Logistic='float64 L2/C1equivalent LBFGS max1000,maxeval1500,grad<=1e-6 required, no extension',train_convergence='last5eval weightedCE range<=.02; diagnostic only',mismatch='ValCE-TrainCE>.2 flags invalid branch; no tuning or continuing',baseline_min=.01,denominator='J0>.01 andgroupbootstrap lower95>0',negative_proxy='keep signed ln2-CE;negative proxy NOT proof of negative trueJSD and not valid90claim',family90='all7instances valid andpointR>=.90;CI reported independently;no model/seed selection'),SWD=dict(directions=1024,seed=4803,metric='mean exact weighted1D Wasserstein1,notfullhighdimW1',bootstrap=1000,denominator='positive lowergroupbootstrap CI;no projection search'),equivalence=dict(realTrain32=True,groups_A=4,groups_B=10,windows_pergroup=7,windowbank='fixed7 sampled once perRealTrainSession andValSession/Root,seed4804',RRdraws=1000,RRdisjoint=True,epsilon='q95sqrt multiscaleMMD V;RealTrain only',calibration_bootstrap='1000 empirical32Session cluster resamples;in each resample1000weightedwithoutreplacement disjoint4+10uniqueSession partitions;fixed7windowbank;partitions notindependent units',validation_bootstrap='4Session resampling,taskstrat10Root resampling,same7banks;epsilon reestimated in same replicate',pass_rule='95%two-sided percentileCI upper ofMMD-epsilon<0;development marginal only',stability='report epsilon clusterCI plus independentgroup count;stop iftoo few uniqueSessions14 or nonpositive/undefined calibration;not certify universalphysicalmargin'),path=dict(source='Original separate fromdecoded h0=Dr(zeros),never call Drzeros Original',save_all_steps=True,K=[1,2,4,8],metrics='full2848 log1p fixedP46 metric plusrawphysicalresponse update stats; no crossindependentlatents'),quality='reuseP47 verifiedoutputs PRDC/HSIC;noP47acceptance thresholds existed,do notinvent',no_Test_Holdout=True,no_C0=True,no_generator_optimizer=True,no_P49=True)
save(O/'registration.json',config)
prot={}
for folder in [P46,P47]:
 m=json.loads((folder/'integrity.json').read_text());prot.update(m['inputs'])
 for p,h in m['outputs'].items():assert sha(folder/p)==h;prot[str(folder/p)]=h
 prot[str(folder/'integrity.json')]=sha(folder/'integrity.json')
for folder in [P44,P45]:
 for p in folder.rglob('*.py'):prot[str(p)]=sha(p)
 for p in folder.rglob('*.pt'):prot[str(p)]=sha(p)
for p in [Path('apps/produce_five_task_dataset.py'),Path('apps/five_task_production.py'),Path('configs/five_task_speed_batch2.json')]:
 if p.exists():prot[str(p)]=sha(p)
assert all(sha(p)==h for p,h in prot.items());save(O/'protected_hashes_before.json',prot)
rr=real('train');rv=real('validation');nt=natural('train');nv,outputs=inputs();assert (len(rr['x']),len(rv['x']),len(nt['x']),len(nv['x']))==(5153,780,4049,905)
old=list(csv.DictReader((P46/'group_inventory.csv').open()));inventory=[]
for role,domain,d in [('train','Real',rr),('validation','Real',rv),('train','Natural',nt),('validation','Natural',nv)]:
 for g in np.unique(d['g']):
  ix=d['g']==g;h=hashlib.sha256(d['x'][ix].tobytes()).hexdigest();assert h==next(r['numeric_sha256'] for r in old if r['domain']==domain and r['role']==role and r['group']==g);inventory.append(dict(domain=domain,role=role,group=g,windows=int(ix.sum()),shape=str(d['x'][ix].shape),sha256_numeric=h,source='P46registeredexactroles'))
table(O/'input_inventory.csv',inventory)
# Shared bootstrap identity, never windowsasindependent observations.
rng=np.random.default_rng(4802);rg=np.unique(rv['g']);sg=np.unique(nv['g']);rtasks=np.array([nv['task'][nv['g']==g][0] for g in sg]);rb=rng.integers(0,4,size=(1000,4));sb=np.concatenate([rng.choice(np.flatnonzero(rtasks==t),(1000,2),replace=True) for t in sorted(set(rtasks))],axis=1);np.savez(O/'group_bootstrap_indices.npz',Real_groups=rg,Sim_groups=sg,Real=rb,Sim=sb)
rng=np.random.default_rng(4803);dirs=rng.standard_normal((2848,1024));dirs/=np.linalg.norm(dirs,axis=0);np.save(O/'projection_directions.npy',dirs)
save(O/'data_counts.json',dict(RealTrain=[5153,32],RealVal=[780,4],NaturalTrain=[4049,50],NaturalVal=[905,10]));save(O/'register_complete.json',dict(protected=len(prot),all_hashes_verified=True,migration_epochs=0,migration_optimizer_steps=0));print('P48 REGISTERED',len(prot),'protectedfiles',flush=True)
