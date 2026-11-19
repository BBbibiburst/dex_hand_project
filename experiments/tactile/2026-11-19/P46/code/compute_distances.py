from common import *
from scipy.stats import beta
import time
assert (O/'prepare_complete.json').exists();assert not (O/'statistics_complete.json').exists();start=time.time();reg=json.loads((O/'registration.json').read_text());r=real('train');rv=real('validation');n,outs=inputs();prep=np.load(O/'RealTrain_preprocessing.npz');mu,sd,sigma=prep['mean'],prep['scale'],float(prep['sigma']);R=transform(r['x'],mu,sd);RV=transform(rv['x'],mu,sd);F={k:transform(v,mu,sd) for k,v in outs.items()};M=reg['unit_query_windows'];draws=reg['MonteCarlo_draws'];rng=np.random.default_rng(4601);rr=[];rootrows=[];drawrows=[];rvcheck=[];rvc=[];bootstrap=[];negative=[];paired=[];thresholds=[];query_banks={}
for g in sorted(set(r['g'])):
 ids=np.flatnonzero(r['g']==g);rg=np.random.default_rng(group_seed('Real:'+str(g)));query_banks['Real:'+str(g)]=np.stack([rg.choice(ids,M,replace=False) for _ in range(draws)])
for g in sorted(set(rv['g'])):
 ids=np.flatnonzero(rv['g']==g);rg=np.random.default_rng(group_seed('RealVal:'+str(g)));query_banks['RealVal:'+str(g)]=np.stack([rg.choice(ids,M,replace=False) for _ in range(draws)])
for g in sorted(set(n['g'])):
 ids=np.flatnonzero(n['g']==g);rg=np.random.default_rng(group_seed('Sim:'+str(g)));query_banks['Sim:'+str(g)]=np.stack([rg.choice(ids,M,replace=False) for _ in range(draws)])
# Query indices shared across all variants; MC draws average within one group, never new independent units.
np.savez_compressed(O/'query_index_banks.npz',**{k:v for k,v in query_banks.items()})
def aggregate(rows):
 result={k:float(np.mean([row[k] for row in rows])) for k in rows[0]}
 for scale in [.5,1.,2.]:result[f'MMD_rbf_{scale:g}']=float(np.sqrt(max(result[f'MMD2V_rbf_{scale:g}'],0)))
 result['MMD']=float(np.sqrt(max(result['MMD2V'],0)));result['Energy']=float(np.sqrt(max(result['Energy_V'],0)));return result
allroots=sorted(set(n['g']));root_tasks=np.array([n['task'][n['g']==g][0] for g in allroots]);assert len(allroots)==10
for cix,cohort in enumerate(reg['reference_sensitivity']):
 name=cohort['name'];refs=cohort['reference'];cal=cohort['calibration'];rg=np.random.default_rng(4601+100*cix);refbank=[]
 for b in range(draws):
  gs=rg.choice(refs,M,replace=False);refbank.append([int(rg.choice(np.flatnonzero(r['g']==g))) for g in gs])
 refbank=np.asarray(refbank);np.savez(O/f'{name}_reference_index_bank.npz',indices=refbank,reference_Sessions=np.array(refs));calvalues=[]
 for g in cal:
  vals=[distances(R[ii],R[jj],sigma) for ii,jj in zip(query_banks['Real:'+g],refbank)];a=aggregate(vals);calvalues.append(a);rr.append(dict(cohort=name,query_Session=g,reference_Sessions=json.dumps(refs),role='TrainCalibration',query_windows=M,reference_windows=M,MC_draws=draws,independent_query_Sessions=24,**a))
  for b,v in enumerate(vals):drawrows.append(dict(cohort=name,query=g,domain='RealTrain',draw=b,**v))
 for g in sorted(set(rv['g'])):
  a=aggregate([distances(RV[ii],R[jj],sigma) for ii,jj in zip(query_banks['RealVal:'+str(g)],refbank)]);rr.append(dict(cohort=name,query_Session=str(g),reference_Sessions=json.dumps(refs),role='RealValidation',query_windows=M,reference_windows=M,MC_draws=draws,independent_query_Sessions=4,**a))
 mat=np.array([[v[k] for k in METRICS] for v in calvalues]);qs={q:np.quantile(mat,q,axis=0) for q in [.5,.9,.95,.99]};bootrng=np.random.default_rng(4601+500+cix);cal_ids=bootrng.integers(0,24,size=(300,24));qboot=np.quantile(mat[cal_ids],.95,axis=1);root_ids=np.empty((300,10),int)
 for b in range(300):
  offset=0
  for task in sorted(set(root_tasks)):
   ix=np.flatnonzero(root_tasks==task);sample=bootrng.choice(ix,len(ix),replace=True);root_ids[b,offset:offset+len(ix)]=sample;offset+=len(ix)
 for j,metric in enumerate(METRICS):thresholds.append(dict(cohort=name,metric=metric,q50=float(qs[.5][j]),q90=float(qs[.9][j]),q95=float(qs[.95][j]),q99=float(qs[.99][j]),q95_bootstrap_CI_low=float(np.quantile(qboot[:,j],.025)),q95_bootstrap_CI_high=float(np.quantile(qboot[:,j],.975)),calibration_Sessions=24,upper_tail_expected_groups=24*.05,bootstrap_scope='conditional on fixed8reference and fixedTrain preprocessing; not unconditional total uncertainty'))
 all_model_values={}
 for model,x in F.items():
  values=[]
  for g in allroots:
   vals=[distances(x[ii],R[jj],sigma) for ii,jj in zip(query_banks['Sim:'+g],refbank)];a=aggregate(vals);values.append(a);rootrows.append(dict(cohort=name,model=model,Root=g,task=str(root_tasks[allroots.index(g)]),query_windows=M,reference_windows=M,MC_draws=draws,**a))
  mm=np.array([[v[k] for k in METRICS] for v in values]);all_model_values[model]=mm
  for j,metric in enumerate(METRICS):
   q95=qs[.95][j];q90=qs[.9][j];count=int((mm[:,j]<=q95).sum());r95=count/10;r90=float((mm[:,j]<=q90).mean());mean=float(mm[:,j].mean());bb=mm[root_ids,j];rb=(bb<=qboot[:,j,None]).mean(1);rho=bb.mean(1)/qboot[:,j];cp_low=0. if count==0 else float(beta.ppf(.025,count,11-count));cp_high=1. if count==10 else float(beta.ppf(.975,count+1,10-count))
   rvc.append(dict(cohort=name,model=model,metric=metric,mean_Root_distance=mean,q90_RR=float(q90),q95_RR=float(q95),rho_NV_group_mean=mean/q95,RVC90=r90,RVC95=r95,count_within95=count,N_source_Roots=10,RVC95_boot_CI_low=float(np.quantile(rb,.025)),RVC95_boot_CI_high=float(np.quantile(rb,.975)),rho_boot_CI_low=float(np.quantile(rho,.025)),rho_boot_CI_high=float(np.quantile(rho,.975)),proportion_exact_CI_low=cp_low,proportion_exact_CI_high=cp_high,target_point_estimate_met=r95>=.90,formal_acceptance='NOT_VALIDATED',Session_Root_exchangeability='NOT_ESTABLISHED'))
   for b in range(300):bootstrap.append(dict(cohort=name,model=model,metric=metric,replicate=b,q95_resampled_Sessions=float(qboot[b,j]),rho=float(rho[b]),RVC95=float(rb[b]),independent_Root_count=10))
 if cix==0:
  for model,mm in all_model_values.items():
   if model=='Original':continue
   for j,metric in enumerate(METRICS):
    delta=mm[:,j]-all_model_values['Original'][:,j];bd=delta[root_ids].mean(1);paired.append(dict(model=model,metric=metric,paired_mean_distance_delta_vsOriginal=float(delta.mean()),CI_low=float(np.quantile(bd,.025)),CI_high=float(np.quantile(bd,.975)),unit='same10sourceRoot, task-stratified bootstrap300',multiple_comparison_adjusted=False,interpretation='development comparison, not final confirmatory significance'))
 # Negative controls have no physical sourceRoot; the labels below are only artificial diagnostic blocks.
 ctrl0=np.broadcast_to(r['x'].mean(0),(905,4,712));pool=np.flatnonzero(np.isin(r['g'],cal));donor=rg.choice(pool,905,replace=True);controls={'Constant_RealTrain_mean':transform(ctrl0,mu,sd),'Unconditional_RealTrain_donor':R[donor]}
 for control,x in controls.items():
  values=[]
  for g in allroots:values.append(aggregate([distances(x[ii],R[jj],sigma) for ii,jj in zip(query_banks['Sim:'+g],refbank)]))
  for j,metric in enumerate(METRICS):
   vv=np.array([v[metric] for v in values]);negative.append(dict(cohort=name,control=control,metric=metric,mean_block_distance=float(vv.mean()),q95_RR=float(qs[.95][j]),diagnostic_block_fraction_below_q95=float((vv<=qs[.95][j]).mean()),block_count=10,unit='artificial diagnostic blocks, NOT independent SimRoot',source_condition_preserved=False,formal_acceptance=False))
 print('COHORT_DONE',name,'elapsed',round(time.time()-start,1),flush=True)
table(O/'real_real_distances.csv',rr);table(O/'root_distances.csv',rootrows);table(O/'q_RR_quantiles.csv',thresholds);table(O/'rvc_metrics.csv',rvc);table(O/'bootstrap_results.csv',bootstrap);table(O/'negative_control_checks.csv',negative);table(O/'paired_method_differences.csv',paired);table(O/'MonteCarlo_draw_distances.csv',drawrows)
# Unadjusted overall marginal distributions, all original windows, no task filtering/reweighting.
pooled=[]
for model,x in F.items():
 a=distances(x,RV,sigma);pooled.append(dict(model=model,scope='all original905source windows vs all780RealVal, unadjusted marginal',source_windows=len(x),Real_windows=len(RV),**a));print('FULL_MARGINAL',model,round(time.time()-start,1),flush=True)
table(O/'sim_real_distances.csv',[v for v in pooled if v['model']=='Original']);table(O/'translated_sim_real_distances.csv',[v for v in pooled if v['model']!='Original'])
# RR grouping-strategy sensitivity:64 disjoint4-vs4 Session domains, no new independent subjects.
macro=[];groups=np.unique(r['g']);rg=np.random.default_rng(4602)
for k in range(64):
 chosen=rg.choice(groups,8,replace=False);ga,gb=chosen[:4],chosen[4:];ia=np.flatnonzero(np.isin(r['g'],ga));ib=np.flatnonzero(np.isin(r['g'],gb));N=min(128,len(ia),len(ib));aa=rg.choice(ia,N,replace=False);bb=rg.choice(ib,N,replace=False);v=distances(R[aa],R[bb],sigma);macro.append(dict(split=k,Session_A=json.dumps(ga.tolist()),Session_B=json.dumps(gb.tolist()),windows_each=N,Trial_A=len(set('/'.join(str(u).split('/')[:2]) for u in r['u'][aa])),Trial_B=len(set('/'.join(str(u).split('/')[:2]) for u in r['u'][bb])),independent_total_Sessions=32,split_draws_not_independent=True,**v))
table(O/'RR_macro_grouping_sensitivity.csv',macro)
# Literal-stage and task views only: no certified physical condition matching.
stage=[];rg=np.random.default_rng(4603)
stage_banks={}
for st in sorted(set(str(s).lower() for s in rv['s']) & set(str(s).lower() for s in n['s'])):
 ia=np.flatnonzero(np.char.lower(n['s'])==st);ib=np.flatnonzero(np.char.lower(rv['s'])==st);N=min(128,len(ia),len(ib));stage_banks[st]=(rg.choice(ia,N,replace=False),rg.choice(ib,N,replace=False))
for model,x in F.items():
 for st,(aa,bb) in stage_banks.items():
  N=len(aa);v=distances(x[aa],RV[bb],sigma);stage.append(dict(model=model,stage=st,view='literal stage label only, not shared physical condition',N_each=N,**v))
 for task in sorted(set(n['task'])):
  ix=n['task']==task;v=distances(x[ix],RV,sigma);stage.append(dict(model=model,task=task,view='task-specific source vs fullReal reference, task confounded',N_source=int(ix.sum()),N_Real=len(RV),**v))
table(O/'task_stage_marginal_distances.csv',stage)
# Mechanism descriptors only; no latent result replaces full712D distances.
OFF=np.cumsum([0]+[32,32,56]*5+[112]);feature_rows=[]
for model,x in dict(outs,RealValidation=rv['x'],RealTrain=r['x']).items():
 mask=x>0;v=mask[:,1:];u=mask[:,:-1];mass=np.stack([x[:,:,l:h].sum(-1) for l,h in zip(OFF[:-1],OFF[1:])],-1);pos=x[x>0];feature_rows.append(dict(model=model,zero_fraction=float((x==0).mean()),mean_active_Patches=float((mass>0).sum(-1).mean()),positive_mean=float(pos.mean()),positive_P95=float(np.quantile(pos,.95)),positive_P99=float(np.quantile(pos,.99)),transition01=float((~u&v).sum()/max((~u).sum(),1)),transition10=float((u&~v).sum()/max(u.sum(),1)),absolute_frame_delta=float(np.abs(np.diff(x,axis=1)).mean()),mean_window_std=float(x.std(0).mean()),unique_windows=len({v.tobytes() for v in x})))
table(O/'structure_diagnostic.csv',feature_rows);assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'statistics_complete.json',dict(seconds=time.time()-start,full712D_preserved=True,independent_RealTrain_Sessions=32,primary_calibration_Sessions=24,reference_Sessions=8,SimVal_Roots=10,MC_draws_are_not_independent=True,no_new_generator_inference=True,not_formal_equivalence_acceptance=True));print('P46 STATISTICS COMPLETE',flush=True)
