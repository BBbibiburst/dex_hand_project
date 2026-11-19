from audit_utils import *
from evaluate import Evaluation,structure
from architecture import MultiView
import torch
P49=Path('outputs/tactile_p49_round_frozen_adversarial_2026-11-19');assert (O/'complete.json').exists();torch.set_num_threads(2);ck=torch.load(O/'checkpoints/F50.pt',map_location='cpu',weights_only=False);D=MultiView(ck['logscale']);D.load_state_dict(ck['D']);D.cuda().eval();D.requires_grad_(False);r=real('validation');s=natural('validation');train=real('train');bank=np.load(Path('outputs/tactile_p47_conditional_transfer_evaluation_2026-11-18')/'index_banks.npz');boot=np.load(Path('outputs/tactile_p48_unified_four_route_audit_2026-11-19')/'group_bootstrap_indices.npz')
arrays={'F0':np.load(O/'outputs/F0_validation.npz')['values'],'Original':s['x'],'P49_F3':np.load(P49/'outputs/F3_validation.npz')['values'],'F50':np.load(O/'outputs/F50_validation.npz')['values'],'F50_without_refiner':np.load(O/'outputs/F50_without_refiner_validation.npz')['values']}
arrays.update(Constant=np.broadcast_to(train['x'].mean(0),(905,4,712)).copy(),RealDonor=train['x'][bank['donor_indices']],PairShuffle=arrays['F50'][bank['pair_permutation']])
def pred(x):
 with torch.inference_mode():return torch.cat([D(torch.from_numpy(x[i:i+128].copy()).cuda()).cpu() for i in range(0,len(x),128)]).numpy()
lr=pred(r['x']);rows=[];groups=[];boots=[]
for name,y in arrays.items():
 ls=pred(y);np.savez_compressed(O/f'outputs/{name}_D50_predictions.npz',real_logits=lr,sim_logits=ls,real_groups=r['g'],sim_groups=s['g'])
 for j,branch in enumerate(['Global','Spatial','Temporal','Joint']):
  ar=(lr[:,j]>=0) if j<3 else (lr>=0).all(1);aa=(ls[:,j]>=0) if j<3 else (ls>=0).all(1);rg=np.array([ar[r['g']==g].mean() for g in boot['Real_groups']]);sg=np.array([aa[s['g']==g].mean() for g in boot['Sim_groups']]);bb=sg[boot['Sim']].mean(1);br=rg[boot['Real']].mean(1)
  # Joint score=min branchlogit exactly implements AND atzero; AUC is diagnostic, not FRgate.
  rl=lr[:,j] if j<3 else lr.min(1);sl=ls[:,j] if j<3 else ls.min(1);m=evalstats(np.r_[np.ones(len(rl)),np.zeros(len(sl))],np.r_[rl,sl],np.r_[.5*weights(r['g']),.5*weights(s['g'])]);m.update(model=name,branch=branch,FR_macro=float(sg.mean()),FR_micro=float(aa.mean()),Real_acceptance_macro=float(rg.mean()),FR_CI_low=float(np.quantile(bb,.025)),FR_CI_high=float(np.quantile(bb,.975)),Real_CI_low=float(np.quantile(br,.025)),Real_CI_high=float(np.quantile(br,.975)),confusion_matrix=json.dumps([[int((~aa).sum()),int(aa.sum())],[int((~ar).sum()),int(ar.sum())]]));rows.append(m)
  for i in range(1000):boots.append(dict(model=name,branch=branch,replicate=i,FR=float(bb[i]),Real_acceptance=float(br[i])))
  for domain,gg,ac,logits in [('Real',r['g'],ar,rl),('Sim',s['g'],aa,sl)]:
   for g in np.unique(gg):ix=gg==g;groups.append(dict(model=name,branch=branch,domain=domain,group=g,windows=int(ix.sum()),acceptance=float(ac[ix].mean()),prob_mean=float(expit(logits[ix]).mean())))
 table(O/'acceptance_metrics.csv',rows);table(O/'acceptance_groups.csv',groups);table(O/'FR_bootstrap.csv',boots)
print('ALL D50 acceptance complete',flush=True)
E=Evaluation();quality=json.loads((O/'quality_summary.json').read_text()) if (O/'quality_summary.json').exists() else {}
# All point and guard formulas are byte-identical P49 evaluator; F0 first fixes reference.
for name,y in arrays.items():
 if name in quality:continue
 quality[name]=E.evaluate(name,y,name in ['Constant','RealDonor','PairShuffle']);save(O/'quality_summary.json',quality)
# Additional structure descriptions only; no new thresholds.
extra=[]
for name,y in dict(RealValidation=r['x'],**arrays).items():
 gg=r['g'] if name=='RealValidation' else s['g'];w=weights(gg);pos=y>0;both=pos[:,1:]&pos[:,:-1];d=np.abs(np.diff(y,axis=1));num=(both*d).sum((1,2));den=both.sum((1,2));mass=np.stack([y[:,:,a:b].sum(-1) for a,b in zip([0]+list(np.cumsum([32,32,56]*5)),np.cumsum([32,32,56]*5+[112]))],-1);shares=mass/np.maximum(mass.sum(-1,keepdims=True),1e-12)
 for g in np.concatenate([np.array(['ALL']),np.unique(gg)]):
  ix=np.ones(len(y),bool) if g=='ALL' else gg==g;ww=weights(gg[ix]);flat=y[ix].reshape(len(y[ix]),-1);nn=(flat>0).sum(1);pmean=np.divide(flat.sum(1),nn,out=np.zeros(len(flat)),where=nn>0)
  extra.append(dict(model=name,group=g,positive_mean=float(ww@pmean),mean_nonzero_Patches=float(ww@(mass[ix]>0).sum(-1).mean(1)),concentration=float(ww@(shares[ix]**2).sum(-1).mean(1)),both_positive_delta=float(ww@np.divide(num[ix],den[ix],out=np.zeros(len(num[ix]),float),where=den[ix]>0)),physical_spatial_mapping='index proxies',time='frameindexonly'))
table(O/'extra_structure_metrics.csv',extra)
q=quality['F50'];j=next(v for v in rows if v['model']=='F50' and v['branch']=='Joint');fr=j['FR_macro']>=.9 and j['Real_acceptance_macro']>=.9;ctrl=all(not quality[n]['all_quality_pass'] for n in ['Constant','RealDonor','PairShuffle']);ref=quality['F50_without_refiner'];ref_improved=q['SWD']<ref['SWD'] and q['MMD']<ref['MMD'];worse=q['SWD']>quality['F0']['SWD'] and q['MMD']>quality['F0']['MMD'];classification='A' if fr and q['all_quality_pass'] and ctrl else ('C' if not fr or q['severe_collapse'] or worse or not ref_improved else 'B')
save(O/'decision.json',dict(classification=classification,joint_FR_pass=fr,all_quality_pass=q['all_quality_pass'],negative_controls_rejected=ctrl,Refiner_SWD_MMD_improves_over_disabled_endpoint=ref_improved,overall_distribution_worse_than_F0=worse,severe_collapse=q['severe_collapse'],quality_gates=q['gates'],no_next_experiment=True))
print('P50 ASSESS COMPLETE',classification,j,flush=True)
