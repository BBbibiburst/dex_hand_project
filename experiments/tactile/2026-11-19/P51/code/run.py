from audit_utils import *
import torch
from architecture import MultiView
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P50=Path('outputs/tactile_p50_residual_multiview_2026-11-19');P49=Path('outputs/tactile_p49_round_frozen_adversarial_2026-11-19')
assert not (O/'registration.json').exists()
reg=dict(date='2026-11-19',purpose='P51 readonly frozenD50 reliability; no training',classification_threshold=.5,Real_joint_acceptance_necessary_for_GO=.9,original_discrimination='report BA/AUC/recalls; no posthoc hard threshold added',envelope=dict(calibration='8 historical D-inner Real Sessions; no D gradients, development exposed',evaluation='4 existing external Real Validation Sessions and10 Sim Roots; also development exposed',quantiles=[.025,.975],fit_weights='equal Sessions, uniform within each',coordinate='raw logits per view and minlogit Joint',interpretation='auxiliary description only, not threshold or acceptance replacement',LOSO='8calibrationSessions: fit7evaluate1 diagnostic; no selecting best split',no_envelope_hard_gate='no empirically justified stability tolerance; report uncertainty rather than invent one'),decision='NO-GO if existing fixed Real90 requirement fails; GO only if mandatory acceptance and discriminator reliability supported; UNKNOWN otherwise; no P52 on NO-GO/UNKNOWN',no_Test_Holdout=True,no_new_model=True,no_threshold_search=True,no_C0=True,no_production=True,no_commit_push=True)
save(O/'registration.json',reg)
protected=json.loads((P50/'protected_hashes_before.json').read_text())
for p in P50.rglob('*'):
 if p.is_file():protected[str(p)]=sha(p)
save(O/'protected_hashes_before.json',protected)
fit=np.load(P50/'D50_fit_probabilities.npz');cal=np.load(P50/'D50_inner_probabilities.npz');files={n:P50/f'outputs/{n}_D50_predictions.npz' for n in ['Original','F0','P49_F3','F50','F50_without_refiner','Constant','RealDonor','PairShuffle']};z={n:np.load(p) for n,p in files.items()};ref=z['F0'];rg=ref['real_groups'];sg=ref['sim_groups'];cg=cal['real_groups'];fg=fit['real_groups'];assert not set(cg)&set(fg) and not set(cg)&set(rg) and not set(fg)&set(rg)
assert (len(set(fg)),len(set(cg)),len(set(rg)),len(set(sg)))==(24,8,4,10)
for n,p in z.items():assert np.array_equal(p['real_logits'],ref['real_logits']) and np.array_equal(p['real_groups'],rg) and np.array_equal(p['sim_groups'],sg)
# Recompute frozen model inference only on external Real to verify scores, never fit.
torch.set_num_threads(2);ck=torch.load(P50/'checkpoints/D50.pt',map_location='cpu',weights_only=False);D=MultiView(ck['logscale']);D.load_state_dict(ck['D']);D.cuda().eval();D.requires_grad_(False);r=real('validation')
with torch.inference_mode():replay=torch.cat([D(torch.from_numpy(r['x'][i:i+128].copy()).cuda()).cpu() for i in range(0,len(r['x']),128)]).numpy()
assert np.allclose(replay,ref['real_logits'],rtol=1e-5,atol=1e-5);assert np.array_equal(replay>=0,ref['real_logits']>=0)
save(O/'replay.json',dict(D50_hash=sha(P50/'checkpoints/D50.pt'),max_logit_error=float(np.abs(replay-ref['real_logits']).max()),classification_identical=True,optimizer_steps=0))
def extend(a):return np.c_[a,a.min(1)]
branches=['Global','Spatial','Temporal','Joint'];realv=extend(ref['real_logits']);calv=extend(cal['real_logits']);fitv=extend(fit['real_logits']);bank=np.load('outputs/tactile_p48_unified_four_route_audit_2026-11-19/group_bootstrap_indices.npz')
def quant(x,w,qs):
 ix=np.argsort(x,kind='stable');v=x[ix];cw=np.cumsum(w[ix]);return np.interp(qs,cw/cw[-1],v)
env=np.array([quant(calv[:,j],weights(cg),[.025,.975]) for j in range(4)]);save(O/'score_envelope.json',dict(calibration_groups=sorted(set(cg)),evaluation_groups=sorted(set(rg)),bounds={b:dict(lower=float(env[j,0]),upper=float(env[j,1])) for j,b in enumerate(branches)},auxiliary_only=True))
summ=[];groups=[];metrics=[];coverage=[];loso=[];bs=[]
for domain,role,ll,gg in [('Real','Dfit',fitv,fg),('Real','calibration_inner',calv,cg),('Real','evaluation',realv,rg)]+[(n,'evaluation',extend(v['sim_logits']),sg) for n,v in z.items()]:
 for j,b in enumerate(branches):
  score=ll[:,j];w=weights(gg);qq=quant(score,w,[.01,.05,.25,.5,.75,.95,.99]);inside=(score>=env[j,0])&(score<=env[j,1]);cov=float(w@inside);acc=float(np.mean([(score[gg==g]>=0).mean() for g in np.unique(gg)]));summ.append(dict(domain=domain,role=role,branch=b,groups=len(set(gg)),windows=len(gg),acceptance_macro=acc,acceptance_micro=float((score>=0).mean()),logit_mean=float(w@score),logit_std=float(np.sqrt(w@(score-w@score)**2)),**{f'logit_P{int(q*100)}':float(v) for q,v in zip([.01,.05,.25,.5,.75,.95,.99],qq)},envelope_coverage=cov,below_envelope=float(w@(score<env[j,0])),above_envelope=float(w@(score>env[j,1]))))
  for g in np.unique(gg):ix=gg==g;groups.append(dict(domain=domain,role=role,branch=b,group=g,windows=int(ix.sum()),acceptance=float((score[ix]>=0).mean()),envelope_coverage=float(inside[ix].mean()),logit_P5=float(np.quantile(score[ix],.05)),logit_median=float(np.median(score[ix])),logit_P95=float(np.quantile(score[ix],.95))))
 forj=0
 # Separate AND of all three per-view envelopes; minlogit Joint envelope is a different auxiliary description.
 inside_all=((ll[:,:3]>=env[:3,0])&(ll[:,:3]<=env[:3,1])).all(1);coverage.append(dict(domain=domain,role=role,three_view_envelope_AND_macro=float(weights(gg)@inside_all)))
for n,v in z.items():
 sim=extend(v['sim_logits'])
 for j,b in enumerate(branches):
  rl=realv[:,j];sl=sim[:,j];m=evalstats(np.r_[np.ones(len(rl)),np.zeros(len(sl))],np.r_[rl,sl],np.r_[.5*weights(rg),.5*weights(sg)]);m['D']=1-2*abs(m['BA']-.5);ar=rl>=0;ass=sl>=0;rm=np.array([ar[rg==g].mean() for g in bank['Real_groups']]);sm=np.array([ass[sg==g].mean() for g in bank['Sim_groups']]);rb=rm[bank['Real']].mean(1);sb=sm[bank['Sim']].mean(1);m.update(model=n,branch=b,FR_macro=float(sm.mean()),Real_acceptance_macro=float(rm.mean()),Real_CI_low=float(np.quantile(rb,.025)),Real_CI_high=float(np.quantile(rb,.975)),FR_CI_low=float(np.quantile(sb,.025)),FR_CI_high=float(np.quantile(sb,.975)),Sim_recall_macro=float(1-sm.mean()),confusion_matrix=json.dumps([[int((~ass).sum()),int(ass.sum())],[int((~ar).sum()),int(ar.sum())]]));metrics.append(m)
  for i in range(1000):bs.append(dict(model=n,branch=b,replicate=i,FR=float(sb[i]),Real_acceptance=float(rb[i])))
for held in np.unique(cg):
 train=cg!=held;test=~train
 for j,b in enumerate(branches):
  lo,hi=quant(calv[train,j],weights(cg[train]),[.025,.975]);sc=calv[test,j];loso.append(dict(held_Session=held,branch=b,calibration_Sessions=7,lower=float(lo),upper=float(hi),held_coverage=float(((sc>=lo)&(sc<=hi)).mean()),held_acceptance=float((sc>=0).mean()),scope='historicalinnerLOSO diagnostic only'))
table(O/'score_summary.csv',summ);table(O/'session_root_scores.csv',groups);table(O/'classification_metrics.csv',metrics);table(O/'envelope_joint_coverage.csv',coverage);table(O/'envelope_LOSO.csv',loso);table(O/'bootstrap_results.csv',bs)
realjoint=next(v for v in summ if v['domain']=='Real' and v['role']=='evaluation' and v['branch']=='Joint');orig=next(v for v in metrics if v['model']=='Original' and v['branch']=='Joint');decision='NO-GO' if realjoint['acceptance_macro']<.9 else 'UNKNOWN';save(O/'decision.json',dict(decision=decision,reason='fixed Real Session Macro JointAcceptance below0.90; auxiliaryenvelope cannot repair',Real_joint_acceptance=realjoint['acceptance_macro'],Original_joint_BA=orig['BA'],Original_joint_AUC=orig['AUC'],Original_Sim_recall=orig['Sim_recall_macro'],P52_authorized_by_condition=False,P52_started=False,P53_started=False,training_steps=0))
(O/'figures').mkdir(exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for j,(ax,b) in enumerate(zip(axes.ravel(),branches)):
 data=[realv[:,j]]+[extend(z[n]['sim_logits'])[:,j] for n in ['Original','F0','P49_F3','F50']];ax.boxplot(data,labels=['RealVal','Original','F0','P49F3','F50'],showfliers=False);ax.axhspan(*env[j],color='green',alpha=.12,label='Inner8 envelope');ax.axhline(0,color='k',ls='--',lw=1);ax.set(title=b,ylabel='Frozen D50 logit');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(O/'figures/score_envelopes.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,4))
for j,b in enumerate(branches):
 vals=[v for v in groups if v['domain']=='Real' and v['role']=='evaluation' and v['branch']==b];axes[0].plot(range(4),[v['acceptance'] for v in vals],marker='o',label=b);axes[1].plot(range(4),[v['envelope_coverage'] for v in vals],marker='o',label=b)
axes[0].axhline(.9,c='k',ls='--');axes[0].set(title='Real Validation: fixed 0.5 acceptance',xlabel='Session (4 groups)',ylabel='Fraction',ylim=(0,1.05));axes[1].set(title='Auxiliary calibrated envelope coverage',xlabel='Session (4 groups)',ylabel='Fraction',ylim=(0,1.05))
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'figures/session_generalization.png',dpi=170);plt.close(fig)
assert all(sha(p)==h for p,h in protected.items());save(O/'integrity.json',dict(protected_files=len(protected),all_unchanged=True,group_disjoint=True,no_training=True,no_Test_Holdout=True,no_C0=True,no_server=True,no_P52=True,code_hashes={str(p.relative_to(O)):sha(p) for p in (O/'code').glob('*.py')}));print('P51 COMPLETE',decision,'Realjoint',realjoint['acceptance_macro'],'Original joint',orig,flush=True)
