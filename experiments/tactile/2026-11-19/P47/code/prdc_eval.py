from support47 import *
import time
assert (O/'design_complete.json').exists() and not (O/'prdc_complete.json').exists()
t0=time.time();r=real('train');v=real('validation');n,d=datasets();reps=representations(r,v,d);banks=np.load(O/'index_banks.npz');indices=banks['equal_indices'];pair=banks['pair_permutation'];metrics=[];rootrows=[];bs=[];rrrows=[];rawdraws=[];checks=[];group_arrays={g:np.flatnonzero(n['g']==g) for g in np.unique(n['g'])};grouptask={g:n['task'][ix[0]] for g,ix in group_arrays.items()};real_groups={g:np.flatnonzero(v['g']==g) for g in np.unique(v['g'])};config=json.loads((O/'experiment_config.json').read_text())
# Verify original PRDC formula independently on an exact small example.
a=np.array([[0.],[1.],[3.],[7.],[11.]]);b=np.array([[.2],[2.],[4.],[9.],[13.]])
for k in [1,2]:
 xx=cdist(a,a);yy=cdist(b,b);xy=cdist(a,b);radx=np.sort(xx,axis=1)[:,k];rady=np.sort(yy,axis=1)[:,k];ours=prdc(xy,radx,rady,k);manual=dict(precision=sum(any(abs(float(a[i,0]-b[j,0]))<radx[i] for i in range(5)) for j in range(5))/5,recall=sum(any(abs(float(a[i,0]-b[j,0]))<rady[j] for j in range(5)) for i in range(5))/5,density=sum(sum(xy[i,j]<radx[i] for i in range(5)) for j in range(5))/(5*k),coverage=sum(min(xy[i])<radx[i] for i in range(5))/5);assert all(abs(ours[z]-manual[z])<1e-12 for z in ours)
for space,(realval,fd,realtrain) in reps.items():
 realdd=cdist(realval,realval);realrad=neighbor_radii(realdd)
 for model,fake in fd.items():
  cross=cdist(realval,fake);fakedd=cdist(fake,fake);allfrad=neighbor_radii(fakedd);vals=[]
  for draw,ix in enumerate(indices):
   c=cross[:,ix];frad=neighbor_radii(fakedd[np.ix_(ix,ix)])
   for j,k in enumerate(KS):
    value=prdc(c,realrad[:,j],frad[:,j],k);vals.append(dict(draw=draw,k=k,**value));rawdraws.append(dict(model=model,space=space,draw=draw,k=k,**value))
  for j,k in enumerate(KS):
   vs=[value for value in vals if value['k']==k];row=dict(model=model,space=space,k=k,N_real=780,N_generated=780,draws=100,statistical_unit='same4RealSessions/10sourceRoots;100draws not independent')
   for key in ['precision','recall','density','coverage']:
    aa=np.array([z[key] for z in vs]);row[key]=float(aa.mean());row[key+'_median']=float(np.median(aa));row[key+'_draw_std']=float(aa.std());row[key+'_draw_P025']=float(np.quantile(aa,.025));row[key+'_draw_P975']=float(np.quantile(aa,.975))
   metrics.append(row)
   # Conditional grouped bootstrap of draw0 point contributions; local geometry stays fixed.
   ix=indices[0];frad=neighbor_radii(fakedd[np.ix_(ix,ix)])[:,j];con=contributions(cross[:,ix],realrad[:,j],frad,k);rng=np.random.default_rng(4701);groups=sorted(set(n['g']));tasks=sorted(set(n['task']))
   for bi in range(300):
    rg=rng.choice(list(real_groups),len(real_groups),replace=True);ri=np.concatenate([real_groups[g] for g in rg]);gg=np.concatenate([rng.choice([g for g in groups if grouptask[g]==task],2,replace=True) for task in tasks]);fi=np.concatenate([np.flatnonzero(n['g'][ix]==g) for g in gg]);br={key:float(value[fi if key in ['precision','density'] else ri].mean()) for key,value in con.items()};bs.append(dict(model=model,space=space,k=k,replicate=bi,scope='draw0;cluster reweight fixed neighborhood contributions, not refitkNN',**br))
   for g,gix in group_arrays.items():
    value=prdc(cross[:,gix],realrad[:,j],allfrad[gix,j] if False else neighbor_radii(fakedd[np.ix_(gix,gix)])[:,j],k);rootrows.append(dict(model=model,space=space,k=k,Root=g,task=grouptask[g],N_generated=len(gix),N_real=780,note='unequal sizes; descriptive Root check only; local generated radius withinRoot',**value))
  # Pair-shuffle full sample invariance: exact output permutation, including local radii.
  if model in candidate_paths():
   for j,k in enumerate(KS):
    before=prdc(cross,realrad[:,j],allfrad[:,j],k);after=prdc(cross[:,pair],realrad[:,j],allfrad[pair,j],k);assert all(abs(before[z]-after[z])<1e-12 for z in before);checks.append(dict(model=model,space=space,k=k,full905_PRDC_exact_invariant=True,max_abs_error=max(abs(before[z]-after[z]) for z in before),MC_rule='inverseperm indices preserve same output subsets'))
   for row in [z for z in metrics if z['model']==model and z['space']==space]:metrics.append(dict(row,model=model+'_PairShuffle',derivation='exact same output multiset and inverseperm samplebank; no recalculation needed'))
  print('PRDC',space,model,'seconds',round(time.time()-t0,1),flush=True)
 # RR uses P46 fixed cohorts, independently Session-disjoint pools, equal per side.
 for cohort in json.loads((P46/'registration.json').read_text())['reference_sensitivity']:
  ia=np.flatnonzero(np.isin(r['g'],cohort['reference']));ib=np.flatnonzero(np.isin(r['g'],cohort['calibration']));N=min(780,len(ia),len(ib));rng=np.random.default_rng(4701);pool=np.r_[ia,ib];full=cdist(realtrain[pool],realtrain[pool]);aa=np.arange(len(ia));bb=np.arange(len(ia),len(pool))
  for draw in range(100):
   a=rng.choice(aa,N,replace=False);b=rng.choice(bb,N,replace=False);ar=neighbor_radii(full[np.ix_(a,a)]);br=neighbor_radii(full[np.ix_(b,b)]);cr=full[np.ix_(a,b)]
   for j,k in enumerate(KS):rrrows.append(dict(cohort=cohort['name'],space=space,draw=draw,k=k,N_each=N,Session_A=json.dumps(cohort['reference']),Session_B=json.dumps(cohort['calibration']),independent_Sessions=32,draws_not_independent=True,**prdc(cr,ar[:,j],br[:,j],k)))
  print('RR_PRDC',space,cohort['name'],round(time.time()-t0,1),flush=True)
table(O/'prdc_metrics.csv',metrics);table(O/'prdc_sampling_draws.csv',rawdraws);table(O/'real_real_prdc.csv',rrrows);table(O/'root_metrics.csv',rootrows);table(O/'bootstrap_results.csv',bs);table(O/'pair_shuffle_invariance.csv',checks);assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'prdc_complete.json',dict(seconds=time.time()-t0,spaces=list(reps),candidates=len(d),formula_handcheck=True,pair_multiset_invariance=True,no_training=True));print('P47 PRDC COMPLETE',flush=True)
