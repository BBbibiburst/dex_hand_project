from support47 import *
import time
assert (O/'prdc_complete.json').exists() and not (O/'rr_bootstrap_complete.json').exists();t=time.time();r=real('train');z=np.load(P46/'RealTrain_preprocessing.npz');full=transform(r['x'],z['mean'],z['scale']);desc=descriptor(r['x']);ds=np.load(O/'descriptor_Train_preprocessing.npz');desc=(desc-ds['mean'])/ds['scale'];rows=[]
for space,R in [('full2848',full),('descriptor52',desc)]:
 for cohort in json.loads((P46/'registration.json').read_text())['reference_sensitivity']:
  ia=np.flatnonzero(np.isin(r['g'],cohort['reference']));ib=np.flatnonzero(np.isin(r['g'],cohort['calibration']));N=min(780,len(ia),len(ib));rng=np.random.default_rng(4701);a=rng.choice(ia,N,replace=False);b=rng.choice(ib,N,replace=False);aa,bb,ab=cdist(R[a],R[a]),cdist(R[b],R[b]),cdist(R[a],R[b]);ar,br=neighbor_radii(aa),neighbor_radii(bb);groupsa=np.unique(r['g'][a]);groupsb=np.unique(r['g'][b]);rng=np.random.default_rng(4701);banks=[]
  for rep in range(300):banks.append((np.concatenate([np.flatnonzero(r['g'][a]==g) for g in rng.choice(groupsa,len(groupsa),replace=True)]),np.concatenate([np.flatnonzero(r['g'][b]==g) for g in rng.choice(groupsb,len(groupsb),replace=True)])))
  for j,k in enumerate(KS):
   con=contributions(ab,ar[:,j],br[:,j],k)
   for rep,(asamp,bsamp) in enumerate(banks):rows.append(dict(model='RealReal_'+cohort['name'],space=space,k=k,replicate=rep,scope='draw0;Session cluster reweight fixed neighborhoods;not refit kNN',**{key:float(vv[bsamp if key in ['precision','density'] else asamp].mean()) for key,vv in con.items()}))
  print('RR_BOOTSTRAP',space,cohort['name'],round(time.time()-t,1),flush=True)
table(O/'real_real_bootstrap.csv',rows);save(O/'rr_bootstrap_complete.json',dict(seconds=time.time()-t,replicates=300,conditional_fixed_neighborhoods=True));print('RR BOOTSTRAP COMPLETE',flush=True)
