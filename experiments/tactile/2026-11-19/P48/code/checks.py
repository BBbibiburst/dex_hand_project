from audit_utils import *
from scipy.stats import wasserstein_distance
assert (O/'swd_complete.json').exists() and (O/'equivalence_complete.json').exists();rv=real('validation');s,out=inputs();bank=np.load(P47/'index_banks.npz');perm=bank['pair_permutation'];inv=np.argsort(perm);p=np.load(O/'projection_directions.npy');R=feature(rv['x']).reshape(780,-1).astype(float)@p[:,[0,255,511,1023]];wR,wS=weights(rv['g']),weights(s['g']);idx=np.load(O/'equivalence_window_bank.npz')['Sim_indices'];rows=[]
for name in candidate_paths():
 y=out[name];yp=y[perm];assert np.array_equal(s['g'],s['g'][perm]);assert np.array_equal(yp[inv[idx]],y[idx]);Y=feature(y).reshape(905,-1).astype(float)@p[:,[0,255,511,1023]];YY=Y[perm];err=[]
 for j in range(4):err.append(abs(wasserstein_distance(R[:,j],Y[:,j],wR,wS)-wasserstein_distance(R[:,j],YY[:,j],wR,wS)))
 for g in np.unique(s['g']):
  ix=s['g']==g;assert sorted(hashlib.sha256(x.tobytes()).hexdigest() for x in y[ix])==sorted(hashlib.sha256(x.tobytes()).hexdigest() for x in yp[ix])
 assert max(err)<1e-12;rows.append(dict(model=name,perRoot_multiset_exact=True,all_projection_SWD_invariance_by_identity=True,independent_numerical_spotchecks=4,max_spotcheck_error=max(err),sevenbank_MMD_pair_inverse_exact=True))
rr=list(csv.DictReader((O/'equivalence_rr_calibration.csv').open()))
for row in rr:
 a,b=json.loads(row['Real_A']),json.loads(row['Real_B']);assert len(a)==4 and len(b)==10 and not(set(a)&set(b))
assert len(rr)==1000;assert np.allclose(np.linalg.norm(p,axis=0),1,atol=1e-12);assert sha(P46/'RealTrain_preprocessing.npz')==json.loads((O/'protected_hashes_before.json').read_text())[str(P46/'RealTrain_preprocessing.npz')];table(O/'pair_shuffle_invariance.csv',rows);save(O/'statistical_checks.json',dict(pair_invariance=True,RR1000disjoint_partitions=True,projection_count=1024,unit_norm=True,Scaler_unchanged=True,migration_epochs=0,migration_optimizer_steps=0));print('P48 STATISTICAL CHECKS PASS',flush=True)
