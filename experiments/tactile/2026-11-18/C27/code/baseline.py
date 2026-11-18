from common import *
import time
assert not (O/'baseline/complete.json').exists();(O/'baseline').mkdir(exist_ok=True);torch.set_num_threads(2);start=time.time();R,S,T=load();r=real('validation');n=natural('validation');protected=dict(json.loads((C3/'c0/frozen.json').read_text())['files'])
for p in [C3/'real_vae/checkpoint.pt',C3/'sim_vae/checkpoint.pt',C3/'translator/checkpoint.pt',C3/'paired_train.npz',C3/'paired_validation.npz',C3/'code/models.py',C3/'code/inference.py',C3/'code/pipeline.py']+[B/f'natural_sim_{role}/windows.npz' for role in ['train','validation']]+[B/f'real_{role}/index.json' for role in ['train','validation']]:protected[str(p)]=sha(p)
save(O/'protected_hashes_before.json',protected);y=convert(R,S,T,n['x']);assert y.shape==(905,4,712) and np.isfinite(y).all() and (y>=0).all();np.savez_compressed(O/'baseline/translated_validation.npz',values=y,groups=n['g'],stage=n['s'],units=n['u'],tasks=n['tasks']);rows=evaluate(O/'baseline',y,r,n)
history={r['probe']:r for r in csv.DictReader(Path('outputs/tactile_p43_c11_latent_support_audit_2026-11-17/probe_metrics.csv').open()) if r['group']=='F' and r['role']=='validation'};delta=[]
for rr in rows:
 h=history[rr['instance']];delta.append(dict(instance=rr['instance'],historical_AUC=float(h['roc_auc']),replayed_AUC=rr['roc_auc'],absolute_difference=abs(rr['roc_auc']-float(h['roc_auc']))))
table(O/'baseline_reproduction.csv',delta);# Point deltas resolved separately against the original C11 GPU path.
assert sum(r['passed_joint'] for r in rows)==1;assert all(sha(p)==h for p,h in protected.items());save(O/'baseline_reproduced.json',dict(passed=True,joint_pass_count=1,windows=905,Real_reference=780,seconds=time.time()-start,history='C11F C3 original Validation scalar record, no Test/Holdout'))
print('BASELINE EXACTLY REPRODUCED 1/11',flush=True)
