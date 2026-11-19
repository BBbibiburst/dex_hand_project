from audit_utils import *
from model import Discriminator
import torch
assert (O/'complete.json').exists();torch.set_num_threads(2);r=real('validation');s=natural('validation');rt=real('train');bank=np.load(P47/'index_banks.npz');x={'Original':s['x']}
for p in sorted((O/'outputs').glob('F*_validation.npz')):x[p.name.split('_')[0]]=np.load(p)['values']
x.update(Constant=np.broadcast_to(rt['x'].mean(0),(905,4,712)).copy(),RealDonor=rt['x'][bank['donor_indices']]);rows=[];groups=[]
for dfile in sorted((O/'checkpoints').glob('D[0-3].pt')):
 ck=torch.load(dfile,map_location='cpu',weights_only=False);D=Discriminator(ck['logscale']);D.load_state_dict(ck['model']);D.eval();D.requires_grad_(False)
 def predict(a):
  with torch.inference_mode():return torch.cat([D(torch.from_numpy(np.array(a[i:i+256],copy=True))) for i in range(0,len(a),256)]).numpy().astype(float)
 lr=predict(r['x']);ra=np.array([(lr[r['g']==g]>=0).mean() for g in np.unique(r['g'])])
 for name,a in x.items():
  ls=predict(a);mac=np.array([(ls[s['g']==g]>=0).mean() for g in np.unique(s['g'])]);m=evalstats(np.r_[np.ones(780),np.zeros(905)],np.r_[lr,ls],np.r_[.5*weights(r['g']),.5*weights(s['g'])]);rows.append(dict(D=dfile.stem,F=name,FR_macro=float(mac.mean()),FR_micro=float((ls>=0).mean()),Real_macro=float(ra.mean()),BA=m['BA'],AUC=m['AUC'],device='CPU frozen diagnostic,not substituteGPUprimary'))
  for g in np.unique(s['g']):groups.append(dict(D=dfile.stem,F=name,Root=g,FR=float((ls[s['g']==g]>=0).mean()),windows=int((s['g']==g).sum())))
  pp=ls[bank['pair_permutation']];assert np.array_equal(s['g'],s['g'][bank['pair_permutation']]);assert abs(float(weights(s['g'])@(pp>=0))-float(weights(s['g'])@(ls>=0)))<1e-12
 table(O/'frozen_snapshot_matrix.csv',rows);table(O/'snapshot_group_metrics.csv',groups)
# Model/optimizercontinuity and originalstoredD1 protection.
review=json.loads((O/'D1_group_review_evidence.json').read_text());assert sha(O/'checkpoints/D1.pt')==review['D1_model_hash'];f0=torch.load(O/'checkpoints/F0.pt',map_location='cpu',weights_only=False);old=torch.load(P45/'A/generator.pt',map_location='cpu',weights_only=False);assert all(torch.equal(v,old['model'][k]) for k,v in f0['model'].items());hashes=[]
for p in sorted((O/'checkpoints').glob('[FD][0-3].pt')):
 c=torch.load(p,map_location='cpu',weights_only=False);hashes.append(dict(path=str(p),sha256=sha(p),optimizer_steps=c['cumulative_optimizer_steps'],optimizer_parameter_states=len(c['optimizer']['state'])))
table(O/'checkpoint_manifest.csv',hashes);save(O/'snapshot_audit_complete.json',dict(D1_matchespreG1_review_hash=True,F0_matches_P45A=True,PairShuffle_FR_invariant=True,no_training=True));print('SNAPSHOT AUDIT COMPLETE',flush=True)
