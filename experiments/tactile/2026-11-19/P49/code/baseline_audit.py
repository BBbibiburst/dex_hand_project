from audit_utils import *
import evaluate as ev
from model import Discriminator
import torch
P49=O;P48=Path('outputs/tactile_p48_unified_four_route_audit_2026-11-19');torch.set_num_threads(1);folder=O/'stopped_audit';folder.mkdir(exist_ok=True);ev.O=folder;E=ev.Evaluation();y=np.load(P49/'outputs/F0_validation.npz')['values'];q0=E.evaluate('F0',y);qraw=E.evaluate('Original',E.s['x']);qp=E.evaluate('F0_PairShuffle',y[E.pair]);bank=np.load(P47/'index_banks.npz');r=real('train');controls={'Constant':np.broadcast_to(r['x'].mean(0),(905,4,712)).copy(),'RealDonor':r['x'][bank['donor_indices']],'PairShuffle':y[E.pair]};rows=[]
for phase in [0,1]:
 ck=torch.load(P49/f'checkpoints/D{phase}.pt',map_location='cpu',weights_only=False);D=Discriminator(ck['logscale']);D.load_state_dict(ck['model']);D.eval();D.requires_grad_(False)
 def pred(x):
  with torch.inference_mode():return torch.cat([D(torch.from_numpy(np.array(x[i:i+256],copy=True))) for i in range(0,len(x),256)]).numpy()
 for name,x in dict(F0=y,Original=E.s['x'],**controls).items():
  scores=pred(x);pReal=pred(E.r['x']);a=np.array([(scores[E.s['g']==g]>=0).mean() for g in np.unique(E.s['g'])]);ra=np.array([(pReal[E.r['g']==g]>=0).mean() for g in np.unique(E.r['g'])]);rows.append(dict(discriminator=f'D{phase}',model=name,FR_macro=float(a.mean()),FR_micro=float((scores>=0).mean()),Real_acceptance_macro=float(ra.mean()),G_optimizer_steps=0,scope='stopped baseline diagnosis;not F1 result'))
table(P49/'baseline_controls.csv',rows);save(P49/'baseline_supplement_complete.json',dict(no_training=True,Original_baseline_in=str(folder),PairShuffle_in=str(folder),states_not_modified=True));print('P49 BASELINE AUDIT COMPLETE',flush=True)
