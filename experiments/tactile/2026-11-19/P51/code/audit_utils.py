from common import *
from scipy.special import expit
P46=Path('outputs/tactile_p46_real_variability_envelope_audit_2026-11-18');P47=Path('outputs/tactile_p47_conditional_transfer_evaluation_2026-11-18')
def weights(groups):
 gs,counts=np.unique(groups,return_counts=True);lookup={g:1/(len(gs)*c) for g,c in zip(gs,counts)};return np.array([lookup[g] for g in groups])
def feature(x):
 z=np.load(P46/'RealTrain_preprocessing.npz');return transform(x,z['mean'],z['scale']).reshape(-1,4,712).astype('float32')
def evalstats(y,l,w):
 p=expit(l);ce=float(np.sum(w*(np.logaddexp(0.,l)-y*l)));pos=y==1;neg=~pos;ba=.5*(np.sum(w[pos]*(l[pos]>=0))/w[pos].sum()+np.sum(w[neg]*(l[neg]<0))/w[neg].sum());ix=np.argsort(p,kind='stable');ps=p[ix];yy=y[ix];ww=w[ix];starts=np.r_[0,np.flatnonzero(np.diff(ps)!=0)+1];pw=np.add.reduceat(ww*yy,starts);nw=np.add.reduceat(ww*(1-yy),starts);auc=float(np.sum(pw*(np.cumsum(nw)-nw+.5*nw))/(pw.sum()*nw.sum()));return dict(CE=ce,BA=float(ba),AUC=auc,JSD_proxy=float(np.log(2)-ce))
def candidate(role,name):
 if name=='Original':return natural(role)['x']
 if role=='validation':return np.load(candidate_paths()[name])['values']
 return np.load(O/f'outputs/{name}_train.npz')['values']
def bootstrap_scores(real_loss,sim_loss,baseline_real_loss,baseline_sim_loss,rg,sg):
 bank=np.load(O/'group_bootstrap_indices.npz');rmeans=np.array([real_loss[rg==g].mean() for g in bank['Real_groups']]);smeans=np.array([sim_loss[sg==g].mean() for g in bank['Sim_groups']]);br=np.array([baseline_real_loss[rg==g].mean() for g in bank['Real_groups']]);bs=np.array([baseline_sim_loss[sg==g].mean() for g in bank['Sim_groups']]);ce=.5*rmeans[bank['Real']].mean(1)+.5*smeans[bank['Sim']].mean(1);ce0=.5*br[bank['Real']].mean(1)+.5*bs[bank['Sim']].mean(1);return np.log(2)-ce,np.log(2)-ce0
