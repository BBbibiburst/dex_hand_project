"""Frozen original C0. Reports D-only and original joint verdict separately."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from probes import DeepProbe,metrics
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--generated',required=True);p.add_argument('--real',required=True);p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
 if out.exists():raise FileExistsError('Output already exists')
 torch.set_num_threads(2);frozen=json.loads((ROOT/'c0/frozen.json').read_text())
 for path,h in frozen['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
 gen=np.load(args.generated)['values'];real=np.load(args.real)['values']
 for a in [gen,real]:assert a.dtype==np.float32 and a.ndim==3 and a.shape[1:]==(4,712) and np.isfinite(a).all() and (a>=0).all()
 sc=np.load(ROOT/'c0/scaler.npz');assert str(sc['transform'])=='log1p';x=((np.log1p(np.concatenate([gen,real]))-sc['mean'])/sc['scale']).astype('float32');labels=np.r_[np.zeros(len(gen),int),np.ones(len(real),int)];rows=[]
 for e in frozen['instances']:
  kind=e['kind'];folder=ROOT/'c0'/e['name'];parts=[]
  with torch.inference_mode():
   if kind=='logistic':ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
   elif kind=='knn':ck=np.load(folder/'checkpoint.npz');assert int(ck['k'])==5;ref=torch.from_numpy(ck['values'].reshape(len(ck['labels']),-1)).double();lab=torch.from_numpy(ck['labels']).double();rn=(ref*ref).sum(1)
   else:ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);model=DeepProbe(kind);model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
   for i in range(0,len(x),128):
    q=torch.from_numpy(x[i:i+128])
    if kind=='logistic':pred=(q.flatten(1).double()@ck['weight']+ck['bias']).sigmoid()
    elif kind=='knn':q=q.flatten(1).double();dd=(q*q).sum(1)[:,None]+rn[None]-2*q@ref.T;pred=lab[dd.topk(5,largest=False,dim=1).indices].mean(1)
    else:pred=model(q).sigmoid()
    parts.append(pred.numpy())
  m=metrics(labels,np.concatenate(parts));rows.append(dict(instance=e['name'],**m,D_only_pass=m['domain_confusion']>=.9,original_joint_pass=.45<=m['balanced_accuracy']<=.55 and .475<=m['roc_auc']<=.525 and m['domain_confusion']>=.9))
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(dict(D_only_pass_count=sum(r['D_only_pass'] for r in rows),original_joint_pass_count=sum(r['original_joint_pass'] for r in rows),instances=11),ensure_ascii=False))
if __name__=='__main__':main()
