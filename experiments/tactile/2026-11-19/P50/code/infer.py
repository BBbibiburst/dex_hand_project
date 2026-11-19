import argparse
from pathlib import Path
import numpy as np
import torch
from architecture import Transfer
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda',choices=['cpu','cuda']);a=p.parse_args();torch.set_num_threads(2);out=Path(a.output)
if out.exists():raise FileExistsError(out)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);G=Transfer(ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);data=np.load(a.input);x=data['values'];assert x.dtype==np.float32 and x.ndim==3 and x.shape[1:]==(4,712) and np.isfinite(x).all() and (x>=0).all()
with torch.inference_mode():z=torch.cat([G.base.sim.posterior_mean(torch.from_numpy(x[i:i+64])) for i in range(0,len(x),64)])
G.to(a.device)
with torch.inference_mode():y=torch.cat([G.from_latent(z[i:i+64].to(a.device))[0].cpu() for i in range(0,len(x),64)]).numpy()
assert y.shape==x.shape and np.isfinite(y).all() and (y>=0).all();out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,values=y,**{k:data[k] for k in ['groups','stage','units','task'] if k in data.files});print('SAVED',out)
