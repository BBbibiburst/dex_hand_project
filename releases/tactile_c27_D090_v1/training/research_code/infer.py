"""Complete frozen C3 VAEs + selected GRU candidate; input/output[N,4,712]."""
from common import *
import argparse
torch.set_num_threads(2)
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--key',default='values');p.add_argument('--output',required=True);a=p.parse_args();dest=Path(a.output)
if dest.exists():raise FileExistsError('Do not overwrite output')
R,S,T=load();ck=torch.load(O/'candidate/checkpoint.pt',weights_only=False,map_location='cpu');T.load_state_dict(ck['model']);T.eval();T.requires_grad_(False);x=np.asarray(np.load(a.input)[a.key]);assert x.dtype==np.float32 and x.ndim==3 and x.shape[1:]==(4,712) and np.isfinite(x).all() and (x>=0).all();y=convert(R,S,T,x) if len(x) else x.copy();assert y.shape==x.shape and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all();dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,values=y);print('SAVED',dest,y.shape)
