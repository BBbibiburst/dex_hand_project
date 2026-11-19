"""Portable P45 group-checkpoint inference, same shared weights for requested K; no Real/GMM/C0 dependency."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model import Generator
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--key',default='values');p.add_argument('--output',required=True);p.add_argument('--K',type=int,choices=[1,2,4,8],required=True);a=p.parse_args();dest=Path(a.output)
 if dest.exists():raise FileExistsError('Output exists')
 torch.set_num_threads(2);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);G=Generator(**ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);x=np.load(a.input)[a.key]
 if x.dtype!=np.float32 or x.ndim!=3 or x.shape[1:]!=(4,712) or not np.isfinite(x).all() or (x<0).any():raise ValueError('Need finite nonnegative float32[N,4,712]')
 with torch.inference_mode():y=np.concatenate([G(torch.from_numpy(x[i:i+64]),a.K).numpy() for i in range(0,len(x),64)]) if len(x) else x.copy()
 assert y.shape==x.shape and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all();dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,values=y);print('SAVED',dest,'K',a.K,y.shape)
if __name__=='__main__':main()
