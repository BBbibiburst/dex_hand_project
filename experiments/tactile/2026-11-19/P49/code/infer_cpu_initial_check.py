"""Standalone frozen P49 F: raw nonnegative processed[N,4,712] -> translated[N,4,712], fixedK1."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model import Generator

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--key',default='values');p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(2);ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);G=Generator(**ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);src=np.load(args.input);x=src[args.key]
 if x.dtype!=np.float32 or x.ndim!=3 or x.shape[1:]!=(4,712) or not np.isfinite(x).all() or (x<0).any():raise ValueError('Expected finite nonnegativefloat32 [N,4,712] processedinput;not externallystandardized')
 with torch.inference_mode():y=np.concatenate([G(torch.from_numpy(x[i:i+64]),1).numpy() for i in range(0,len(x),64)]) if len(x) else x.copy()
 assert y.shape==x.shape and np.isfinite(y).all() and (y>=0).all();data={'values':y}
 for key in ['groups','stage','units']:
  if key in src.files:data[key]=src[key]
 out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**data);print('SAVED',out,y.shape,'K=1')
if __name__=='__main__':main()
