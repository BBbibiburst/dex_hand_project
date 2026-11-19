"""Standalone frozen P49 F: raw nonnegative processed[N,4,712] -> translated[N,4,712], fixedK1."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model import Generator

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--key',default='values');p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(2);ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);G=Generator(**ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);src=np.load(args.input);x=src[args.key]
 if x.dtype!=np.float32 or x.ndim!=3 or x.shape[1:]!=(4,712) or not np.isfinite(x).all() or (x<0).any():raise ValueError('Expected finite nonnegativefloat32 [N,4,712] processedinput;not externallystandardized')
 device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is required for registeredbackend exact replay')
 with torch.inference_mode():z=torch.cat([G.sim.posterior_mean(torch.from_numpy(x[i:i+64])) for i in range(0,len(x),64)]) if len(x) else torch.zeros((0,4,64))
 G.to(device);G.eval()
 with torch.inference_mode():y=np.concatenate([G.from_latent(z[i:i+64].to(device),1,False)[0].cpu().numpy() for i in range(0,len(x),64)]) if len(x) else x.copy()
 assert y.shape==x.shape and np.isfinite(y).all() and (y>=0).all();data={'values':y}
 for key in ['groups','stage','units']:
  if key in src.files:data[key]=src[key]
 out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**data);print('SAVED',out,y.shape,'K=1','CPU Encoder +',device,'field/decoder')
if __name__=='__main__':main()
