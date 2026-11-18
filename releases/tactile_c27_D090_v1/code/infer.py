"""Standalone frozen Dual-VAE + GRU conversion; no Real/C0 dependency."""
import argparse
from pathlib import Path
import numpy as np
import torch
from models import DomainVAE,Translator
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--key',default='values');p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
 if out.exists():raise FileExistsError('Output already exists')
 torch.set_num_threads(2);net=[]
 for name in ['real','sim']:
  ck=torch.load(ROOT/'models'/f'{name}_vae.pt',map_location='cpu',weights_only=False);m=DomainVAE(ck['encoder_scale'],ck['positive_scale']);m.load_state_dict(ck['model']);m.eval();m.requires_grad_(False);net.append(m)
 real,sim=net;ck=torch.load(ROOT/'models/gru_translator.pt',map_location='cpu',weights_only=False);T=Translator();T.load_state_dict(ck['model']);T.eval();T.requires_grad_(False);x=np.load(args.input)[args.key]
 if x.dtype!=np.float32 or x.ndim!=3 or x.shape[1:]!=(4,712) or not np.isfinite(x).all() or (x<0).any():raise ValueError('Expected finite nonnegative float32[N,4,712]')
 with torch.inference_mode():y=np.concatenate([real.decoder.deterministic_decode(T(sim.posterior_mean(torch.from_numpy(x[i:i+64])))).numpy() for i in range(0,len(x),64)]) if len(x) else x.copy()
 assert y.shape==x.shape and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all();out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,values=y);print('SAVED',out,y.shape)
if __name__=='__main__':main()
