"""One shared source-conditioned Euler field, frozen independent VAEs, one full-touch D."""
import math
import torch
from torch import nn
from vae import DomainVAE,Translator
class VectorField(nn.Module):
 def __init__(self):
  super().__init__();self.translator=Translator();self.project=nn.Sequential(nn.Linear(132,128),nn.GELU());self.temporal=nn.GRU(128,128,batch_first=True);self.out=nn.Linear(128,64);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,h,z,t):
  time=z.new_tensor([t,1-t,math.sin(math.pi*t),math.cos(math.pi*t)]).view(1,1,4).expand(len(z),4,4);correction=self.out(self.temporal(self.project(torch.cat([h,z,time],-1)))[0]);return self.translator(z)+correction,correction
class Generator(nn.Module):
 def __init__(self,real_encoder_scale,real_positive_scale,sim_encoder_scale,sim_positive_scale):
  super().__init__();self.real=DomainVAE(real_encoder_scale,real_positive_scale);self.sim=DomainVAE(sim_encoder_scale,sim_positive_scale);self.field=VectorField();self.register_buffer('initial_state',torch.zeros(1,4,64));self.real.requires_grad_(False);self.sim.requires_grad_(False)
 def integrate(self,z,K):
  if K not in (1,2,4,8):raise ValueError('K must be1/2/4/8')
  h=self.initial_state.expand(len(z),-1,-1);corrections=[]
  for k in range(K):v,c=self.field(h,z,k/K);h=h+v/K;corrections.append(c)
  return h,torch.stack(corrections)
 def decode(self,z,straight_through=False):
  if not straight_through:return self.real.decoder.deterministic_decode(z)
  D=self.real.decoder;hidden=z.new_zeros((len(z),128));prev=z.new_zeros((len(z),712));ys=[]
  for t in range(4):
   context=D.bos[None].expand(len(z),-1) if t==0 else D.previous(prev);hidden=D.cell(torch.cat([z[:,t],context],1),hidden);logits=D.mask(hidden);p=logits.sigmoid();hard=(logits>=0).float();mask=hard+(p-p.detach());mix,shape,rate=D.parameters_for(hidden,mask);amplitude=(mix.softmax(-1)*shape/rate).sum(-1)*D.positive_scale;ys.append(mask*amplitude);prev=mask
  y=torch.stack(ys,1)
  if not torch.isfinite(y).all() or (y<0).any():raise ValueError('Nonfinite/negative straight-through forward')
  return y
 def from_latent(self,z,K,straight_through=False):
  h,c=self.integrate(z,K);return self.decode(h,straight_through),h,c
 def forward(self,x,K=1):return self.from_latent(self.sim.posterior_mean(x),K,False)[0]
class Discriminator(nn.Module):
 def __init__(self,log_scale):
  super().__init__();self.register_buffer('log_scale',torch.as_tensor(log_scale,dtype=torch.float32));self.net=nn.Sequential(nn.Linear(5696,192),nn.LayerNorm(192),nn.LeakyReLU(.2),nn.Linear(192,64),nn.LeakyReLU(.2),nn.Linear(64,1))
 def forward(self,x):
  features=torch.cat([(x>0).float().flatten(1),(torch.log1p(x)/self.log_scale).flatten(1)],1);return self.net(features).squeeze(-1)
