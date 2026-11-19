import torch
from torch import nn
from model import Generator
OFF=[0]
for n in [32,32,56]*5+[112]:OFF.append(OFF[-1]+n)
class Refiner(nn.Module):
 def __init__(self):
  super().__init__();self.project=nn.Sequential(nn.Linear(776,64),nn.GELU());self.temporal=nn.GRU(64,64,batch_first=True);self.out=nn.Linear(64,712);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,y,z):
  r=self.out(self.temporal(self.project(torch.cat([torch.log1p(y),z],-1)))[0]);l=torch.log1p(y)
  # Exact identity at r=0; nonnegative log response residual, clamp boundary subgradient=1.
  delta=torch.expm1((l+r).clamp_min(0))-torch.expm1(l)
  return (y+delta).clamp_min(0),r
class Transfer(nn.Module):
 def __init__(self,scales):super().__init__();self.base=Generator(**scales);self.refiner=Refiner()
 def from_latent(self,z,training=False):
  y,h,c=self.base.from_latent(z,1,training);out,res=self.refiner(y,z);return out,h,c,res
class MultiView(nn.Module):
 def __init__(self,scale):
  super().__init__();self.register_buffer('scale',torch.tensor(float(scale)))
  def head(n):return nn.Sequential(nn.Linear(n,128),nn.LayerNorm(128),nn.LeakyReLU(.2),nn.Linear(128,64),nn.LeakyReLU(.2),nn.Linear(64,1))
  self.global_head=head(2848);self.spatial_head=head(4*16*3);self.temporal_head=head(3*712)
 def features(self,x):
  l=torch.log1p(x)/self.scale
  # Fixed index Patch membership only. No physical nearest-neighbor assumptions, no binary Mask.
  f=[]
  for a,b in zip(OFF[:-1],OFF[1:]):
   p=l[:,:,a:b];f.extend([p.mean(-1),p.square().mean(-1),p.amax(-1)])
  return [l.flatten(1),torch.stack(f,-1).flatten(1),(l[:,1:]-l[:,:-1]).flatten(1)]
 def forward(self,x):return torch.stack([h(v).squeeze(-1) for h,v in zip([self.global_head,self.spatial_head,self.temporal_head],self.features(x))],1)
