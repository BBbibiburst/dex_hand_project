"""Only treatment: explicit occupancy representation / generator-side mask derivative."""
import torch
from torch import nn
class AblationDiscriminator(nn.Module):
 def __init__(self,mode,log_scale,epsilon,tau):
  super().__init__();assert mode in ('A','B','C');self.mode=mode;self.register_buffer('log_scale',torch.as_tensor(log_scale,dtype=torch.float32));self.register_buffer('epsilon',torch.as_tensor(epsilon,dtype=torch.float32));self.register_buffer('tau',torch.as_tensor(tau,dtype=torch.float32));self.net=nn.Sequential(nn.Linear(2848 if mode=='A' else 5696,192),nn.LayerNorm(192),nn.LeakyReLU(.2),nn.Linear(192,64),nn.LeakyReLU(.2),nn.Linear(64,1))
 def continuous(self,x):return (torch.log1p(x)/self.log_scale).flatten(1)
 def mask(self,x,mask_backward=True):
  if self.mode=='A':raise RuntimeError('Continuous-only D must never request binary Mask')
  hard=(x>0).float()
  if self.mode=='C' and mask_backward:
   soft=torch.sigmoid((x-self.epsilon)/self.tau);return hard+(soft-soft.detach())
  return hard
 def features(self,x,mask_backward=True):
  a=self.continuous(x)
  if self.mode=='A':return a,None,a
  m=self.mask(x,mask_backward).flatten(1);return torch.cat([m,a],1),m,a
 def forward(self,x,mask_backward=True):return self.net(self.features(x,mask_backward)[0]).squeeze(-1)
 def load_common_P44(self,state):
  own=self.state_dict()
  for k,v in state.items():
   if self.mode=='A' and k=='net.0.weight':own[k]=v[:,2848:].clone()
   else:own[k]=v.clone()
  self.load_state_dict(own)
