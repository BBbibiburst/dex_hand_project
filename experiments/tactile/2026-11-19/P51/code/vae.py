"""Independent domain VAEs64 and two-layer GRU translator. No alignment losses.
Code/contract implementation only until real data-generation gates pass.
"""
import torch
from torch import nn
from torch.nn import functional as F
class Encoder(nn.Module):
 def __init__(self,scale):
  super().__init__();self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32));self.input=nn.Sequential(nn.Linear(1424,128),nn.GELU(),nn.LayerNorm(128));self.gru=nn.GRU(128,128,batch_first=True);self.mean=nn.Linear(128,64);self.logvar=nn.Linear(128,64)
 def forward(self,x):
  if x.ndim!=3 or x.shape[1:]!=(4,712):raise ValueError('Expected[B,4,712]')
  h=self.gru(self.input(torch.cat([(x>0).float(),torch.log1p(x)/self.scale],-1)))[0];return self.mean(h),self.logvar(h).clamp(-8,4)
class Decoder(nn.Module):
 def __init__(self,positive_scale):
  super().__init__();self.previous=nn.Sequential(nn.Linear(712,32),nn.GELU());self.bos=nn.Parameter(torch.zeros(32));self.cell=nn.GRUCell(96,128);self.mask=nn.Linear(128,712);self.current=nn.Sequential(nn.Linear(712,32),nn.GELU());self.amplitude=nn.Sequential(nn.Linear(160,128),nn.GELU(),nn.Linear(128,712*6));self.register_buffer('positive_scale',torch.as_tensor(positive_scale,dtype=torch.float32))
 def parameters_for(self,h,mask):
  logits,shape,rate=self.amplitude(torch.cat([h,self.current(mask)],1)).reshape(-1,712,6).split(2,-1);return logits,F.softplus(shape)+torch.finfo(shape.dtype).eps,F.softplus(rate)+torch.finfo(rate.dtype).eps
 def run(self,z,target=None):
  if z.ndim!=3 or z.shape[1:]!=(4,64):raise ValueError('Expected latent[B,4,64]')
  h=z.new_zeros((len(z),128));prev=z.new_zeros((len(z),712));values=[];ps=[];laws=[]
  for t in range(4):
   context=self.bos[None].expand(len(z),-1) if t==0 else self.previous(prev);h=self.cell(torch.cat([z[:,t],context],1),h);p=self.mask(h);hard=(p>=0).float();current=hard if target is None else (target[:,t]>0).float();law=self.parameters_for(h,current);amp=(law[0].softmax(-1)*law[1]/law[2]).sum(-1)*self.positive_scale;values.append(hard*amp);ps.append(p);laws.append(law);prev=current
  x=torch.stack(values,1)
  if not torch.isfinite(x).all() or (x<0).any():raise ValueError('Nonfinite/negative generated tactile; no clipping')
  return x,torch.stack(ps,1),tuple(torch.stack([q[k] for q in laws],1) for k in range(3))
 def deterministic_decode(self,z):return self.run(z)[0]
 def nll(self,z,target):
  _,logits,(mix,k,rate)=self.run(z,target);active=target>0;binary=F.binary_cross_entropy_with_logits(logits,active.float(),reduction='none').mean((1,2));scale=self.positive_scale.double();y=torch.where(active,target,torch.ones_like(target)).double()/scale;xx=y[...,None];kk=k.double();rr=rate.double();logp=kk*rr.log()-torch.lgamma(kk)+(kk-1)*xx.log()-rr*xx;logp=torch.logsumexp(mix.double().log_softmax(-1)+logp,-1)-scale.log();positive=-(logp*active).sum((1,2))/active.sum((1,2)).clamp_min(1);return binary+positive.to(binary.dtype),binary,positive.to(binary.dtype)
class DomainVAE(nn.Module):
 def __init__(self,encoder_scale,positive_scale):super().__init__();self.encoder=Encoder(encoder_scale);self.decoder=Decoder(positive_scale)
 def posterior_mean(self,x):return self.encoder(x)[0]
 def forward(self,x,sample=True):
  mu,lv=self.encoder(x);z=mu+torch.exp(.5*lv)*torch.randn_like(mu) if sample else mu;nll,mask,amp=self.decoder.nll(z,x);kl=.5*(mu.square()+lv.exp()-1-lv).mean((1,2));return {'mu':mu,'logvar':lv,'NLL':nll,'mask_loss':mask,'amplitude_loss':amp,'KL':kl}
class Translator(nn.Module):
 def __init__(self):super().__init__();self.gru=nn.GRU(64,128,2,batch_first=True);self.projection=nn.Linear(128,64)
 def forward(self,mu_sim):return self.projection(self.gru(mu_sim)[0])
