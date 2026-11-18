import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata

class Probe(nn.Module):
    def __init__(self, dimension, kind):
        super().__init__()
        self.net = (
            nn.Linear(dimension, 1)
            if kind == "logistic"
            else nn.Sequential(
                nn.Linear(dimension, 192),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(192, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        )

    def forward(self, values):
        return self.net(values).squeeze(-1)

class DeepProbe(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  if kind=='mlp':self.net=Probe(2848,'mlp')
  elif kind=='gru':self.net=nn.GRU(712,128,num_layers=1,batch_first=True);self.head=nn.Sequential(nn.LayerNorm(128),nn.Dropout(.15),nn.Linear(128,1))
  elif kind=='transformer':
   self.project=nn.Linear(712,128);self.position=nn.Parameter(torch.zeros(1,4,128));nn.init.normal_(self.position,std=.02)
   self.net=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,256,.15,activation='gelu',batch_first=True),2,enable_nested_tensor=False);self.head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,1))
  else:raise ValueError(kind)
 def forward(self,x):
  if self.kind=='mlp':return self.net(x.flatten(1))
  if self.kind=='gru':return self.head(self.net(x)[1][-1]).squeeze(-1)
  return self.head(self.net(self.project(x)+self.position).mean(1)).squeeze(-1)

def metrics(y,p):
 y=np.asarray(y,int);p=np.asarray(p);pred=p>=.5;n1=int(sum(y==1));n0=int(sum(y==0))
 if not n0 or not n1:return {'status':'MISSING_DOMAIN','windows':len(y)}
 tpr=float(pred[y==1].mean());tnr=float((~pred[y==0]).mean());ba=(tpr+tnr)/2;ranks=rankdata(p,method='average');auc=float((ranks[y==1].sum()-n1*(n1+1)/2)/(n0*n1))
 return {'balanced_accuracy':ba,'roc_auc':auc,'domain_confusion':1-2*abs(ba-.5),'confusion_matrix':[[int(sum((y==0)&~pred)),int(sum((y==0)&pred))],[int(sum((y==1)&~pred)),int(sum((y==1)&pred))]],'recall_real':tpr,'recall_sim':tnr}
