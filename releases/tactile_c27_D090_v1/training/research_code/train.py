"""One C3 GRU adaptation candidate: paired anchors + Natural coverage + Real-chart moments + source patch shares."""
from common import *
import time
from torch.nn import functional as F
assert json.loads((O/'baseline_reproduced.json').read_text())['training_may_proceed'];assert not (O/'config.json').exists();start=time.time();torch.set_num_threads(2);torch.manual_seed(4303);device='cuda:0' if torch.cuda.is_available() else 'cpu';R,S,T=load();initial={k:v.detach().clone() for k,v in T.state_dict().items()};R.to(device);S.to(device);T.to(device);T.requires_grad_(True);data={};OFF=np.cumsum([0]+[32,32,56]*5+[112]);trainvis=np.zeros(4049,int);pairvis=np.zeros(1200,int)
@torch.no_grad()
def means(net,x):return torch.cat([net.posterior_mean(torch.from_numpy(x[i:i+64]).to(device)).detach() for i in range(0,len(x),64)])
for role in ['train','validation']:
 pair=np.load(C3/f'paired_{role}.npz');n=natural(role);r=real(role);data[role]=dict(zs=means(S,pair['sim']),zr=means(R,pair['real']),target=torch.from_numpy(pair['real']).to(device),zn=means(S,n['x']),natural=torch.from_numpy(n['x']).to(device),real=means(R,r['x']))
mu=data['train']['real'].mean((0,1));sd=data['train']['real'].std((0,1),unbiased=False).clamp_min(.1)
def standardized(z):return (z-mu)/sd
# All moment targets are in Real Encoder coordinates, never S vs R coordinate distances.
def moment_target(z):
 a=standardized(z).reshape(len(z),-1);center=a-a.mean(0);return a.mean(0),center.T@center/len(a)
mt={role:moment_target(d['real']) for role,d in data.items()}
def moments(z,target):
 a=standardized(z).reshape(len(z),-1);center=a-a.mean(0);return (a.mean(0)-target[0]).square().mean()+(center.T@center/len(a)-target[1]).square().mean()
def shares(x):
 p=torch.stack([x[:,:,l:r].sum(-1) for l,r in zip(OFF[:-1],OFF[1:])],-1);return p/p.sum(-1,keepdim=True).clamp_min(1e-12)
def structure(y,x):
 # Soft auxiliary response-share model, used only in training loss, not inference output.
 # For gradients preserve fixed C3 mask logits via probabilities, rather than differentiating hard comparisons.
 return ((shares(y)-shares(x)).square().sum(-1)).mean()
def soft_output(z):
 values,logits,laws=R.decoder.run(z);mix,k,rate=laws;amp=(mix.softmax(-1)*k/rate).sum(-1)*R.decoder.positive_scale;return logits.sigmoid()*amp
@torch.no_grad()
def training_eval(role):
 d=data[role];anchor=[];zz=[];struct=[]
 for i in range(0,len(d['zs']),64):
  z=T(d['zs'][i:i+64]);h=F.huber_loss(z,d['zr'][i:i+64]);n=R.decoder.nll(z,d['target'][i:i+64])[0].mean();anchor.append([float(h),float(n),len(z)])
 for i in range(0,len(d['zn']),64):
  z=T(d['zn'][i:i+64]);zz.append(z);v=structure(soft_output(z),d['natural'][i:i+64]);struct.append([float(v),len(z)])
 a=np.array(anchor);ss=np.array(struct);hub=float((a[:,0]*a[:,2]).sum()/a[:,2].sum());nll=float((a[:,1]*a[:,2]).sum()/a[:,2].sum());return dict(Huber=hub,Decoder_NLL=nll,anchor=hub+.1*nll,moment=float(moments(torch.cat(zz),mt[role])),structure=float((ss[:,0]*ss[:,1]).sum()/ss[:,1].sum()))
T.eval();initialmetrics={role:training_eval(role) for role in data};scale=initialmetrics['train']['Huber']+.1*abs(initialmetrics['train']['Decoder_NLL']);lm=.25*scale/max(initialmetrics['train']['moment'],1e-12);ls=.25*scale/max(initialmetrics['train']['structure'],1e-12)
config=dict(method='C3 GRU-only Natural coverage adaptation with paired anchors and target-chart distribution/response-share losses',single_candidate=True,seed=4303,Real_Encoder_Decoder='original frozen C3, NOT C13 or C26',Sim_Encoder_Decoder='original frozen C3, independent ofReal',Translator='original C3 two-layer GRU181824 parameters, initialized from C3 best checkpoint',GMM='existing dictionary480 and physical paired1200/240 unchanged, no new pairing or physics',training_pairs=1200,natural_train=4049,real_train=5153,natural_validation=905,real_validation=780,paired_validation=240,loss='original pairHuber+.1 frozenRealDecoder NLL + lambda_m Real-chart mean/covariance + lambda_s source patch response-share',moment_scope='Translator output vs Real posterior ONLY in Real-chart; no direct independent-latent distance, no OT',source_structure='squared patch share difference, array-index proxy NOT certified physical map; sigmoidMask*GammaMean smooth readout for training only; original hardMAP decoder at deployment',lambda_m=lm,lambda_s=ls,weight_rule='each new term initially25% of Train anchor magnitude (Huber+.1 abs NLL); one fixed choice, no C0 feedback',initial_losses=initialmetrics,loss_scaling_mean_std='RealTrain-only loss coordinates, NOT input/C0 Scaler; never applied to deployed tactile',batch64=True,learning_rate=.0003,weight_decay=.0001,gradient_clip=5,max_epochs=45,min_checkpoint_epoch=10,patience=10,selection='Validation same composite objective, no C0; both paired and Natural validation/Real reference latent moments',budget_seconds=900,optimizer='AdamW new state, original GRU lr',Test_Holdout=False,additional_seed=False,no_future_GAN=True)
save(O/'config.json',config);save(O/'code_hashes_before_training.json',{str(p):sha(p) for p in (O/'code').glob('*.py')});(O/'candidate').mkdir();torch.save(dict(model=initial),O/'candidate/initial_translator.pt');opt=torch.optim.AdamW(T.parameters(),lr=.0003,weight_decay=.0001);history=[];best=float('inf');bestep=0;stale=0;steps=0;torch.cuda.reset_peak_memory_stats() if device.startswith('cuda') else None
for epoch in range(1,46):
 assert time.time()-start<900;T.train();rng=np.random.default_rng(4303+epoch);order=rng.permutation(4049);ids=np.concatenate([rng.permutation(1200) for _ in range(4)])[:4096];online=[]
 for j,i in enumerate(range(0,4049,64)):
  assert time.time()-start<900;ni=order[i:i+64];pi=ids[j*64:(j+1)*64];trainvis[ni]+=1;np.add.at(pairvis,pi,1);opt.zero_grad(set_to_none=True);d=data['train'];z=T(d['zs'][pi]);h=F.huber_loss(z,d['zr'][pi]);n=R.decoder.nll(z,d['target'][pi])[0].mean();zn=T(d['zn'][ni]);m=moments(zn,mt['train']);s=structure(soft_output(zn),d['natural'][ni]);loss=h+.1*n+lm*m+ls*s;assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(T.parameters(),5);assert torch.isfinite(norm);opt.step();steps+=1
 T.eval();tr=training_eval('train');va=training_eval('validation');selection=va['anchor']+lm*va['moment']+ls*va['structure'];row=dict(epoch=epoch,optimizer_steps=steps,natural_sample_visits=int(trainvis.sum()),paired_sample_visits=int(pairvis.sum()),**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()},selection_loss=selection,seconds=time.time()-start)
 if epoch>=10:
  if selection<best-1e-5:
   best=selection;bestep=epoch;stale=0;torch.save(dict(model={k:v.detach().cpu().clone() for k,v in T.state_dict().items()},epoch=epoch,optimizer_steps=steps,selection_loss=selection),O/'candidate/checkpoint.pt')
  else:stale+=1
 row['best_epoch']=bestep;history.append(row);table(O/'training_history.csv',history);print('EPOCH',json.dumps(row),flush=True)
 if epoch>=10 and stale>=10:break
assert np.all(trainvis==epoch);assert (pairvis>0).all();np.savez(O/'candidate/coverage.npz',natural_visits=trainvis,paired_visits=pairvis);torch.save(dict(model={k:v.detach().cpu().clone() for k,v in T.state_dict().items()},epoch=epoch),O/'candidate/endpoint.pt');save(O/'training_summary.json',dict(epochs=epoch,best_epoch=bestep,steps=steps,natural_unique=4049,natural_visits=int(trainvis.sum()),paired_unique=1200,paired_visits=int(pairvis.sum()),trainable_parameters=sum(p.numel() for p in T.parameters()),seconds=time.time()-start,device=device,GPU_peak_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,VAE_frozen=True,new_probe_training=0))
# Freeze single selected candidate before original C0 evaluation. Inference uses original deterministic C3 interface.
R,S,_=load();selected=torch.load(O/'candidate/checkpoint.pt',weights_only=False,map_location='cpu');T=M.Translator();T.load_state_dict(selected['model']);T.eval();T.requires_grad_(False);n=natural('validation');y=convert(R,S,T,n['x']);assert y.shape==(905,4,712) and np.isfinite(y).all() and (y>=0).all();np.savez_compressed(O/'candidate/translated_validation.npz',values=y,groups=n['g'],stage=n['s'],units=n['u'],tasks=n['tasks']);save(O/'candidate/frozen.json',dict(checkpoint_sha256=sha(O/'candidate/checkpoint.pt'),output_sha256=sha(O/'candidate/translated_validation.npz'),time=time.time(),selected_without_C0=True));r=real('validation');evaluate(O/'candidate',y,r,n);print('C27 CANDIDATE FINISHED NO_MORE_SEARCH',flush=True)
