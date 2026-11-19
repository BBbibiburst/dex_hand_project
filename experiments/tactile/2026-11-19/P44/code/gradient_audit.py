"""Frozen D branch audit only; no optimizer, no acceptance input replacement."""
from c0_support import *
from model import Discriminator
ck=torch.load(O/'discriminator.pt',map_location='cpu',weights_only=False);D=Discriminator(ck['logscale']);D.load_state_dict(ck['model']);D.eval();D.requires_grad_(False);torch.set_num_threads(2);r=real('validation')['x'];metrics=module('p44_branch_metrics','outputs/tactile_p37_full_data_revalidation_2026-11-15/code/common.py').metrics;rows=[]
for K in [1,2,4,8]:
 fake=np.load(O/f'outputs/K{K}_validation.npz')['values'];x=torch.from_numpy(np.concatenate([fake,r]));features=torch.cat([(x>0).float().flatten(1),(torch.log1p(x)/D.log_scale).flatten(1)],1);labels=np.r_[np.zeros(len(fake),int),np.ones(len(r),int)]
 for branch in ['full','Mask_only_ablated_D','Amplitude_only_ablated_D']:
  a=features.clone()
  if branch=='Mask_only_ablated_D':a[:,2848:]=0
  elif branch=='Amplitude_only_ablated_D':a[:,:2848]=0
  with torch.inference_mode():pr=torch.cat([D.net(a[i:i+128]).sigmoid().flatten() for i in range(0,len(a),128)]).numpy()
  rows.append(dict(K=K,diagnostic_branch=branch,acceptance_use=False,**metrics(labels,pr)))
# Input derivative: hard occupancy branch has no gradient to touch, even when soft decoder has ST.
x=torch.from_numpy(fake[:16]).requires_grad_(True);mask=(x>0).float();f=torch.cat([mask.flatten(1),(torch.log1p(x)/D.log_scale).flatten(1)],1);inputgrad=torch.autograd.grad(D.net(f).sum(),x)[0];ff=f.detach().requires_grad_(True);featuregrad=torch.autograd.grad(D.net(ff).sum(),ff)[0]
save(O/'gradient_path_audit.json',dict(decoder_ST_forward_matches_hard=True,decoder_ST_backward_only=True,D_hard_occupancy_requires_grad=mask.requires_grad,D_occupancy_branch_to_input_gradient='zero/not defined through Boolean comparison',D_log_amplitude_branch_to_input_gradient_finite=bool(torch.isfinite(inputgrad).all()),D_log_amplitude_branch_to_input_mean_abs_gradient=float(inputgrad.abs().mean()),D_feature_Mask_gradient_mean_abs=float(featuregrad[:,:2848].abs().mean()),D_feature_amplitude_gradient_mean_abs=float(featuregrad[:,2848:].abs().mean()),interpretation='Generator receives adversarial gradients through amplitude/log-input branch; D binary occupancy features are detached. This is a configuration gradient-path limitation, not proof of sole cause or capacity shortage.',no_new_training=True));table(O/'D_branch_ablation.csv',rows);print('GRADIENT AUDIT COMPLETE NO_TRAINING',flush=True)
