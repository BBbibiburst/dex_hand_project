from audit_utils import *
s=natural('validation');w=weights(s['g']);rows=[];reference=[]
for group in ['A','B','C']:
 endpoint8=np.load(O/f'paths/{group}_K8.npz')['latent'][:,-1].astype(float)
 for K in [1,2,4,8]:
  z=np.load(O/f'paths/{group}_K{K}.npz');h=z['latent'].astype(float);y=z['decoded'];d=np.diff(h,axis=1);norm=np.linalg.norm(d.reshape(905,K,-1),axis=-1);end=np.linalg.norm((h[:,-1]-h[:,0]).reshape(905,-1),axis=-1);ratio=norm.sum(1)/np.maximum(end,1e-12);ref=np.linalg.norm((h[:,-1]-endpoint8).reshape(905,-1),axis=1);reference.append(dict(model=group,K=K,latent_mean_PathRatio=float(w@ratio),latent_end_distance_from_K8=float(w@ref),interpretation='same target-coordinate trajectory only;K8 is observedgridreference not exactODEsolution'))
  for j in range(1,K+1):
   row=dict(model=group,K=K,step=j,target_latent_update_mean=float(w@norm[:,j-1]),decoded_Mask_change_fraction=float(w@((y[:,j]>0)!=(y[:,j-1]>0)).mean((1,2))))
   if j>1:
    a=d[:,j].reshape(905,-1) if False else d[:,j-1].reshape(905,-1);b=d[:,j-2].reshape(905,-1);den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);ok=den>1e-12;row['latent_direction_cosine']=float(w[ok]@(np.sum(a*b,axis=1)[ok]/den[ok])/w[ok].sum()) if ok.any() else None
   rows.append(row)
table(O/'latent_path_diagnostics.csv',rows);table(O/'integration_grid_sensitivity.csv',reference);print('LATENT PATH AUDIT COMPLETE',flush=True)
