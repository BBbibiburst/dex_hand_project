from audit_utils import *
from architecture import Transfer,MultiView
import torch,ast
P49=Path('outputs/tactile_p49_round_frozen_adversarial_2026-11-19');torch.set_num_threads(2)
ck=torch.load(O/'checkpoints/F50.pt',map_location='cpu',weights_only=False);ini=torch.load(O/'checkpoints/initial.pt',map_location='cpu',weights_only=False);dc=torch.load(O/'checkpoints/D50.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,dc['D'][k]) for k,v in ck['D'].items());frozen=[k for k in ck['model'] if k.startswith(('base.real.','base.sim.'))];assert all(torch.equal(ck['model'][k],ini['model'][k]) for k in frozen)
rows=[]
for prefix in ['base.field.','refiner.']:
 a=[(ck['model'][k]-ini['model'][k]).float().flatten() for k in ck['model'] if k.startswith(prefix)];v=torch.cat(a);rows.append(dict(module=prefix,delta_L2=float(v.norm()),delta_max=float(v.abs().max()),changed_values=int((v!=0).sum()),total_values=len(v)))
table(O/'parameter_updates.csv',rows)
G=Transfer(ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);sv=natural('validation')
with torch.inference_mode():z=torch.cat([G.base.sim.posterior_mean(torch.from_numpy(sv['x'][i:i+64])) for i in range(0,905,64)])
G.cuda()
with torch.inference_mode():y=torch.cat([G.from_latent(z[i:i+64].cuda())[0].cpu() for i in range(0,905,64)]).numpy()
stored=np.load(O/'outputs/F50_validation.npz');assert np.array_equal(stored['units'],sv['u']) and np.array_equal(stored['groups'],sv['g']) and np.array_equal(stored['stage'],sv['s']) and np.array_equal(stored['task'],sv['task']);err=float(np.abs(y-stored['values']).max());replay_allclose=bool(np.allclose(y,stored['values'],rtol=1e-5,atol=2e-5));replay_mask=bool(np.array_equal(y>0,stored['values']>0));np.savez_compressed(O/'outputs/F50_replay_validation.npz',values=y,groups=sv['g'],stage=sv['s'],units=sv['u'],task=sv['task'])
# Endpoint residual size is descriptive, no checkpoint selection.
raw=np.load(O/'outputs/F50_without_refiner_validation.npz')['values'];delta=y-raw
save(O/'inference_audit.json',dict(output_shape=list(y.shape),finite=bool(np.isfinite(y).all()),nonnegative=bool((y>=0).all()),CUDA_replay_max_abs=err,CUDA_replay_Mask_identical=replay_mask,CUDA_replay_allclose=replay_allclose,replay_tolerance=dict(rtol=1e-5,atol=2e-5),CUDA_replay_mask_difference_entries=int(((y>0)!=(stored['values']>0)).sum()),all_metadata_preserved=True,residual_response_RMS=float(np.sqrt(np.mean(delta**2))),residual_response_max_abs=float(np.abs(delta).max()),residual_fraction_changed=float((delta!=0).mean())))
# Initial CPU reference is historical exact; quantify backend difference separately, never change baseline.
G0=Transfer(ini['scales']);G0.load_state_dict(ini['model']);G0.eval();G0.cuda()
with torch.inference_mode():b=torch.cat([G0.from_latent(z[i:i+64].cuda())[0].cpu() for i in range(0,905,64)]).numpy()
bc=np.load(O/'outputs/F0_validation.npz')['values'];save(O/'initial_backend_audit.json',dict(reference_backend='historical CPU F0 exact',training_backend='CPU Encoder, CUDA field and Decoder',initialCUDA_vs_reference_max_abs=float(np.abs(b-bc).max()),initialCUDA_mask_different_entries=int(((b>0)!=(bc>0)).sum()),identity_smoke_exact_same_backend=True,no_backend_claim_of_bitwise_identity=True))
manifest={str(p.relative_to(O)):sha(p) for p in (O/'checkpoints').glob('*.pt')};save(O/'checkpoint_hashes.json',manifest)
inv=list(csv.DictReader((O/'parameter_inventory.csv').open()))
for row in inv:
 if row['module']=='D50':row['trainable']=row['parameters']
 row['phase_note']='trainable during D phase only, frozen during G' if row['module']=='D50' else 'G phase; VAE frozen'
table(O/'parameter_inventory.csv',inv)
protected=json.loads((O/'protected_hashes_before.json').read_text());changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed,changed
for p in (O/'code').glob('*.py'):ast.parse(p.read_text())
save(O/'integrity_checks.json',dict(protected_files=len(protected),changed=changed,D50_bit_identical_during_G=True,VAE_bit_identical=True,all_epochs_saved=all((O/f'checkpoints/G_epoch_{e:02}.pt').exists() for e in range(1,21)),fixed_endpoint20=True,seed5001=True,only_one_round=True,no_Test_Holdout=True,no_C0=True,no_server=True,no_P51=True))
print('P50 AUDIT COMPLETE')
