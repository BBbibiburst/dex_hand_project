from audit_utils import *
import ast,torch
for p in (O/'code').glob('*.py'):ast.parse(p.read_text())
protected=json.loads((O/'protected_hashes_before.json').read_text());changed=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h];assert not changed,changed
c=json.loads((O/'complete.json').read_text());assert c['D_steps']==660 and c['G_steps']==1280 and c['rounds']==1 and c['seed']==5001 and c['optimization_seconds']<1800
logs=list(csv.DictReader((O/'training_log.csv').open()));assert len(logs)==1940 and len({(v['phase'],v['epoch'],v['step']) for v in logs})==1940
assert all((O/f'checkpoints/G_epoch_{e:02}.pt').exists() for e in range(1,21))
f=torch.load(O/'checkpoints/F50.pt',map_location='cpu',weights_only=False);d=torch.load(O/'checkpoints/D50.pt',map_location='cpu',weights_only=False);i=torch.load(O/'checkpoints/initial.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,d['D'][k]) for k,v in f['D'].items());assert all(torch.equal(v,i['model'][k]) for k,v in f['model'].items() if k.startswith(('base.real.','base.sim.')))
s=natural('validation');out=np.load(O/'outputs/F50_validation.npz');x=out['values'];assert x.shape==(905,4,712) and x.dtype==np.float32 and np.isfinite(x).all() and (x>=0).all();assert np.array_equal(out['units'],s['u']) and np.array_equal(out['task'],s['task'])
cli=np.load(O/'outputs/F50_cli_validation.npz')['values'];cliok=bool(np.allclose(cli,x,rtol=1e-5,atol=2e-5));assert cliok
save(O/'integrity.json',dict(protected_files=len(protected),changed=changed,D_frozen=True,VAE_frozen=True,AST_valid=True,unique_D_steps=660,unique_G_steps=1280,budget_seconds=c['optimization_seconds'],all20_checkpoints=True,endpoint20=True,output_valid=True,metadata_preserved=True,CLI_inference_allclose=cliok,CLI_mask_difference_entries=int(((cli>0)!=(x>0)).sum()),CLI_mask_disagreement_not_hidden=True,no_Test_Holdout=True,no_C0=True,no_P51=True,no_commit_push=True))
save(O/'run_commands.json',dict(python='/home/zmy/miniconda3/envs/mujoco/bin/python',training_env=dict(PYTHONDONTWRITEBYTECODE=1,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1),evaluation_BLAS_threads=2,script_order=['register.py','train.py','assess.py','audit.py','figures.py','report.py','infer.py --checkpoint checkpoints/F50.pt --input original Natural Validation NPZ --output outputs/F50_cli_validation.npz --device cuda','finalize.py'],training_restarts=0,notes='preoptimization syntax/backend checks corrected; assess postprocessing recovery reused all saved quality metrics; no repeated optimizersteps'))
# Preserve current code and artifact hashes; no historical files are modified.
save(O/'code_hashes.json',{str(p.relative_to(O)):sha(p) for p in (O/'code').glob('*.py')})
files={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name not in ['artifact_hashes.json','finalize.log']};save(O/'artifact_hashes.json',files)
print('P50 FINAL VERIFIED',len(protected),'protectedfiles; classificationC; STOP')
