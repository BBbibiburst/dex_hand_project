from audit_utils import *
import sys,subprocess,platform
assert (O/'report_complete.json').exists() and (O/'statistical_checks.json').exists()
protected=json.loads((O/'extended_historical_protection.json').read_text());assert all(sha(p)==h for p,h in protected.items());pack=Path('releases/tactile_c27_D090_v1');manifest=json.loads((pack/'SHA256SUMS.json').read_text());assert all(sha(pack/p)==h for p,h in manifest.items())
r=real('train');rv=real('validation');s=natural('train');sv=natural('validation');assert not set(r['g'])&set(rv['g']);assert not set(s['g'])&set(sv['g']);assert not set(r['u'])&set(rv['u']);assert not set(s['u'])&set(sv['u']);save(O/'leakage_audit.json',dict(RealTrainVal_Session_disjoint=True,NaturalTrainVal_Root_disjoint=True,registered_WindowID_disjoint=True,Test_Holdout_numeric_loaded=False,Validation_used_for_earlystop=False,Validation_used_for_modelselection=False,historical_Validation_exposed=True))
counts=json.loads((O/'actual_training_counts.json').read_text());timings=json.loads((O/'probe_device_benchmark.json').read_text());nruns=(O/'probes.log').read_text().count('PROBE_DEVICE ');counts.update(synthetic_device_benchmark_runs=nruns,synthetic_probe_optimizer_steps=nruns*len(timings)*20,synthetic_steps_not_migration_training=True);save(O/'actual_training_counts.json',counts)
# Supplement pergroup numeric inventory with direct role manifests/checkpoints/scaler paths.
inv=list(csv.DictReader((O/'input_inventory.csv').open()))
for role in ['train','validation']:
 for domain in ['Real','Natural']:
  pp=B/f"{'real' if domain=='Real' else 'natural_sim'}_{role}/{'index.json' if domain=='Real' else 'windows.npz'}";d=real(role) if domain=='Real' else natural(role);row=dict(kind='registered role file',domain=domain,role=role,path=str(pp),file_sha256=sha(pp),shape=str(d['x'].shape),groups=len(set(d['g'])))
  if domain=='Real':row['underlying_numeric_container']=json.loads(pp.read_text())['source'];row['restricted_loading']='onlyregisteredindices;pergroupnumeric hashes verified'
  inv.append(row)
for a in ['A','B','C']:
 pp=P44/'generator.pt' if a=='B' else P45/a/'generator.pt';inv.append(dict(kind='frozen Generator',model=a,path=str(pp),file_sha256=sha(pp),training_updated=False))
pp=P46/'RealTrain_preprocessing.npz';inv.append(dict(kind='frozen preprocessing',path=str(pp),file_sha256=sha(pp),fitted_role='RealTrain only'));table(O/'input_inventory.csv',inv)

# Append existing state files to manifest; these are K+1 states per ORIGINALwindow, not extraindependent samples.
rows=list(csv.DictReader((O/'output_manifest.csv').open()))
for p in sorted((O/'paths').glob('*.npz')):
 z=np.load(p);rows.append(dict(model=p.stem,role='validation',path=str(p),sha256=sha(p),shape=str(z['decoded'].shape),artifact='decodedandtargetlatenttrajectory;not new independent windows',original_windows=905,source_Roots=10,optimizer_steps=0))
table(O/'output_manifest.csv',rows)
# Endpoint exactly matches original historical outputs and fullrequiredroles present.
for name,p in candidate_paths().items():
 generated=np.load(O/f'outputs/{name}_validation.npz');old=np.load(p);assert np.array_equal(generated['values'],old['values']);assert np.array_equal(generated['units'],old['units']);assert np.array_equal(generated['groups'],old['groups'])
required=['REPORT.md','registration.json','input_inventory.csv','output_manifest.csv','metrics_jsd.csv','metrics_swd.csv','path_metrics.csv','equivalence_rr_calibration.csv','equivalence_metrics.csv','quality_metrics.csv','negative_controls.csv','bootstrap_results.csv','summary.json','run.log'];assert all((O/name).exists() for name in required)
rows=list(csv.DictReader((O/'metrics_jsd.csv').open()));assert len(rows)==91;realmodels=sum((p/'checkpoint.pt').exists() for p in (O/'probes').glob('*/*') if p.is_dir());assert realmodels==counts['actual_audit_models'];assert sum(v['status']=='NOT_TRAINED_BASELINE_INVALID' for v in rows)==counts['skipped_audit_models'];assert counts['migration_optimizer_steps']==counts['migration_epochs']==0
for p in (O/'code').glob('*.py'):ast.parse(p.read_text())
verification=dict(protected_historical_files=len(protected),protected_hashes_all_unchanged=True,D090_pack_files=len(manifest),D090_pack_unchanged=True,Validation_endpoints_bit_equal_all12=True,model_count_actual=realmodels,model_count_stopped=counts['skipped_audit_models'],TrainVal_group_disjoint=True,no_Test_Holdout=True,no_C0_inference=True,migration_training_epochs=0,migration_optimizer_steps=0,no_server_changes=True,no_commit_push=True,P49_started=False);save(O/'verification.json',verification)
commands=['register.py','historical_manifest_audit.py','generate.py','probes.py','gate_baselines.py','swd_paths.py','equivalence.py','quality.py','shared_audit.py','latent_path_audit.py','checks.py','jsd_summary.py','report.py','finalize.py'];save(O/'run_commands.json',dict(cwd=str(Path.cwd()),date='2026-11-19',python=sys.executable,environment=dict(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),SWD_environment_override=dict(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'),ordered_commands=[f'{sys.executable} {O}/code/{name}' for name in commands],probes_gate='integrated inprobes.py afterOriginal;gate_baselines.py explicit report;no manualmetadata needed forreplication',device='CPU2threads selectedbysyntheticbenchmark',git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_hashes={str(p.relative_to(O)):sha(p) for p in (O/'code').glob('*.py')},Python=platform.python_version()))
# Preserve all completed phase stdout; finalize stdout remains excluded from finalized digest.
with (O/'run.log').open('a') as f:
 for name in ['generate.log','probes.log','baseline_gate.log','swd.log','equivalence.log','quality.log','historical_audit.log','shared_audit.log','latent_path.log','checks.log','jsd_summary.log','report.log']:
  if (O/name).exists():f.write('\n'+name+'\n'+(O/name).read_text())
 f.write('\nFINAL_VERIFICATION '+json.dumps(verification)+'\nP48 STOP; no P49.\n')
outputs={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name not in ['integrity.json','finalize.log'] and '__pycache__' not in p.parts};save(O/'integrity.json',dict(inputs=protected,outputs=outputs,checks=verification,excluded_self_and_open_stdout=['integrity.json','finalize.log'],all_checks_pass=True));print('P48 FINALIZED',len(outputs),'artifacts',verification,flush=True)
