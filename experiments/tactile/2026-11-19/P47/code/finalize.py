from support47 import *
import sys,time,subprocess
assert (O/'REPORT.md').exists() and (O/'control_verification_complete.json').exists()
# Recheck permitted numeric group hashes against immutable P46 registration, not only counts.
groups=list(csv.DictReader((P46/'group_inventory.csv').open()));inventory=list(csv.DictReader((O/'input_inventory.csv').open()));grouphashes=[]
for label,domain,role,d in [('RealTrain','Real','train',real('train')),('RealValidation','Real','validation',real('validation')),('NaturalTrain','Natural','train',natural('train')),('NaturalValidation','Natural','validation',natural('validation'))]:
 for g in np.unique(d['g']):
  h=hashlib.sha256(d['x'][d['g']==g].tobytes()).hexdigest();expected=next(row['numeric_sha256'] for row in groups if row['domain']==domain and row['role']==role and row['group']==g);assert h==expected;grouphashes.append(dict(dataset=label,group=g,numeric_sha256=h,verified_against_P46=True))
 source=B/f"{'real' if domain=='Real' else 'natural_sim'}_{role}/{'index.json' if domain=='Real' else 'windows.npz'}";inventory.append(dict(name=label,path=str(source),sha256=sha(source),shape=str(d['x'].shape),role=role,source_groups=len(set(d['g'])),numeric_sha256=hashlib.sha256(d['x'].tobytes()).hexdigest(),identity='selected registerednumeric role only'))
table(O/'input_inventory.csv',inventory);table(O/'numeric_group_hash_verification.csv',grouphashes)
# Verify independent implementation against official PRDC function definitions; no sklearn import/network/training.
namespace=dict(np=np,compute_pairwise_distance=lambda x,y=None:cdist(x,x if y is None else y));tree=ast.parse((O/'references/prdc.py').read_text());funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name!='compute_pairwise_distance'];exec(compile(ast.Module(body=funcs,type_ignores=[]),'official PRDC functions, distance provider scipy','exec'),namespace);rng=np.random.default_rng(4701)
for constant in [False,True]:
 a=rng.normal(size=(25,8));b=np.ones((27,8)) if constant else rng.normal(size=(27,8));aa,bb,ab=cdist(a,a),cdist(b,b),cdist(a,b);ar,br=neighbor_radii(aa),neighbor_radii(bb)
 for j,k in enumerate(KS):
  official=namespace['compute_prdc'](a,b,k);own=prdc(ab,ar[:,j],br[:,j],k);assert all(abs(own[key]-official[key])<1e-12 for key in official)
# Pair invariant metrics, defined input contract, all archived outputs, old C0/weights protected bytewise.
protected=json.loads((O/'protected_hashes_before.json').read_text());assert all(sha(p)==h for p,h in protected.items());pack=Path('releases/tactile_c27_D090_v1');packhash=json.loads((pack/'SHA256SUMS.json').read_text());assert all(sha(pack/p)==h for p,h in packhash.items())
for file in ['prdc_metrics.csv','root_metrics.csv','real_real_prdc.csv']:
 for row in csv.DictReader((O/file).open()):
  for metric in ['precision','recall','coverage']:assert 0<=float(row[metric])<=1
  assert float(row['density'])>=0
for row in csv.DictReader((O/'pair_marginal_invariance.csv').open()):assert float(row['max_abs_error'])<1e-10
for p in (O/'code').glob('*.py'):ast.parse(p.read_text())
required=['REPORT.md','experiment_config.json','input_inventory.csv','alignment_audit.csv','prdc_metrics.csv','real_real_prdc.csv','root_metrics.csv','negative_control_metrics.csv','hsic_metrics.csv','task_stage_retention.csv','condition_semantics_audit.csv','bootstrap_results.csv','run_commands.json'];assert all((O/name).exists() for name in required)
verification=dict(actual_numeric_groups_verified=len(grouphashes),protected_historical_files=len(protected),all_protected_hashes_unchanged=True,D090_pack_files=len(packhash),D090_pack_unchanged=True,official_PRDC_equivalence_pass=True,pair_PRDC_MMD_Energy_invariance_pass=True,no_C0_inference=True,no_generator_inference=True,no_neural_training=True,no_Test_Holdout_numeric_load=True,no_server_access=True,no_commit_push=True,P48_started=False,required_deliverables_present=True)
save(O/'verification.json',verification)
commands=json.loads((O/'run_commands.json').read_text());commands['commands']=['design.py','prdc_eval.py','dependence.py','rr_bootstrap.py','report.py','verify_controls.py','finalize.py'];commands['code_hashes']={str(p.relative_to(O)):sha(p) for p in (O/'code').glob('*.py')};commands['log_files']=['run.log','prdc.log','dependence.log','rr_bootstrap.log','report.log','control_checks.log','finalize.log'];save(O/'run_commands.json',commands)
with (O/'run.log').open('a') as f:
 f.write('\nP47 FINAL EXECUTION RECORD\n')
 for name in ['prdc.log','dependence.log','rr_bootstrap.log','report.log','control_checks.log']:
  f.write('\n'+name+'\n');f.write((O/name).read_text())
 f.write('\nVERIFIED: '+json.dumps(verification)+'\nP47 STOP.\n')
# finalize.log is still open as stdout; exclude it explicitly from finalized output hashes.
outputs={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name not in ['integrity.json','finalize.log'] and '__pycache__' not in p.parts};save(O/'integrity.json',dict(inputs=protected,outputs=outputs,code_hashes=commands['code_hashes'],checks=verification,excluded_self_and_open_stdout=['integrity.json','finalize.log'],all_checks_pass=True))
print('P47 FINALIZED',len(outputs),'artifacts',json.dumps(verification),flush=True)
