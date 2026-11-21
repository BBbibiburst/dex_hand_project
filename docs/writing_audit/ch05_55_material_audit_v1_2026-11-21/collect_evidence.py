"""Read-only static audit. No project module imports, model construction or forward."""
from pathlib import Path
import json,csv,hashlib,collections,ast
import numpy as np
import torch
D=Path(__file__).resolve().parent;C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');C4=Path('outputs/tactile_p43_c4_root_cause_audit_2026-11-16');seen={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):p=Path(p);seen[str(p)]=sha(p);return p.read_text()
def js(p):return json.loads(read(p))
def rows(p):return list(csv.DictReader(read(p).splitlines()))
def save(n,v):(D/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
# Protect existing thesis, source, historical audit and prior deliverables; never rewrite them.
base=js('docs/writing_audit/ch05_54_material_audit_v1_2026-11-21/baseline.json')['protected'];protected={p:sha(p) for p in base}
for root in [Path('thesis/releases/ch05_5_4_2026-11-21'),Path('docs/writing_audit/ch05_54_material_audit_v1_2026-11-21')]:
 for p in root.rglob('*'):
  if p.is_file():protected[str(p)]=sha(p)
save('baseline.json',{'HEAD':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'protected':protected})
for root in [Path('docs/writing_audit/ch05_material_audit_v1_2026-11-20')]:
 for p in root.glob('*.md'):read(p)
for root in [Path('thesis/releases/ch05_5_3_2026-11-21'),Path('thesis/releases/ch05_5_4_2026-11-21')]:
 for p in root.iterdir():
  if p.suffix in ['.md','.tex','.json']:read(p)
for n in ['REFERENCE_THESIS_ANALYSIS.md','THESIS_WRITING_STANDARD.md','THESIS_PLAN_V2.md','ROBOTICS_EVIDENCE.md','EXPERIMENT_EVIDENCE.md','SYMBOLS_AND_TERMS.md']:read(Path('thesis')/n)
code={}
for p in [C3/'code/models.py',C3/'code/pipeline_cpu.py',C3/'code/inference.py',C3/'code/expand.py',C3/'code/verify_pairs.py',C4/'code/models_audit.py',C4/'code/pairs.py',C4/'code/input_ambiguity.py']:
 s=read(p);tr=ast.parse(s);code[str(p)]=[dict(name=n.name,line=n.lineno,end=n.end_lineno) for n in ast.walk(tr) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
for name in ['tactile_p43_c13_amplitude_context_repair_2026-11-18','tactile_c23_real_reconstruction_c0_2026-11-18','tactile_c24_temporal_amplitude_repair_2026-11-18','tactile_c25_latent_gmm_ot_feasibility_2026-11-18','tactile_c26_shared_frozen_encoder_2026-11-18','tactile_c27_c3_route_improvement_2026-11-18']:
 root=Path('outputs')/name
 for n in ['registration.json','decision.json','config.json']:
  if (root/n).exists():js(root/n)
 for p in (root/'code').glob('*.py'):
  if p.name in ['run.py','train.py','evaluate.py','audit.py','common.py','infer.py']:read(p)
legacy=Path('outputs/tactile_gmm_dual_vae_2026-11-16')
for p in [legacy/'code/train_vae.py',legacy/'code/models.py']:
 if p.exists():read(p)
ckpath=C3/'translator/checkpoint.pt';seen[str(ckpath)]=sha(ckpath);ck=torch.load(ckpath,map_location='cpu',weights_only=True);meta={'path':str(ckpath),'sha256':sha(ckpath),'checkpoint_keys':list(ck),'epoch':ck['epoch'],'optimizer_steps':ck['optimizer_steps'],'parameter_count':sum(x.numel() for x in ck['model'].values()),'state_dict':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'numel':v.numel()} for k,v in ck['model'].items()}}
assert meta['parameter_count']==181824
complete=js(C3/'translator/complete.json');reg=js(C3/'model_training_registration.json');hist=rows(C3/'translator/history.csv');registry=js(C3/'converter_frozen.json');assert registry['files'][str(ckpath.resolve()) if str(ckpath.resolve()) in registry['files'] else str(ckpath)]==meta['sha256']
manifest=rows(C3/'paired_manifest.csv');pairs={}
for role in ['train','validation']:
 rr=[r for r in manifest if r['role']==role];path=C3/f'paired_{role}.npz';seen[str(path)]=sha(path)
 with np.load(path,allow_pickle=False) as z:
  sim=z['sim'];real=z['real'];assert len(rr)==len(sim)==len(real)
  matches=[sha_==hashlib.sha256(a.tobytes()).hexdigest() for r,a in zip(rr,sim) for sha_ in [r['sim_content_sha256']]]
  realmatches=[r['real_content_sha256']==hashlib.sha256(a.tobytes()).hexdigest() for r,a in zip(rr,real)]
  pairs[role]={'shape':list(sim.shape),'dtype':str(sim.dtype),'real_shape':list(real.shape),'manifest_row_hashes_match':all(matches+realmatches),'unique':{k:len({r[k] for r in rr}) for k in ['real_source_window_id','real_session_id','sim_root_fingerprint','sim_content_sha256','real_content_sha256','target_id','actual_sensor_episode']},'max_real_source_reuse':max(collections.Counter(r['real_source_window_id'] for r in rr).values()),'max_placed_state_reuse':max(collections.Counter(r['sim_root_fingerprint'] for r in rr).values()),'stages':dict(collections.Counter(r['stage'] for r in rr))}
train=[r for r in manifest if r['role']=='train'];val=[r for r in manifest if r['role']=='validation'];cross={k:len({r[k] for r in train}&{r[k] for r in val}) for k in pairs['train']['unique']}
err=rows(C4/'translator_paired_error.csv');allrows=[r for r in err if r['group_type']=='ALL'];session_fail=[r for r in err if r['group_type']=='real_groups' and float(r['Huber_improvement_fraction'])<0]
training={'complete':complete,'registration':reg,'first':hist[0],'selected':next(r for r in hist if int(r['epoch'])==ck['epoch']),'last':hist[-1],'min_validation_loss_epoch':min(hist,key=lambda r:float(r['validation_loss']))['epoch'],'minimums':{k:min(hist,key=lambda r:float(r[k])) for k in ['train_Huber','validation_Huber','train_RealDecoder_NLL','validation_RealDecoder_NLL']},'all_history':hist}
save('checkpoint_metadata.json',meta);save('training_evidence.json',training);save('pairing_recount.json',{'roles':pairs,'cross_manifest':cross,'historical_independence':js(C4/'paired_independence_counts.json'),'historical_cross':js(C4/'paired_cross_split.json'),'source_inventory_rows':len(rows(C4/'paired_source_inventory.csv'))})
save('mapping_evidence.json',{'all':allrows,'session_worse_than_mean_reference':session_fail,'historical_ambiguity':rows(C4/'paired_mapping_ambiguity_summary.csv')});save('code_locations.json',code);save('source_hashes.json',seen)
print(json.dumps({'parameters':meta['parameter_count'],'sha256':meta['sha256'],'epoch':ck['epoch'],'checkpoint_steps':ck['optimizer_steps'],'training_steps':complete['optimizer_steps'],'train':pairs['train'],'validation':pairs['validation'],'selected':training['selected'],'last':training['last'],'global_val_min_epoch':training['min_validation_loss_epoch'],'session_reference_losses':len(session_fail)},ensure_ascii=False,indent=2))
