"""Static source/checkpoint metadata and archived Train/Validation log inspection only.
No project imports, model construction, forward/backward, inference, fitting or simulator.
"""
from pathlib import Path
import json,csv,hashlib,ast
import numpy as np
import torch
O=Path(__file__).resolve().parent;C3=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');C4=Path('outputs/tactile_p43_c4_root_cause_audit_2026-11-16');C13=Path('outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18');C23=Path('outputs/tactile_c23_real_reconstruction_c0_2026-11-18');C24=Path('outputs/tactile_c24_temporal_amplitude_repair_2026-11-18');B=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16');seen={};data={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):p=Path(p);seen[str(p)]=sha(p);return p
def text(p):return record(p).read_text()
def js(p):return json.loads(text(p))
def rows(p):return list(csv.DictReader(text(p).splitlines()))
def dump(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
# Required context and frozen material: read-only.
A=Path('docs/writing_audit/ch05_material_audit_v1_2026-11-20')
for n in ['CH05_MATERIAL_AUDIT.md','CH05_EVIDENCE_MATRIX.md','CH05_C0_PROTOCOL_AND_METRIC_AUDIT.md','CH05_STRUCTURE_AND_WORD_BUDGET.md','CH05_OPEN_ISSUES.md','CH05_GMM_PSEUDOPAIR_AUDIT.md']:text(A/n)
F=Path('thesis/releases/ch05_5_3_2026-11-21')
for n in ['section_5_3.tex','FREEZE_STATUS.json','README.md','SHA256SUMS']:text(F/n)
# Execute none of these modules; preserve an AST line index.
index=[]
code=[C3/'code'/n for n in ['models.py','pipeline_cpu.py','inference.py','common.py']]+[C13/'code/run.py',C23/'code/evaluate.py',C24/'code/train.py',C24/'code/audit.py']
for p in code:
 t=text(p);tree=ast.parse(t);index.append({'path':str(p),'definitions':[{'name':n.name,'line':n.lineno,'end':n.end_lineno} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]})
dump('code_locations.json',index)
for label,root in [('C13',C13),('C23',C23),('C24',C24)]:
 for name in ['registration.json','training_summary.json','parameter_change_audit.json','stage0.json','training_freeze.json','data_contract.json','reconstruction.json','decision.json','decode_equivalence.json']:
  p=root/name
  if p.exists():data[label+'_'+name]=js(p)
for label,root in [('C13',C13),('C24',C24)]:
 h=rows(root/'training_history.csv');data[label+'_history']={'rows':len(h),'first':h[0],'last':h[-1]}
 key='validation_positive_NLL' if label=='C13' else 'validation_NLL';best=min(h,key=lambda x:float(x[key]));data[label+'_history']['best_positive_NLL_row']=best
# Model state metadata only; tensor comparisons do not execute the architecture.
paths={'C3_Real':C3/'real_vae/checkpoint.pt','C3_Sim':C3/'sim_vae/checkpoint.pt','C3_Translator':C3/'translator/checkpoint.pt','C13_Real':C13/'checkpoint.pt','C24_initial':C24/'initial_checkpoint.pt','C24_selected':C24/'checkpoint.pt','C24_endpoint':C24/'endpoint_checkpoint.pt'}
states={};meta={}
for label,p in paths.items():
 ck=torch.load(record(p),weights_only=True,map_location='cpu');s=ck['model'];states[label]=s
 scalars={k:(v.item() if isinstance(v,torch.Tensor) and v.numel()==1 else v) for k,v in ck.items() if k!='model' and not isinstance(v,dict)}
 meta[label]={'path':str(p),'sha256':sha(p),'checkpoint_fields':list(ck),'metadata':scalars,'state_shapes':{k:list(v.shape) for k,v in s.items()},'state_dtypes':{k:str(v.dtype) for k,v in s.items()}}
 if 'encoder.scale' in s:
  params={k:v for k,v in s.items() if k not in ['encoder.scale','decoder.positive_scale']}
  meta[label].update(encoder_scale_buffer=float(s['encoder.scale']),positive_scale_buffer=float(s['decoder.positive_scale']),parameter_count=sum(v.numel() for v in params.values()),encoder_parameters=sum(v.numel() for k,v in params.items() if k.startswith('encoder.')),decoder_parameters=sum(v.numel() for k,v in params.items() if k.startswith('decoder.')),amplitude_trainable_parameters=sum(v.numel() for k,v in params.items() if k.startswith(('decoder.current.','decoder.amplitude.'))))
assert meta['C3_Real']['parameter_count']==meta['C3_Sim']['parameter_count']==1094232
assert meta['C3_Real']['state_shapes']==meta['C3_Sim']['state_shapes']
assert all(meta[label]['metadata']['epoch']==30 for label in ['C3_Real','C3_Sim','C3_Translator'])
changes={}
for a,b in [('C3_Real','C13_Real'),('C13_Real','C24_initial'),('C13_Real','C24_selected')]:
 keys=[k for k,v in states[a].items() if not torch.equal(v,states[b][k])];changes[a+'->'+b]=keys
 if b=='C24_initial':assert not keys
 else:assert all(k.startswith(('decoder.current.','decoder.amplitude.')) for k in keys)
data['checkpoint_changes']=changes
dump('checkpoint_metadata.json',meta)
# Frozen original model registry must match exact files; C0 files not opened or evaluated.
frozen=js(C3/'converter_frozen.json');data['C3_frozen_registry_matches']={p:sha(record(p))==h for p,h in frozen['files'].items()};assert all(data['C3_frozen_registry_matches'].values())
data['C3_training_registration']=js(C3/'model_training_registration.json');data['C3_device']=js(C3/'model_device_registration.json')
for branch in ['real_vae','sim_vae']:
 c=js(C3/branch/'complete.json');h=rows(C3/branch/'history.csv');assert len(h)==c['epochs']==45
 assert int(h[-1]['optimizer_steps'])==c['optimizer_steps'];best_saved=next(r for r in h if int(r['epoch'])==c['best_epoch']);eligible_min=min((r for r in h if int(r['epoch'])>=30),key=lambda r:float(r['selection_loss']));assert int(eligible_min['epoch'])==30
 data[branch]={'complete':c,'history_first':h[0],'history_last':h[-1],'checkpoint_selected_epoch_row':best_saved,'global_min_selection_epoch':min(h,key=lambda r:float(r['selection_loss']))['epoch'],'eligible_min_selection_epoch':eligible_min['epoch'],'columns':list(h[0])}
 coverage=C3/branch/('coverage_exact.npz' if branch=='sim_vae' else 'coverage.npz');record(coverage)
 with np.load(coverage,allow_pickle=False) as z:data[branch]['sample_visit_arrays']={k:{'shape':list(z[k].shape),'total':int(z[k].sum()),'minimum':int(z[k].min()) if z[k].size else None,'maximum':int(z[k].max()) if z[k].size else None} for k in z.files}
data['sim_counter_correction']=js(C3/'sim_vae/coverage_counter_audit.json')
data['amplitude_scales']=rows(C4/'amplitude_scale_audit.csv');data['scale_formula_audit']=js(C4/'scale_formula_audit.json')
data['C3_domain_reconstruction']=rows(C4/'frozen_reconstruction_metrics.csv');data['C13_domain_reconstruction']=rows(C13/'metrics.csv');data['C24_structure']=rows(C24/'Real_C13_C24_structure.csv')
# Role/normalization metadata only; no source 4x712 or Test arrays are opened.
for role in ['train','validation']:
 p=B/f'real_{role}/index.json';ix=js(p);data['real_'+role+'_index_metadata']={'path':str(p),'fields':list(ix),'source':ix.get('source'),'windows':len(ix.get('indices',[])),'role':role}
# Additional source IDs for group counts were already directly checked in the chapter audit.
data['group_contract']=json.loads(text(A/'verified_observations.json'))['paired_independence_recount']
# Show that no optimizer resume was silently inferred from checkpoint names.
data['no_optimizer_state_in_selected_checkpoints']={k:('optimizer' not in v['checkpoint_fields'] and 'optimizer_state_dict' not in v['checkpoint_fields']) for k,v in meta.items()}
dump('verified_facts.json',data);dump('audited_inputs.json',seen)
print('Inputs',len(seen),'parameters',meta['C3_Real']['encoder_parameters'],meta['C3_Real']['decoder_parameters'],'changes', {k:len(v) for k,v in changes.items()})
print('Real coverage',data['real_vae']['sample_visit_arrays']);print('Sim coverage',data['sim_vae']['sample_visit_arrays'])
