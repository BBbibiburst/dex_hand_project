"""Read existing files only; no production objects/SQLite connections/model loads."""
from pathlib import Path
import json,hashlib,ast,collections
import numpy as np
import pyarrow.parquet as pq
R=Path.cwd();O=R/'docs/writing_audit/ch04_material_audit_v1_2026-11-20'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected={str(p):sha(p) for p in (R/'thesis').rglob('*') if p.is_file() and '.tools' not in p.parts and (p.suffix in ['.tex','.bib','.cls','.cfg','.sty'] or ('releases' in p.parts and p.suffix=='.pdf'))}
readhash={};datasets=[];samplefiles=[]
def readj(p):
 readhash[str(p.relative_to(R))]=sha(p);return json.loads(p.read_text())
def digest_arrays(arrs):
 h=hashlib.sha256()
 for k,v in arrs:
  h.update(k.encode());h.update(str(v.shape).encode());h.update(str(v.dtype).encode());h.update(np.ascontiguousarray(v).tobytes())
 return h.hexdigest()
for rel in ['outputs/dp_2026-10-11/lerobot_5','outputs/dp_p1_2026-10-11/lerobot_formal30','outputs/dp_p2_2026-10-11/lerobot_single_sugar_fixed']:
 base=R/rel;info=readj(base/'meta/info.json');pieces=[];schema_names=set()
 for p in sorted((base/'data').rglob('*.parquet')):
  f=pq.ParquetFile(p);schema_names.update(f.schema_arrow.names)
  cols=[c for c in ['observation.state','action','observation.tactile','timestamp','frame_index','episode_index','task_index'] if c in f.schema_arrow.names]
  t=f.read(columns=cols);pieces.append({c:t[c].to_pylist() for c in cols});readhash[str(p.relative_to(R))]=sha(p)
 taskfile=base/'meta/tasks.parquet';tasklabels=pq.read_table(taskfile).to_pylist() if taskfile.exists() else [];readhash[str(taskfile.relative_to(R))]=sha(taskfile) if taskfile.exists() else 'MISSING'
 data={k:sum([x[k] for x in pieces],[]) for k in pieces[0]};state=np.asarray(data['observation.state'],dtype=np.float32);action=np.asarray(data['action'],dtype=np.float32);tact=np.asarray(data['observation.tactile'],dtype=np.float32);eid=np.asarray(data['episode_index']);ts=np.asarray(data['timestamp']);fi=np.asarray(data['frame_index'])
 rows=[]
 for e in np.unique(eid):
  mask=eid==e;dt=np.diff(ts[mask]);fidx=fi[mask]
  rows.append({'episode_index':int(e),'frames':int(mask.sum()),'timestamp_dt_min':float(dt.min()) if len(dt) else None,'timestamp_dt_max':float(dt.max()) if len(dt) else None,'nonincreasing_timestamps':int((dt<=0).sum()),'frame_index_contiguous':bool(np.array_equal(fidx,np.arange(len(fidx)))),'numeric_episode_digest':digest_arrays([('state',state[mask]),('action',action[mask]),('tactile',tact[mask])])})
 rec={'path':rel,'format':'LeRobot','declared_episodes':info.get('total_episodes'),'declared_frames':info.get('total_frames'),'actual_episode_indices':len(rows),'actual_frames':len(eid),'state_shape':list(state.shape),'action_shape':list(action.shape),'tactile_shape':list(tact.shape),'state_finite':bool(np.isfinite(state).all()),'action_finite':bool(np.isfinite(action).all()),'tactile_finite':bool(np.isfinite(tact).all()),'tactile_nonnegative':bool((tact>=0).all()),'tactile_nonzero_fraction':float(np.count_nonzero(tact)/tact.size),'actual_task_indices':np.unique(data['task_index']).tolist(),'declared_splits':info.get('splits'),'task_labels':tasklabels,'image_feature':info.get('features',{}).get('observation.images.agentview'),'parquet_columns':sorted(schema_names),'episodes':rows,'independent_roots':'UNKNOWN','RGB_pixels_or_video_decoded':False}
 datasets.append(rec)
 # Keep the fixed-version numeric output to compare to a documented raw source below.
 if 'single_sugar_fixed' in rel:fixed_action=action;fixed_state=state;fixed_tact=tact
# Independent source-to-export check for fixed single-object LeRobot.
actionreport=readj(R/'outputs/dp_p2_2026-10-11/diagnostics/action_chain/action_chain_report.json');m=Path(actionreport['manifest']);md=readj(m);npz=m.parent/md.get('arrays','episode.npz');readhash[str(npz.relative_to(R))]=sha(npz)
with np.load(npz,allow_pickle=False) as a:
 n=len(fixed_action);start=len(a['ctrl'])-n;expected=a['ctrl'][start:,:13];state_command_expected=a['ctrl'][start-1:-1,:13];tact_expected=a['tactile'][start-1:-1]
 fixedcheck={'source_manifest':str(m.relative_to(R)),'frames':n,'source_total_frames':len(a['ctrl']),'inferred_tail_start':start,'action_vs_source_ctrl_max_abs':float(np.max(np.abs(fixed_action-expected))),'state_command_vs_previous_ctrl_max_abs':float(np.max(np.abs(fixed_state[:,-13:]-state_command_expected))),'tactile_vs_previous_source_row_max_abs':float(np.max(np.abs(fixed_tact-tact_expected))),'scope':'numeric tail mapping only; does not verify RGB pixels, all state coordinates or physical replay'}
# Examine five old online-v2 cases and five packaged v4 development episodes.
def examine(base,kind):
 mp=base/'source_episode/manifest.json';md=readj(mp);ap=base/'source_episode'/md.get('arrays','episode.npz');readhash[str(ap.relative_to(R))]=sha(ap)
 with np.load(ap,allow_pickle=False) as a:
  ts=a['timestamp'];key='sensor_tactile';t=a[key];n=len(ts)
  row={'path':str(base.relative_to(R)),'kind':kind,'source_schema':md.get('metadata',{}).get('online_contract',{}).get('schema'),'frames':n,'tactile_shape':list(t.shape),'tactile_dtype':str(t.dtype),'tactile_nonzero_fraction':float(np.count_nonzero(t)/t.size),'tactile_nonnegative':bool((t>=0).all()),'all_numeric_finite':all(np.isfinite(a[k]).all() for k in a.files if np.issubdtype(a[k].dtype,np.number)),'timestamp_dt_range':[float(np.diff(ts).min()),float(np.diff(ts).max())],'next_timestamp_dt_max_error':float(np.max(np.abs(a['next_timestamp']-ts-.05))),'pre_action_command_max_error':float(np.max(np.abs(a['state'][1:,-13:]-a['position_action'][:-1]))),'pre_action_command_mismatch_row_count':int(np.count_nonzero(np.any(np.abs(a['state'][1:,-13:]-a['position_action'][:-1])>1e-6,axis=1))),'physical_qpos_continuity':bool(np.array_equal(a['pre_qpos'][1:],a['qpos'][:-1])) if 'pre_qpos' in a.files else None,'physical_qvel_continuity':bool(np.array_equal(a['pre_qvel'][1:],a['qvel'][:-1])) if 'pre_qvel' in a.files else None,'tactile_fields_identical':bool(np.array_equal(a['tactile'],a['sensor_tactile'])),'numeric_shapes':{k:{'shape':list(a[k].shape),'dtype':str(a[k].dtype)} for k in ['state','position_action','ik_action','sensor_tactile','object_state','timestamp','next_timestamp','pre_integration_state','applied_ctrl64'] if k in a.files},'independent_root':'UNKNOWN','RGB_file_exists':(base/'agentview.mp4').is_file()}
  if (base/'modalities.npz').is_file():
   op=base/'modalities.npz';readhash[str(op.relative_to(R))]=sha(op)
   with np.load(op,allow_pickle=False) as b:
    row['modalities_fields']={k:list(b[k].shape) for k in b.files};row['export_action_matches_position_action']=bool(np.array_equal(b['action'],a['position_action']));row['export_tactile_matches_sensor_tactile']=bool(np.array_equal(b['tactile'],t));row['export_state_matches_pre_state']=bool(np.array_equal(b['state'],a['state']))
  else:row['modalities_missing']=True
 if (base/'episode.json').is_file():
  ep=readj(base/'episode.json');row.update({k:ep.get(k) for k in ['task','asset_hash','lineage_hash','content_hash','admission','canonical_action']});row['registered_file_hashes_match']={k:sha(base/k)==h for k,h in ep.get('files',{}).items() if (base/k).is_file()};row['missing_registered_files']=[k for k in ep.get('files',{}) if not (base/k).is_file()]
  for k in ep.get('files',{}):
   if (base/k).is_file():readhash[str((base/k).relative_to(R))]=sha(base/k)
 samplefiles.append(row)
old=R/'outputs/five_task_p32_b_2026-11-03'
for case in ['lift_sugar_rgb_00','pick_tomato_rgb_00','stack_ring_rgb_01','push_tomato_rgb_00','nut_retain_lower_rgb_00']:examine(old/case,'online_v2_development')
for base in sorted((R/'outputs/five_task_p44_integration_2026-11-05/dataset/episodes').iterdir()):
 if base.is_dir():examine(base,'packaged_v4_development')
# Trace one saved v4 fix comparison, without re-running it.
comparison=readj(R/'outputs/five_task_p38_exact_replay_2026-11-03/comparison.json');vm=(R/Path(comparison['exact_replay']['manifest'])).resolve();vbase=vm.parent.parent;examine(vbase,'v4_sink_precision_case')
# Inventory repetition only; it does not certify cross-group leakage absence.
digests=collections.defaultdict(list)
for d in datasets:
 for e in d['episodes']:digests[e['numeric_episode_digest']].append([d['path'],e['episode_index']])
repeats=[v for v in digests.values() if len(v)>1]
(O/'dataset_checks.json').write_text(json.dumps({'scope':'three complete selected LeRobot numeric datasets; 11 named NPZ source cases; no global production inventory','lerobot':datasets,'fixed_export_numeric_check':fixedcheck,'npz_cases':samplefiles,'exact_numeric_episode_duplicates_across_selected_lerobot':repeats,'no_duplicate_is_not_no_overlap_or_no_leakage':True,'historical_v4_comparison':comparison},ensure_ascii=False,indent=2)+'\n')
# Index only the relevant implementation, without importing classes that may mutate storage.
sources=['source/imitation/observation.py','source/data/lerobot_schema.py','source/data/lerobot_recorder.py','apps/collect_generated_lerobot.py','source/data/production_recorder.py','source/data/production_queue.py','source/data/production_video.py','source/data/production_capture.py','source/data/production_artifacts.py','source/data/production_inventory.py','source/data/production_duration.py','source/data/five_task_dataset.py','source/grasping/executor.py','apps/export_five_task_dataset.py']
code={}
for name in sources:
 p=R/name;readhash[name]=sha(p);code[name]={'sha256':sha(p),'definitions':[{'name':n.name,'line':n.lineno,'end':n.end_lineno} for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]}
(O/'source_index.json').write_text(json.dumps(code,ensure_ascii=False,indent=2)+'\n')
for name in ['docs/progress/2026-10-24_imitation_learning_roadmap_audit.md','docs/progress/2026-11-03_p32_b_online_recorder.md','docs/progress/2026-11-03_p38_exact_replay_contract.md','docs/fast_collection.md']:
 readhash[name]=sha(R/name)
assert all(sha(Path(p))==h for p,h in protected.items())
(O/'integrity.json').write_text(json.dumps({'protected_before':protected,'protected_unchanged':True,'audited_input_sha256':readhash,'all_input_hashes_still_match':all(sha(R/p)==h for p,h in readhash.items()),'no_data_writes':True,'no_sqlite_connection':True,'no_L3':True},ensure_ascii=False,indent=2)+'\n')
print('LeRobot',[(d['path'],d['actual_episode_indices'],d['actual_frames']) for d in datasets]);print('fixed alignment',fixedcheck)
for r in samplefiles:print(r['kind'],r['path'].split('/')[-1],r['frames'],r['pre_action_command_max_error'],r['physical_qpos_continuity'],r.get('export_action_matches_position_action'))
