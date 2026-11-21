#!/usr/bin/env python3
"""Read selected historical robotics artifacts; no simulator or training invocation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1]; P=R.parent; O=R/'docs/writing_audit/v2'
def hashfile(p): return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
def record(p,data): records.append({'path':str(p.relative_to(P)),'sha256':hashfile(p),**data})
for dirname in ['dex_hand_top100_v2','pick_place_top100_final','stack_task_valid_v24','push_top100_final','nut_assembly_top100_final']:
 p=P/'outputs'/dirname/'summary.json'; d=json.loads(p.read_text()); rows=d.get('results',[])
 record(p,{'kind':'historical_summary_not_independent_final_acceptance','count':d.get('count'),'selected_count':d.get('selected_count'),'result_rows':len(rows),'statuses':{str(s):sum(x.get('status')==s for x in rows) for s in set(x.get('status') for x in rows)},'cached_rows':sum(bool(x.get('cached')) for x in rows),'variant_rows':sum('variant_index' in x for x in rows),'unique_object_ids':len({x['object_id'] for x in rows if 'object_id' in x})})
root=P/'outputs/five_task_p32_b_2026-11-03'
for name in ['lift_sugar_rgb_00','pick_tomato_rgb_00','stack_ring_rgb_01','push_tomato_rgb_00','nut_retain_lower_rgb_00']:
 p=root/name/'source_episode/episode.npz'
 with np.load(p,allow_pickle=False) as z:
  key='sensor_tactile' if 'sensor_tactile' in z else 'tactile'; x=z[key]
  ts=z['timestamp'] if 'timestamp' in z else None
  record(p,{'kind':'sampled_development_episode','case':name,'tactile_key':key,'tactile_shape':list(x.shape),'tactile_finite':bool(np.isfinite(x).all()),'tactile_nonzero_fraction':float((x>0).mean()),'timestamp_20hz':None if ts is None else bool(np.allclose(np.diff(ts),.05,atol=1e-7)),'fields':{k:list(z[k].shape) for k in z.files}})
for dirname in ['dp_2026-10-11/lerobot_5','dp_p1_2026-10-11/lerobot_formal30']:
 root=P/'outputs'/dirname; info=root/'meta/info.json'; d=json.loads(info.read_text()); files=sorted((root/'data').rglob('*.parquet')); rows=0; ids=set(); shapes=set(); tactile_finite=True; nz=0; elems=0
 for p in files:
  table=pq.read_table(p,columns=['episode_index','observation.tactile']); rows+=table.num_rows; ids.update(table.column('episode_index').to_pylist())
  for v in table.column('observation.tactile').to_pylist():
   x=np.asarray(v); shapes.add(tuple(x.shape)); tactile_finite &= bool(np.isfinite(x).all()); nz+=int((x>0).sum()); elems+=x.size
  record(p,{'kind':'lerobot_actual_parquet','rows':table.num_rows})
 record(info,{'kind':'lerobot_metadata_and_actual_content_check','declared_episodes':d['total_episodes'],'declared_frames':d['total_frames'],'actual_parquet_rows':rows,'actual_episode_indices':len(ids),'parquet_files':len(files),'tactile_shapes':[list(s) for s in shapes],'tactile_finite':tactile_finite,'tactile_nonzero_fraction':nz/elems if elems else None,'features':d['features'],'note':'No proof of episode physical independence, video PTS, causal alignment or successful task acceptance from this read'})
for name in ['source/robots/arms/rm75b.py','source/robots/hands/dex_hand.py','source/sensors/tactile/dex_hand.py','source/sensors/tactile/signal_processing.py','source/grasping/graspqp_adapter.py','source/grasping/dexevolve.py','tools/grasping/batch_grasp_edit.py','source/grasping/executor.py','source/verification/strict_replay.py','source/grasp_pipeline/replay.py','source/data/production_recorder.py','source/data/production_queue.py','source/data/lerobot_schema.py']:
 p=P/name; record(p,{'kind':'selected_code_source','scope':'specific functions reviewed; not entire pipeline independently verified'})
O.mkdir(exist_ok=True,parents=True); (O/'robotics_source_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
for r in records:
 if r['kind']!='selected_code_source': print(r['path'],{k:v for k,v in r.items() if k not in ['fields','features','sha256','path','note']})
