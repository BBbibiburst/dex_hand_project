"""Read-only source/XML/NPZ audit. No MuJoCo load, compile, step or experiment."""
from pathlib import Path
import ast, json, hashlib, xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=['thesis/AGENTS.md','thesis/THESIS_PLAN_V2.md','thesis/THESIS_WRITING_STANDARD.md','thesis/THESIS_LENGTH_BUDGET.md','thesis/ROBOTICS_EVIDENCE.md','thesis/EXPERIMENT_EVIDENCE.md','thesis/SYMBOLS_AND_TERMS.md','thesis/REFERENCE_THESIS_ANALYSIS.md','thesis/body/ch01_introduction.tex','thesis/reference.bib','thesis/thesis.tex']
paths+=['assets/robots/rm75b/rm75b.xml','assets/robots/rm75b/rm75b.urdf','assets/grippers/dex_hand/dex_hand.xml','configs/robot_profiles/rm75b_dex_hand.json']
paths+=[str(p.relative_to(ROOT)) for folder in ['source/robots','source/control','source/sensors/tactile','source/envs/manipulation'] for p in (ROOT/folder).rglob('*.py')]
paths+=['source/envs/rl_env.py','source/data/production_recorder.py']
paths += [str(p.relative_to(ROOT)) for d in ['assets/robots/rm75b','assets/grippers/dex_hand'] for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix.lower() in ['.stl','.obj','.msh']]
record={}
for s in sorted(set(paths)):
 p=ROOT/s
 if not p.exists():continue
 text=p.read_text() if p.suffix.lower() not in ['.stl','.obj','.msh'] else ''
 functions=[]
 if p.suffix=='.py':
  for n in ast.walk(ast.parse(text)):
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):functions.append({'name':n.name,'line':n.lineno,'end':n.end_lineno})
 record[s]={'sha256':sha(p),'lines':len(text.splitlines()),'functions':functions}
models={}
for name,s in [('arm','assets/robots/rm75b/rm75b.xml'),('hand','assets/grippers/dex_hand/dex_hand.xml')]:
 r=ET.parse(ROOT/s).getroot()
 models[name]={'path':s,'sha256':sha(ROOT/s),'option':r.find('option').attrib,'mesh_assets':[x.attrib for x in r.findall('asset/mesh')], 'body_transforms':[x.attrib for x in r.findall('.//worldbody//body')],'joints':[x.attrib for x in r.findall('.//worldbody//joint')], 'actuators':[dict(type=x.tag,**x.attrib) for x in r.find('actuator')], 'tendons':[{'name':x.get('name'),'terms':[j.attrib for j in x]} for x in r.findall('tendon/fixed')], 'equality_count':len(list(r.find('equality'))) if r.find('equality') is not None else 0,'flex_count':len(r.findall('.//flex')),'skin_geoms':[x.attrib for x in r.findall('.//geom') if x.get('name','').startswith('skin_')], 'explicit_contact_pairs':len(r.findall('contact/pair'))}
patches=[];start=0
for f in range(5):
 for j,rows,cols,kind in [(0,4,8,'segment'),(1,4,8,'segment'),(2,7,8,'fingertip-ellipsoid')]:
  patches.append({'name':f'skin_{f}_{j}_p','rows':rows,'cols':cols,'kind':kind,'start':start,'stop':start+rows*cols});start+=rows*cols
patches.append({'name':'skin_palm_p','rows':7,'cols':16,'kind':'mesh-uv','start':start,'stop':start+112})
cases=['lift_sugar_rgb_00','pick_tomato_rgb_00','stack_ring_rgb_01','push_tomato_rgb_00','nut_retain_lower_rgb_00']
episodes=[]
for case in cases:
 base=ROOT/'outputs/five_task_p32_b_2026-11-03'/case
 p=base/'source_episode/episode.npz'
 result=json.loads((base/'result.json').read_text());manifest=json.loads((base/'source_episode/manifest.json').read_text())
 with np.load(p,allow_pickle=False) as a:
  fields={k:{'shape':list(a[k].shape),'dtype':str(a[k].dtype)} for k in a.files}
  key='sensor_tactile' if 'sensor_tactile' in a.files else 'tactile'
  v=a[key];times=next((a[k] for k in a.files if k=='timestamp'),None)
  ep={'case':case,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'fields':fields,'tactile_key':key,'finite':bool(np.isfinite(v).all()),'nonnegative':bool((v>=0).all()),'nonzero_fraction':float(np.count_nonzero(v)/v.size),'minimum':float(v.min()),'maximum':float(v.max()),'timestamp_deltas':None if times is None else [float(np.diff(times).min()),float(np.diff(times).max())], 'historical_success':result.get('success'),'contract':result.get('contract'),'schema':manifest.get('schema_version'),'independent_root':'UNKNOWN','current_model_compatibility':'UNKNOWN'}
  ep['tactile_vs_sensor_tactile_equal']=bool(np.array_equal(a['tactile'],a['sensor_tactile'])) if 'tactile' in a.files and 'sensor_tactile' in a.files else None
  episodes.append(ep)
 for extra in ['result.json','source_episode/manifest.json']:
  q=base/extra;record[str(q.relative_to(ROOT))]={'sha256':sha(q),'lines':len(q.read_text().splitlines())}
(OUT/'source_inventory.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
(OUT/'model_and_episode_audit.json').write_text(json.dumps({'mode':'offline XML parsing and existing NPZ statistics; no simulation','models':models,'patches':patches,'episodes':episodes},ensure_ascii=False,indent=2)+'\n')
for e in episodes:print(e['case'],e['fields'][e['tactile_key']],e['nonzero_fraction'],e['timestamp_deltas'])
for k,v in models.items():print(k,'joints',len(v['joints']),'actuators',len(v['actuators']),'equality',v['equality_count'],'flex',v['flex_count'])
