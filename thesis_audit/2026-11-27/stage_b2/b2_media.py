from pathlib import Path
import json,hashlib,csv,av,numpy as np
from PIL import Image
root=Path('/home/zmy/dex_hand_project'); d=Path(json.loads(Path(Path('/tmp/stage_b2_state_path').read_text().strip()).read_text())['directory']); out=d/'candidate_frames';out.mkdir(exist_ok=True)
base=root/'outputs/five_task_p32_b_2026-11-03'; reports=[];frames=[]
names=['settle','transit','pregrasp','approach','close','hold','lift','verify','transport','descend','release','retreat','task_verify','clear']
for case,indices in [('lift_sugar_rgb_00',[246,311,401]),('nut_retain_lower_rgb_00',[289,428,548])]:
 folder=base/case;v=folder/'agentview.mp4'; manifest=folder/'source_episode/manifest.json';m=json.loads(manifest.read_text());a=np.load(folder/'source_episode/episode.npz',allow_pickle=False);stage=a['stage'];ts=a['timestamp'];c=av.open(str(v));s=c.streams.video[0]
 info={'case':case,'status':'VERIFIED_DIRECT','video':str(v.relative_to(root)),'sha256':hashlib.sha256(v.read_bytes()).hexdigest(),'manifest':str(manifest.relative_to(root)),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'seed':m.get('seed'),'root':'UNKNOWN','codec':s.codec_context.name,'width':s.width,'height':s.height,'fps':str(s.average_rate),'declared_frames':s.frames,'time_base':str(s.time_base),'duration_seconds':float(s.duration*s.time_base) if s.duration else None,'npz_shapes':{k:list(a[k].shape) for k in a.files},'stage_names_source':'manifest and executor stage_codes' if case.startswith('lift') else 'executor stage_codes; historical manifest lacks explicit mapping'}
 pts=[];black=[];kept={}; dims=set()
 for i,f in enumerate(c.decode(s)):
  pts.append(float(f.pts*f.time_base));dims.add((f.width,f.height));arr=f.to_ndarray(format='rgb24');gray=arr.mean(axis=2)
  if gray.mean()<=3 and gray.std()<=2:black.append(i)
  if i in indices:kept[i]=arr
 c.close();info.update(decoded_frames=len(pts),monotonic_pts=bool(np.all(np.diff(pts)>0)),pts_gap_min=float(np.min(np.diff(pts))),pts_gap_max=float(np.max(np.diff(pts))),dimension_consistent=len(dims)==1,black_frame_heuristic_indices=black,decode_error=None,video_vs_npz_timestamp_max_error=float(np.max(np.abs(np.array(pts)-pts[0]-(ts-ts[0])))) )
 assert len(pts)==len(ts) and info['monotonic_pts'] and info['dimension_consistent']
 runs=[]
 for i in range(len(stage)):
  if i==0 or stage[i]!=stage[i-1]:runs.append({'stage_code':int(stage[i]),'stage':names[int(stage[i])],'start':i,'end':i})
  runs[-1]['end']=i
 info['stage_ranges']=runs
 for i,arr in kept.items():
  p=out/f'{case}_frame_{i:04d}_{names[int(stage[i])]}.png';Image.fromarray(arr).save(p)
  frames.append({'case_id':case,'episode_id':case+'/source_episode','root':'UNKNOWN','video_sha256':info['sha256'],'frame_index_zero_based':i,'timestamp_seconds':pts[i],'stage_code':int(stage[i]),'stage':names[int(stage[i])],'manifest':info['manifest'],'manifest_sha256':info['manifest_sha256'],'evidence':'L2 original video + stored stage; not Strict replay screenshot','file':str(p.relative_to(d)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 reports.append(info)
for case in ['nut_original_handle_00','nut_handle_lower_00']:
 r=json.loads((base/case/'result.json').read_text());reports.append({'case':case,'status':'MISSING','historical_result':r,'manifest':str((base/case/'source_episode/manifest.json').relative_to(root)),'manifest_sha256':hashlib.sha256((base/case/'source_episode/manifest.json').read_bytes()).hexdigest(),'root':'UNKNOWN'})
(d/'video_metadata.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2));(d/'candidate_frames.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2))
with (d/'candidate_frames.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
print([(r['case'],r.get('decoded_frames'),r.get('video_vs_npz_timestamp_max_error'),r.get('black_frame_heuristic_indices')) for r in reports])
