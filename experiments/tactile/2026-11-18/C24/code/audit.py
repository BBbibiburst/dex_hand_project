import ast,csv,json,sys,zipfile,hashlib,time
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parents[1]; B=Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16');P=Path('outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18');C23=Path('outputs/tactile_c23_real_reconstruction_c0_2026-11-18')
def save(n,d):(O/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False))
def table(n,r):
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in r for k in row)));w.writeheader();w.writerows(r)
env={'np':np,'zipfile':zipfile};tree=ast.parse(Path('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='npz_rows'],type_ignores=[]),'getrows','exec'),env);get=env['npz_rows']
inv={r['unit']:r for r in csv.DictReader(Path('outputs/tactile_p37_full_data_revalidation_2026-11-15/data_inventory.csv').open()) if r['domain']=='real' and r['accepted']=='True' and r['role'] in ('train','validation')};rows=[];sess=[];tot=0;t0=time.time();protected={}
order=(4,5,3,7,8,6,10,11,9,13,14,12,1,2,0,15)
for role in ('train','validation'):
 idx=json.loads((B/f'real_{role}/index.json').read_text());xx=get(idx['source'],'real',idx['indices']);units=get(idx['source'],'real_unit',idx['indices']);groups=get(idx['source'],'real_group',idx['indices']);tarr=np.empty((len(xx),4));idsarr=np.empty((len(xx),4),int)
 bytrial={}
 for i,u in enumerate(units):bytrial.setdefault('/'.join(str(u).split('/')[:2]),[]).append(i)
 for trial,ii in bytrial.items():
  root=Path('outputs/realdata')/trial;sample=root/'samples.jsonl';protected[str(sample)]=hashlib.sha256(sample.read_bytes()).hexdigest();records={}
  for line in sample.open():
   r=json.loads(line);s=str(r.get('stage','')).upper();ts=r.get('tactile_ts',r.get('sample_ts'));sid=r.get('sync_id')
   if ts is not None and sid is not None:records.setdefault(s,[]).append((float(ts),int(sid)))
  with zipfile.ZipFile(root/'tactile_processed.npz') as zz:
   names=set(zz.namelist());available=lambda sid:all(f'proc_{sid:06d}_{p:02d}.npy' in names for p in range(16));cached={};selected={}
   for stage in {str(units[i]).split('/')[2] for i in ii}:
    rr=records[stage];take=[];nxt=rr[0][0]
    for ts,sid in rr:
     if ts+1e-9>=nxt:
      if available(sid):take.append((ts,sid))
      nxt=ts+.05
    starts=np.arange(0,max(0,len(take)-3),4)
    if len(starts)>3:starts=starts[np.linspace(0,len(starts)-1,3).round().astype(int)]
    selected[stage]=[take[int(st):int(st)+4] for st in starts]
   for i in ii:
    unit=str(units[i]);stage,index=unit.split('/')[-2:];chosen=selected[stage][int(index)];ts=np.array([r[0] for r in chosen]);sid=[r[1] for r in chosen];assert np.all(np.diff(ts)>0);assert np.all(np.diff(sid)>0)
    arrays=[]
    for s in sid:
     if s not in cached:
      patches=[]
      for p in order:
       with zz.open(f'proc_{s:06d}_{p:02d}.npy') as f:patches.append(np.lib.format.read_array(f).reshape(-1))
      cached[s]=np.concatenate(patches).astype('float32')
     arrays.append(cached[s])
    assert np.array_equal(np.stack(arrays),xx[i]),f'raw source alignment mismatch {unit}'
    assert np.allclose(ts,json.loads(inv[unit]['timestamps']),rtol=0,atol=1e-8)
    tarr[i]=ts;idsarr[i]=sid;rows.append(dict(role=role,unit=unit,Session=str(groups[i]),timestamps=json.dumps(ts.tolist()),sync_ids=json.dumps(sid),min_dt=float(np.diff(ts).min()),max_dt=float(np.diff(ts).max()),source_numeric_exact=True,consecutive_in_greedy_selected_stream=True))
  if len(rows)%500<10:print('AUDIT',len(rows),round(time.time()-t0,1),flush=True)
 np.savez_compressed(O/f'{role}_time_contract.npz',timestamps=tarr,sync_ids=idsarr,units=units,groups=groups)
 dt=np.diff(tarr,axis=1);tot+=len(xx)
 for g in sorted(set(groups)):
  ix=groups==g;target=xx[ix];genall=np.load(P/f'{role}_outputs.npz')['values'];gen=genall[ix]
  for name,a in [('Real',target),('C13',gen)]:
   d=np.diff(a.astype('float64'),axis=1);u=a[:,:-1].ravel().astype('float64');v=a[:,1:].ravel().astype('float64');pair=(u>0)&(v>0);sess.append(dict(role=role,Session=str(g),source=name,windows=int(ix.sum()),dt_min=float(dt[ix].min()),dt_median=float(np.median(dt[ix])),dt_max=float(dt[ix].max()),mean_abs_frame_delta=float(np.abs(d).mean()),mean_abs_log1p_frame_delta=float(np.abs(np.diff(np.log1p(a.astype('float64')),axis=1)).mean()),both_positive_abs_delta=float(np.abs(v[pair]-u[pair]).mean()),lag1_correlation=float(np.corrcoef(u,v)[0,1])))
 print('ROLE PASS',role,len(xx),'dt',np.quantile(dt,[0,.5,.95,1]),flush=True)
table('time_source_audit.csv',rows);table('Session_temporal_audit.csv',sess);save('source_metadata_hashes.json',protected)
score=list(csv.DictReader((C23/'score_statistics.csv').open()));table('C23_score_audit.csv',score)
save('stage0.json',dict(status='PASS_ORDERED_ADJACENT_DECIMATED_SAMPLES',windows=tot,all_original_numeric_frames_exact=True,within_same_Trial_and_stage=True,not_uniform_20Hz=True,consecutive='consecutive retained complete frames in historical greedy20Hz downsampling; not every hardware sample',loss_interpretation='ordered retained-frame increment consistency, NOT physical derivative per second',C23_delta_implementation='np.diff on axis1 within each registered4-frame window; no cross-window or cross-Trial pairs',C23_statistics_index_error=False,train_epochs_actual_C13=18,C13_best_epoch=3,trainable_parameters=594512,allowed=['decoder.current','decoder.amplitude'],original_loss='per-window positive Gamma2 mixture NLL in original nonnegative processed response units; positive_scale is likelihood parameterization, not target relabelling',new_loss='mean abs discrepancy of adjacent log1p(decoded_response) increments; full3x712, no thresholding',training_can_proceed=True,wall_seconds=time.time()-t0))
print('STAGE0 PASS',tot,flush=True)
