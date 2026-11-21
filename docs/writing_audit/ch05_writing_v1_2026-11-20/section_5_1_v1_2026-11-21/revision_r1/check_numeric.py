"""Train/Validation descriptive audit only; no models, probes or simulations."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parent;P=Path('outputs/tactile_p37_full_data_revalidation_2026-11-15');Q=Path('outputs/tactile_p39_domain_decomposition_2026-11-15');T=Path('outputs/tactile_p42_dual_vae_diagnostics_2026-11-16');src={};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):src[str(p)]=sha(p);return list(csv.DictReader(Path(p).open()))
mask=rows(Q/'mask_statistics.csv');amp=rows(Q/'amplitude_statistics.csv');tm=rows(Q/'temporal_statistics.csv');out=[]
for role in ['train','validation']:
 p=P/f'{role}.npz';src[str(p)]=sha(p)
 with np.load(p,allow_pickle=False) as z:
  for domain,key in [('Real','real'),('raw','sim')]:
   a=z[key];N=len(a);record={'role':role,'source':str(p),'key':key,'shape':list(a.shape),'dtype':str(a.dtype),'source_version':'P37 old development dataset'}
   if role=='train':
    pos=a[a>0];num=int(np.count_nonzero(a>0));den=int(a.size);r=next(r for r in mask if r['role']==role and r['candidate']==domain and r['stage']=='ALL' and r['patch']=='ALL');q=next(r for r in amp if r['role']==role and r['candidate']==domain and r['stage']=='ALL' and r['patch']=='ALL');v=np.quantile(pos,[.5,.95]);assert num==int(q['positive_values']) and abs(num/den-float(r['activation_fraction']))<1e-14;assert np.allclose(v,[float(q['positive_P50']),float(q['positive_P95'])],rtol=0,atol=1e-15)
    record.update(activation_numerator=num,activation_denominator=den,activation_fraction=num/den,positive_pool_count=len(pos),positive_P50=float(v[0]),positive_P95=float(v[1]),quantile_input='a[a>0] pooled over N,4,712; no per-patch average',existing_csv_agrees=True)
   else:
    num=int(np.count_nonzero((a[:,0]>0)!=(a[:,1]>0)));den=N*712;r=next(r for r in tm if r['role']==role and r['candidate']==domain and r['stage']=='ALL' and r['transition']=='0');assert abs(num/den-float(r['change_fraction']))<1e-14
    record.update(mask_change_numerator=num,mask_change_denominator=den,mask_change_fraction=num/den,frame_pair='zero-based0->1; first->second retained frame only',amplitude_abs_change_mean_recorded=r['amplitude_abs_change_mean'],existing_csv_agrees=True)
   out.append(record)
# Timestamp-derived audit table only contains Train/Validation, not mixed Holdout inventory.
time=rows(T/'window_time_contract.csv');assert all(r['role'] in ['train','validation'] for r in time);timedata={}
for role in ['train','validation']:
 rr=[r for r in time if r['role']==role and r['domain']=='real'];assert len(rr)==(5153 if role=='train' else 1408);examples=[r for r in rr if float(r['dt_max'])-float(r['dt_min'])>1e-6];timedata[role]={'windows':len(rr),'nonuniform_dt_windows':len(examples),'example':examples[0],'evidence':'existing timestamp-derived window_time_contract.csv; extraction code mask_diagnostics.py:48–53 reads recorded timestamps; no resampling or signal inference'}
for p in [Q/'code/describe.py',Q/'code/run.py',T/'code/mask_diagnostics.py',T/'p42_report.md',Q/'p39_report.md',Q/'data_counts.csv',Q/'registration.json',Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/model_training_registration.json')]:src[str(p)]=sha(p)
(R/'numeric_check.json').write_text(json.dumps({'records':out,'time_evidence':timedata,'no_Test_Holdout_read':True,'no_model_execution':True,'scope':'lightweight static re-count on existing Train/Validation arrays only, not new experiment'},ensure_ascii=False,indent=2)+'\n');(R/'numeric_source_hashes.json').write_text(json.dumps(src,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'records':out,'timing_counts':{k:v['nonuniform_dt_windows'] for k,v in timedata.items()}},ensure_ascii=False,indent=2))
