"""Inspect existing metadata and file presence only. No numeric payload or simulator imports."""
from pathlib import Path
import csv,json,hashlib,collections
O=Path(__file__).resolve().parent;C=Path('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16');G=Path('outputs/tactile_gmm_dual_vae_2026-11-16');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();inputs={}
def txt(p):inputs[str(p)]=sha(p);return p.read_text()
dic=list(csv.DictReader(txt(C/'dictionary_manifest.csv').splitlines()));counts=collections.Counter(r['physical_initial_fingerprint'] for r in dic)
pilot=list(csv.DictReader(txt(C/'inverse_attempts.csv').splitlines()));expanded=[json.loads(l) for l in txt(C/'expansion_attempts.jsonl').splitlines() if l.strip()];attempts=pilot+expanded
assert {r['role'] for r in attempts}=={'train','validation'}
failed=[r for r in attempts if str(r['passed']).lower()=='false']
pres={name:sum((Path(r['path'])/name).is_file() for r in failed) for name in ['episode.npz','condition.json','contact_records.json']}
assert all(v==len(failed) for v in pres.values())
sample=failed[0];samplefiles={str(Path(sample['path'])/n):sha(Path(sample['path'])/n) for n in pres};inputs.update(samplefiles)
checks={'dictionary_rows':len(dic),'unique_first_saved_state_fingerprints':len(counts),'rows_per_fingerprint_distribution':dict(collections.Counter(counts.values())),'logged_candidate_attempts':len(attempts),'failed_statistics_gate_attempts':len(failed),'failed_artifact_presence':pres,'first_failed_record':{k:sample[k] for k in ['role','target_id','candidate_index','path','passed']},'sample_file_hashes':samplefiles,'all_candidate_metadata_roles':['train','validation'],'GMM_copy_hashes':{str(p):sha(p) for p in [G/'gmm_parameters.npz',Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16/gmm_parameters.npz'),C/'gmm_parameters.npz']},'no_payload_decoding':True,'no_training_or_model_execution':True}
assert len(set(checks['GMM_copy_hashes'].values()))==1
inputs.update(checks['GMM_copy_hashes']);(O/'artifact_presence.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n');(O/'extra_inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n')
print('dictionary',len(dic),'fingerprints',len(counts),'failed saved',len(failed),pres)
