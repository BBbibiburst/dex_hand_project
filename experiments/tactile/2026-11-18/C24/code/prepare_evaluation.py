from pathlib import Path
O=Path(__file__).resolve().parents[1]
s=Path('outputs/tactile_c23_real_reconstruction_c0_2026-11-18/code/evaluate.py').read_text()
s=s.replace('C23 only: existing Real reconstruction','C24 only: once frozen selected Real reconstruction')
s=s.replace("assert not (O/'registration.json').exists()","assert not (O/'evaluation_registration.json').exists()")
s=s.replace("save('registration.json',dict(scope=", "save('evaluation_registration.json',dict(scope=")
s=s.replace("inputs=[str(C13/'validation_outputs.npz')", "inputs=[str(O/'validation_outputs.npz')")
s=s.replace("candidate='existing C13 frozen Encoder/Decoder free reconstruction; no new converter inference'", "candidate='unique C24 selected frozen Encoder/Decoder reconstruction; generated before any C0 evaluation'")
s=s.replace("new_model_or_probe_training=False", "new_probe_training=False")
s=s.replace("save('protected_hashes_before.json',protected)", "save('evaluation_protected_hashes_before.json',protected); frozen24=json.loads((O/'training_freeze.json').read_text()); assert sha(O/'checkpoint.pt')==frozen24['checkpoint_sha256']; assert sha(O/'validation_outputs.npz')==frozen24['outputs_sha256']")
s=s.replace("z=np.load(C13/'validation_outputs.npz')", "z=np.load(O/'validation_outputs.npz')")
s=s.replace("sha(C13/'checkpoint.pt'),generated_values_sha256=sha(C13/'validation_outputs.npz')", "sha(O/'checkpoint.pt'),generated_values_sha256=sha(O/'validation_outputs.npz')")
start=s.index(' # Historical scalar results only,')
end=s.index(" print('C23 COMPLETE'",start)
s=s[:start]+''' # Comparison only against existing C23 scalar metrics.
 old={r['instance']:r for r in csv.DictReader(Path('outputs/tactile_c23_real_reconstruction_c0_2026-11-18/metrics.csv').open())};comparisons=[]
 for r in rows:
  p=old[r['instance']];assert abs(float(p['recall_real'])-r['recall_real'])<1e-12
  cr=dict(instance=r['instance'])
  for label,rr in [('C23',p),('C24',r)]:
   for f in ['balanced_accuracy','roc_auc','domain_confusion','passed_joint']:cr[label+'_'+f]=rr[f]
  cr['AUC_distance_to_half_change']=abs(r['roc_auc']-.5)-abs(float(p['roc_auc'])-.5);comparisons.append(cr)
 table('C23_C24_comparison.csv',comparisons);budget();assert all(sha(p)==h for p,h in protected.items());assert sha(O/'checkpoint.pt')==frozen24['checkpoint_sha256'];save('evaluation_integrity.json',dict(all_protected_unchanged=True,files=protected,C0_not_refit=True,scaler_not_refit=True,Test_Holdout_not_accessed=True,Natural_tactile_files_not_accessed=True,selected_C24_model_frozen_before_C0=True))
 save('evaluation_complete.json',dict(joint_pass_count=sum(r['passed_joint'] for r in rows),instances=11,all_11_joint_passed=all(r['passed_joint'] for r in rows),scope='REAL_RECONSTRUCTION_READOUT_DIAGNOSTIC_ONLY_NOT_SIM_PRIMARY',BA_pass_count=sum(r['BA_pass'] for r in rows),AUC_pass_count=sum(r['AUC_pass'] for r in rows),D_pass_count=sum(r['D_pass'] for r in rows),wall_seconds=time.time()-start,new_Probe_fits=0,stopped_after_C24=True))
''' +s[end:]
s=s.replace("print('C23 COMPLETE'", "print('C24 EVALUATION COMPLETE'")
(O/'code/evaluate.py').write_text(s)
