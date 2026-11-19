"""Reproduce the explicitly approved D1 continuation; never retrain D0/D1."""
from audit_utils import *
import shutil
assert (O/'protocol_amendment_2026-11-19.json').exists(), 'Need recorded explicit userprotocolamendment'
assert (O/'D1_group_review_evidence.json').exists() and not (O/'checkpoints/F1.pt').exists()
e=json.loads((O/'D1_group_review_evidence.json').read_text());assert e['Train_fit_inner_group_disjoint'] and e['D1_BA_AUC_reproduced'];stop=O/'complete.json';s=json.loads(stop.read_text());assert s['F_optimizer_steps']==0 and s['D_optimizer_steps']==1980
save(O/'resume_authorized_review.json',dict(user_approved=True,D1_review_complete=True,D1_retraining=False,CE_hardstop_disabled_by_user=True,thresholds_and_budget_unchanged=True,decision='continuecontrolled G1;oneweakRealSessionretained explicitly,not allgroupgeneralizationcertification',weak_Real_Sessions=e['innerReal_groups_below_half'],restored_F0='checkpoints/F0.pt',restored_D1='checkpoints/D1.pt',optimization_seconds_already_used=s['cumulative_optimization_wall_seconds']))
assert not (O/'complete_initial_gate_stop.json').exists();shutil.move(stop,O/'complete_initial_gate_stop.json');print('EXPLICIT_APPROVED_D1_RESUME_READY',flush=True)
