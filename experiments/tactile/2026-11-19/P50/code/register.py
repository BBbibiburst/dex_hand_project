from audit_utils import *
import shutil,sys,subprocess
P49=Path('outputs/tactile_p49_round_frozen_adversarial_2026-11-19')
assert not (O/'registration.json').exists()
old=json.loads((P49/'registration.json').read_text())
reg=dict(seed=5001,maximum_rounds=1,K=1,D_epochs=10,G_epochs=20,max_D_steps=660,max_G_steps=1280,batch=64,max_optimization_seconds=1800,loss=old['loss'],quality=old['quality'],optimizer=old['optimizers'],checkpoint_rule='fixed endpoint20;no Validation selection',refiner='776to64 GELU GRU64 Linear712;zero final; log response residual with clamp_min atzero; no thresholdsearch',D='3continuousviews global/spatialindexedPatch/temporal indexedframe;equalBCE',group_split='exact P49fitinner;not new random split;seed5001training only',hard_stop='constant branch predictions, nonfinite gradient/output, group leakage, cumulativebudget;CE gap diagnosticonly',classification='C if FR fails OR D failure OR severe collapse OR no Refiner improvement OR overall quality worse thanF0; B FR passes but qualityfails absent C; A allqualitypasses',no_Test_Holdout=True,no_C0=True,no_production=True)
save(O/'registration.json',reg)
protected=json.loads((P49/'protected_hashes_before.json').read_text())
for p in P49.rglob('*'):
 if p.is_file():protected[str(p)]=sha(p)
save(O/'protected_hashes_before.json',protected)
shutil.copy2(P49/'inner_group_split.npz',O/'inner_group_split.npz');shutil.copy2(P49/'data_inventory.csv',O/'data_inventory.csv')
for role,nr,ns in [('train',5153,4049),('validation',780,905)]:
 r=real(role);s=natural(role);assert len(r['x'])==nr and len(s['x'])==ns
save(O/'environment.json',dict(python=sys.version,git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
print('P50 registered',len(protected),'protectedfiles')
