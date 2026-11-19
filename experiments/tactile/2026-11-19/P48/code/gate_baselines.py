from audit_utils import *
assert (O/'generate_complete.json').exists();reg=json.loads((O/'registration.json').read_text());rv=real('validation');sv=natural('validation');out=[]
for source in (O/'probes/Original').glob('*/complete.json'):
 a=json.loads(source.read_text());ident=a['instance'];p=np.load(source.parent/'predictions.npz');logits=p['validation_logits'];lab=p['validation_labels'];loss=np.logaddexp(0,logits)-lab*logits;rloss=loss[:780];sloss=loss[780:];j,j0=bootstrap_scores(rloss,sloss,rloss,sloss,rv['g'],sv['g']);low=float(np.quantile(j0,.025));high=float(np.quantile(j0,.975));ok=bool(a['valid_for_JSD_claim'] and a['Validation']['JSD_proxy']>.01 and low>0)
 out.append(dict(instance=ident,baseline_JSD=a['Validation']['JSD_proxy'],CI_low=low,CI_high=high,TrainVal_mismatch=a['train_validation_mismatch'],converged=a['converged'],baseline_valid=ok))
 if not ok:
  for model in reg['models'][1:]:
   dest=O/'probes'/model/ident
   if (dest/'complete.json').exists():
    existing=json.loads((dest/'complete.json').read_text());assert existing.get('status')=='NOT_TRAINED_BASELINE_INVALID','Do not overwrite real model results'
   else:
    dest.mkdir(exist_ok=True,parents=True);save(dest/'complete.json',dict(model=model,instance=ident,status='NOT_TRAINED_BASELINE_INVALID',epochs=0,optimizer_steps=0,valid_for_JSD_claim=False,reason='P48 requiredbranchstop:Original baseline nonconverged,mismatch ornonpositiveJ0bootstrap;NO invented candidate scores',baseline_source=str(source)))
table(O/'baseline_validity_gate.csv',out);print('BASELINE_GATE',out,flush=True)
