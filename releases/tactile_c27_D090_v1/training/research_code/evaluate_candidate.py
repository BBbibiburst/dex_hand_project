from common import *
f=json.loads((O/'candidate/frozen.json').read_text());assert sha(O/'candidate/checkpoint.pt')==f['checkpoint_sha256'];assert sha(O/'candidate/translated_validation.npz')==f['output_sha256'];z=np.load(O/'candidate/translated_validation.npz');evaluate(O/'candidate',z['values'],real('validation'),natural('validation'));print('CANDIDATE EVALUATED NO_NEW_TRAINING',flush=True)
