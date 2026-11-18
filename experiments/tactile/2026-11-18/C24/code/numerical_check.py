import torch,numpy as np,sys,importlib.util,json,ast,zipfile
from pathlib import Path
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('m','outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/models.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
a=torch.load('outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18/checkpoint.pt',weights_only=False,map_location='cpu');b=torch.load('outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/real_vae/checkpoint.pt',weights_only=False,map_location='cpu');print('enc changed',[k for k in a['model'] if k.startswith('encoder') and not torch.equal(a['model'][k],b['model'][k])]);env=dict(np=np,zipfile=zipfile);tr=ast.parse(Path('outputs/tactile_p43_c10_conditional_latent_diffusion_2026-11-17/code/common10.py').read_text());exec(compile(ast.Module(body=[n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='npz_rows'],type_ignores=[]),'g','exec'),env)
i=json.loads(Path('outputs/tactile_p43_latent_translator_integrated_2026-11-16/real_train/index.json').read_text());x=env['npz_rows'](i['source'],'real',i['indices'][:128]);old=np.load('outputs/tactile_p43_c4_root_cause_audit_2026-11-16/latents/Real_train.npz');print('storedg',old['g'][:4]);print('xg',env['npz_rows'](i['source'],'real_group',i['indices'][:4]));model=M.DomainVAE(a['encoder_scale'],a['positive_scale']);model.load_state_dict(a['model']);model.eval()
for dev in ['cpu','cuda:0']:
 model.to(dev)
 with torch.no_grad():v=model.posterior_mean(torch.from_numpy(x).to(dev)).cpu().numpy()
 print(dev,'maxdiff',np.abs(v-old['mu'][:128]).max(),'mean',np.abs(v-old['mu'][:128]).mean(),'scales',a['encoder_scale'],b['encoder_scale'])
