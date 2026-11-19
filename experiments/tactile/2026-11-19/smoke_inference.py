"""Packaging smoke only: no training or research metric selection."""
from pathlib import Path
import tempfile,subprocess,sys,json,numpy as np
B=Path(__file__).resolve().parent;root=B.parents[2];results=[]
with tempfile.TemporaryDirectory(prefix='p51_archive_smoke_') as temp:
 t=Path(temp);x=np.zeros((3,4,712),dtype=np.float32);np.savez(t/'input.npz',values=x,groups=np.array(['smoke']*3),units=np.array(['a','b','c']),stage=np.array(['smoke']*3))
 for stage,model in [('P49','F0.pt'),('P50','F50.pt')]:
  cmd=[sys.executable,str(B/stage/'code/infer.py'),'--checkpoint',str(B/stage/'checkpoints'/model),'--input',str(t/'input.npz'),'--output',str(t/f'{stage}.npz'),'--device','cpu'];subprocess.run(cmd,check=True,cwd=root,capture_output=True,text=True);d=np.load(t/f'{stage}.npz');y=d['values'];assert y.shape==x.shape and np.isfinite(y).all() and (y>=0).all();assert np.array_equal(d['units'],['a','b','c']);results.append(dict(model=stage+'/'+model,shape=list(y.shape),finite=True,nonnegative=True,metadata_preserved=True))
(B/'packaging_check.json').write_text(json.dumps(dict(purpose='standalone archived CLI packaging smoke using synthetic zeros; not scientific evaluation',training_steps=0,results=results),indent=2)+'\n');print('PASS: two standalone archived inference CLIs')
