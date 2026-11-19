"""Archive P44-P51 without modifying original experiment artifacts."""
from pathlib import Path
import hashlib,json,shutil
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
archive=[];inventory=[]
for number in range(44,52):
 paths=list((ROOT/'outputs').glob(f'tactile_p{number}_*'));assert len(paths)==1,paths;source=paths[0];dest=BASE/f'P{number}'
 for p in sorted(source.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts:continue
  rel=p.relative_to(source);size=p.stat().st_size;included=p.suffix.lower() in {'.py','.md','.json','.yaml','.yml','.csv','.log','.txt','.png','.svg'} and size<=2*1024*1024
  if any(part in {'checkpoints','outputs','paths','stopped_audit'} for part in rel.parts):included=False
  # Archived source/scripts and small aggregate evidence; raw arrays and optimization snapshots stay local.
  record=dict(experiment=f'P{number}',source=str(p.relative_to(ROOT)),relative=str(rel),size=size,sha256=sha(p),archived=included)
  if included:
   target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);record['archive_path']=str(target.relative_to(BASE))
  inventory.append(record)
 models={49:['F0.pt','F3.pt','D3.pt'],50:['F50.pt','D50.pt']}.get(number,[])
 for name in models:
  p=source/'checkpoints'/name;target=dest/'checkpoints'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  next(r for r in inventory if r['source']==str(p.relative_to(ROOT))).update(archived=True,archive_path=str(target.relative_to(BASE)),reason='selected baseline or final research checkpoint, includes optimizer state')
 archive.append(dict(experiment=f'P{number}',original=str(source.relative_to(ROOT)),archived=str(dest.relative_to(ROOT))))
(BASE/'source_inventory.json').write_text(json.dumps(dict(experiments=archive,files=inventory),ensure_ascii=False,indent=2)+'\n')
(BASE/'.gitattributes').write_text('*.log -text\n*.pt binary\n*.png binary\n')
print('Archived',sum(r['archived'] for r in inventory),'of',len(inventory),'files; selectedmodels included')
