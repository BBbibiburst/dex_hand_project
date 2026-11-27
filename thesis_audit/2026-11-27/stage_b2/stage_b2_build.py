from pathlib import Path
import os,subprocess,json
root=Path('/home/zmy/dex_hand_project');r=root/'thesis';state=json.loads(Path(Path('/tmp/stage_b2_state_path').read_text().strip()).read_text());d=Path(state['directory']);out=d/'build';out.mkdir(exist_ok=True)
for n in ['body','back','front']:(out/n).mkdir(exist_ok=True)
env={**os.environ,'PATH':str(r/'.tools/.TinyTeX/bin/x86_64-linux')+':'+os.environ['PATH'],'BIBINPUTS':str(r)+':','BSTINPUTS':str(r)+':'}
with (d/'build.log').open('w') as f:
 for i in range(4):
  f.write('\nXELATEX PASS '+str(i+1)+'\n');f.flush();ret=subprocess.run(['xelatex','-file-line-error','--shell-escape','-synctex=1','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(out),'thesis.tex'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
  if ret.returncode:raise RuntimeError('xelatex failed '+str(ret.returncode))
  if i==0:
   ret=subprocess.run(['bibtex','thesis'],cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT)
   if ret.returncode:raise RuntimeError('bibtex failed '+str(ret.returncode))
import shutil
shutil.copy2(out/'thesis.pdf',d/'thesis_stage_b2_preview.pdf')
print('BUILD COMPLETE',str(d/'thesis_stage_b2_preview.pdf'))
