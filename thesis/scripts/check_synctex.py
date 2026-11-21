from pathlib import Path
import subprocess,json,re
r=Path(__file__).resolve().parents[1];exe=r/'.tools/.TinyTeX/bin/x86_64-linux/synctex';source=r/'body/ch01_introduction.tex';pdf=r/'build/thesis.pdf'
a=subprocess.run([str(exe),'view','-i',f'2:1:{source}','-o',str(pdf)],capture_output=True,text=True);(r/'logs/synctex_forward.log').write_text(a.stdout+a.stderr);assert a.returncode==0
page=re.search(r'^Page:(.+)$',a.stdout,re.M);x=re.search(r'^x:(.+)$',a.stdout,re.M);y=re.search(r'^y:(.+)$',a.stdout,re.M);assert page and x and y
b=subprocess.run([str(exe),'edit','-o',f'{page[1]}:{x[1]}:{y[1]}:{pdf}'],capture_output=True,text=True);(r/'logs/synctex_reverse.log').write_text(b.stdout+b.stderr);assert b.returncode==0 and 'ch01_introduction.tex' in b.stdout
(r/'docs/synctex_check.json').write_text(json.dumps(dict(forward_search=True,reverse_search=True,source=str(source),page=int(page[1]),GUI_hotkeys_not_manually_tested=True),indent=2)+'\n');print('SyncTeX forward/reverse CLI passed')
