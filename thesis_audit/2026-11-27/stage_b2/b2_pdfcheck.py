from pathlib import Path
import sys,json,re
sys.path.insert(0,'/home/zmy/dex_hand_project/thesis/.tools/python');import pymupdf as f
r=Path('/home/zmy/dex_hand_project/thesis_audit/2026-11-27');d=r/'stage_b2';a=f.open(r/'stage_b1/thesis_stage_b1_preview.pdf');b=f.open(d/'thesis_stage_b2_preview.pdf');out={'pages':len(b),'b1_pages':len(a),'changed_text_pages':[i+1 for i in range(min(len(a),len(b))) if a[i].get_text()!=b[i].get_text()],'blank_pages':[i+1 for i,p in enumerate(b) if not p.get_text().strip()],'page_text_bounds_outside':[]}
for i,p in enumerate(b):
 for bl in p.get_text('blocks'):
  if bl[0]<-1 or bl[1]<-1 or bl[2]>p.rect.width+1 or bl[3]>p.rect.height+1:out['page_text_bounds_outside'].append(i+1)
log=(d/'build/thesis.log').read_text(errors='replace');out['warnings']=[x for x in log.splitlines() if any(k in x for k in ['Warning:','Overfull','undefined','multiply defined'])];out['bib_warnings']=[x for x in (d/'build/thesis.blg').read_text(errors='replace').splitlines() if 'Warning' in x]
(d/'pdf_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));render=d/'page_checks';render.mkdir(exist_ok=True)
for n in [17,29,43,55,66,70,78,80,85,86]:b[n-1].get_pixmap(matrix=f.Matrix(1.1,1.1)).save(render/f'page_{n:03d}.png')
# ten-page overview for visual inspection, preserving page aspect
from PIL import Image,ImageDraw
thumbs=[]
for n in [17,29,43,55,66,70,78,80,85,86]:
 im=Image.open(render/f'page_{n:03d}.png');im.thumbnail((300,425));thumbs.append((n,im.copy()))
canvas=Image.new('RGB',(1500,900),'#dddddd');dr=ImageDraw.Draw(canvas)
for j,(n,im) in enumerate(thumbs):x=(j%5)*300;y=(j//5)*450;canvas.paste(im,(x,y+20));dr.text((x+5,y+2),str(n),fill='black')
canvas.save(render/'overview.png');print(json.dumps(out,ensure_ascii=False))
