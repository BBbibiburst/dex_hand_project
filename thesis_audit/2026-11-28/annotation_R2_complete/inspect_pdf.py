from pathlib import Path
import sys,json,csv
sys.path.insert(0,'thesis/.tools/python')
import pymupdf as fitz
from PIL import Image,ImageDraw
D=Path('thesis_audit/2026-11-28/annotation_R2_complete');o=D/'page_review';o.mkdir(exist_ok=True)
doc=fitz.open(D/'thesis_annotation_R2.pdf');rows=[]; thumbs=[]
for i,p in enumerate(doc):
 txt=p.get_text(); spans=[s for b in p.get_text('dict')['blocks'] if 'lines'in b for l in b['lines'] for s in l['spans']]
 outside=[s['text'] for s in spans if s['bbox'][0]<-1 or s['bbox'][2]>p.rect.width+1 or s['bbox'][1]<-1 or s['bbox'][3]>p.rect.height+1]
 rows.append(dict(pdf_page=i+1,text_chars=len(txt.strip()),images=len(p.get_images()),drawings=len(p.get_drawings()),outside_text=outside))
 pm=p.get_pixmap(matrix=fitz.Matrix(.36,.36),alpha=False);im=Image.frombytes('RGB',[pm.width,pm.height],pm.samples);thumbs.append(im)
 for phrase in ['固定腱系数','策略中间','候选几何','腕部姿态','专家轨迹','装配失败','多模态数据','系统总体架构']:
  if phrase in txt:
   p.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(o/f'page_{i+1:03}.png')
for n in range(0,len(thumbs),16):
 sheet=Image.new('RGB',(900,1320),'#dddddd');dr=ImageDraw.Draw(sheet)
 for j,im in enumerate(thumbs[n:n+16]):
  x=(j%4)*225;y=(j//4)*330;sheet.paste(im,(x,y+18));dr.text((x+8,y+3),str(n+j+1),fill='black')
 sheet.save(o/f'contact_{n//16+1}.jpg')
(D/'pdf_page_checks.json').write_text(json.dumps({'pages':len(doc),'pages_detail':rows},ensure_ascii=False,indent=2))
print('pages',len(doc),'outside',sum(bool(r['outside_text']) for r in rows),'lowtext',[(r['pdf_page'],r['text_chars']) for r in rows if r['text_chars']<40])
