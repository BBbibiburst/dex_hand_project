from pathlib import Path
import re,json,hashlib,sys
sys.path.insert(0,'/home/zmy/dex_hand_project/thesis/.tools/python');import pymupdf as fitz
root=Path('/home/zmy/dex_hand_project');d=root/'thesis_audit/2026-11-27/stage_b2';base=root/'thesis'; seen=set();sources=[]
def walk(p):
 if p in seen:return
 seen.add(p);s=p.read_text();s=re.sub(r'(?<!\\)%.*','',s);sources.append((p,s))
 for x in re.findall(r'\\(?:input|include)\{([^}]+)\}',s):
  q=base/(x if x.endswith('.tex') else x+'.tex')
  if q.exists():walk(q)
walk(base/'thesis.tex');items=[];assets=[];refs=[];labels=[]
for p,s in sources:
 refs+=re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',s);labels+=re.findall(r'\\label\{([^}]+)\}',s)
 for kind,block in re.findall(r'\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}',s,re.S):
  lab=re.findall(r'\\label\{([^}]+)\}',block);cap=re.search(r'\\caption(?:\[[^]]*\])?\{([^\n]+)',block)
  items.append({'file':str(p.relative_to(root)),'line':s[:s.index(block)].count('\n')+1,'kind':kind,'labels':lab,'caption_line':cap.group(1) if cap else 'UNKNOWN','includegraphics':re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',block)})
 for opts,path in re.findall(r'\\includegraphics(?:\[([^]]*)\])?\{([^}]+)\}',s):
  q=base/path
  if not q.exists():
   q=next((z for pref in ['', 'figures/','figures/framework/','figures/network/','figures/experiments/'] for ext in ['', '.pdf','.png','.jpg'] if (z:=base/(pref+path+ext)).exists()),q)
  a={'source':str(p.relative_to(root)),'path':path,'options':opts,'exists':q.exists()}
  if q.exists():
   a['sha256']=hashlib.sha256(q.read_bytes()).hexdigest();a['forced_width_and_height']='width=' in opts and 'height=' in opts
   if q.suffix=='.pdf':
    doc=fitz.open(q);a['pages']=len(doc);a['page_rect']=list(doc[0].rect);a['embedded_images']=[{'width':i[2],'height':i[3]} for i in doc[0].get_images()];a['vector_drawing_paths']=len(doc[0].get_drawings());doc.close()
  assets.append(a)
for i in items:i['body_reference_count']=sum(refs.count(l) for l in i['labels'])
data={'sources':[str(p.relative_to(root)) for p,s in sources],'figures':sum(i['kind'].startswith('figure') for i in items),'tables':sum(i['kind'].startswith('table') for i in items),'assets':assets,'items':items,'duplicate_labels':sorted({l for l in labels if labels.count(l)>1}),'missing_refs':sorted(set(refs)-set(labels)),'equations':sum(len(re.findall(r'\\begin\{equation\*?\}',s)) for p,s in sources)}
(d/'figure_inventory.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print({k:v for k,v in data.items() if k in ['figures','tables','equations','duplicate_labels','missing_refs']});print('assets',len(assets))
