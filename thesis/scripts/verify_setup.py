"""Check generated skeleton/test PDFs; no research data or model access."""
from pathlib import Path
import sys,json,hashlib,gzip,re,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.tools/python'));import pymupdf as fitz
provenance=json.loads((ROOT/'docs/template_provenance.json').read_text())
for name,h in provenance['unmodified_runtime_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
assert not subprocess.check_output(['git','-C',str(ROOT/'upstream/hithesis'),'status','--porcelain'],text=True).strip()
result={'template_runtime_unchanged':True,'upstream_worktree_unchanged':True,'documents':{}}
for name in ['thesis','compile_smoke']:
 pdf=fitz.open(ROOT/f'build/{name}.pdf');text='\n'.join(p.get_text() for p in pdf);normalized=re.sub(r'\s+','',text);(ROOT/f'build/{name}.extracted.txt').write_text(text)
 assert '哈尔滨工业大学' in normalized
 assert sum(len(page.search_for('灵巧手')) for page in pdf)>0
 if name=='thesis':
  for title in ['面向灵巧手触觉信号的仿真到现实迁移方法研究','绪论','灵巧手触觉数据与跨域迁移问题定义','总结与展望']:assert title in normalized,title
  assert '于冬梅' not in normalized and '局部多孔质气体' not in normalized
 else:
  assert '中文复制检索测试：灵巧手触觉，哈尔滨工业大学。' in normalized
  assert 'github.com/hithesis/hithesis' in normalized
  aux=(ROOT/'build/compile_smoke.aux').read_text();bbl=(ROOT/'build/compile_smoke.bbl').read_text();log=(ROOT/'build/compile_smoke.log').read_text()
  for key in ['fig:compile-test','tab:compile-test','eq:compile-test']:assert '\\newlabel{'+key+'}' in aux
  assert '\\bibcite{hithesis-template}' in aux and 'hithesis-template' in bbl
  assert 'Citation `hithesis-template\'' not in log and 'There were undefined references' not in log
 syn=ROOT/f'build/{name}.synctex.gz';assert syn.is_file();s=gzip.open(syn,'rt').read();assert 'Input:' in s
 fonts={f[3] for page in pdf for f in page.get_fonts()};assert any('Fandol' in font for font in fonts)
 cjk=[]
 for page in pdf:
  for f in page.get_fonts():
   if 'Fandol' in f[3]:cjk.append(len(pdf.extract_font(f[0])[3])>0)
 assert cjk and all(cjk)
 result['documents'][name]=dict(pages=len(pdf),Chinese_text_extractable=True,Fandol_embedded=True,SyncTeX_present=True,outline=pdf.get_toc(),fonts=sorted(fonts))
 for pagenum in ([0,min(4,len(pdf)-1)] if name=='thesis' else [len(pdf)-1]):
  pdf[pagenum].get_pixmap(matrix=fitz.Matrix(1,1)).save(ROOT/f'build/{name}_preview_{pagenum+1}.png')
result['bibliography_test']=True;result['figure_table_equation_labels']=True;result['GUI_hotkeys']='configured; not interacted with VSCode GUI'
(ROOT/'docs/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='documents'},ensure_ascii=False,indent=2))
