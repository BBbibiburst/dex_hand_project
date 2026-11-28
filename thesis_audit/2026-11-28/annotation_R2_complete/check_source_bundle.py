from pathlib import Path
import zipfile,os,subprocess,json,sys
sys.path.insert(0,'thesis/.tools/python');import pymupdf
R=Path('/home/zmy/dex_hand_project');D=R/'thesis_audit/2026-11-28/annotation_R2_complete';dest=D/'source_bundle_check';dest.mkdir(exist_ok=True)
with zipfile.ZipFile(D/'thesis_annotation_R2_source.zip') as z:z.extractall(dest)
T=dest/'thesis';out=dest/'build';out.mkdir(exist_ok=True)
for n in ['body','back','front']:(out/n).mkdir(exist_ok=True)
env={**os.environ,'PATH':str(R/'thesis/.tools/.TinyTeX/bin/x86_64-linux')+':'+os.environ['PATH'],'BIBINPUTS':str(T)+':','BSTINPUTS':str(T)+':'}
with (D/'source_bundle_build.log').open('w') as f:
 for i in range(4):
  ret=subprocess.run(['xelatex','-file-line-error','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(out),'thesis.tex'],cwd=T,env=env,stdout=f,stderr=subprocess.STDOUT)
  if ret.returncode:raise RuntimeError('source bundle xelatex failed')
  if i==0:
   ret=subprocess.run(['bibtex','thesis'],cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT)
   if ret.returncode:raise RuntimeError('source bundle bibtex failed')
pages=len(pymupdf.open(out/'thesis.pdf'));full=len(pymupdf.open(D/'thesis_annotation_R2.pdf'))
(D/'source_bundle_check.json').write_text(json.dumps({'compiled_from_extracted_zip':True,'pages':pages,'main_pdf_pages':full,'same_page_count':pages==full,'note':'未要求字节完全相同，PDF时间戳与路径可能不同；源和图稿Hash在包内清单。'},ensure_ascii=False,indent=2))
print('EXTRACTED SOURCE BUILD OK',pages)
