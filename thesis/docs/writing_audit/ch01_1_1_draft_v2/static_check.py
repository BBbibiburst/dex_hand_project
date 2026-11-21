"""Static source checks only. Does not invoke TeX, BibTeX or a build tool."""
from pathlib import Path
import re,json,hashlib
O=Path(__file__).resolve().parent; R=O.parents[2]; P=R.parent
chapter=R/'body/ch01_introduction.tex';bib=R/'reference.bib'; errors=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def balanced(text):
 text='\n'.join(x for x in text.splitlines() if not x.lstrip().startswith('%'));depth=0
 for i,c in enumerate(text):
  if c not in '{}':continue
  n=0;j=i-1
  while j>=0 and text[j]=='\\':n+=1;j-=1
  if n%2:continue
  depth+=1 if c=='{' else -1
  if depth<0:return False
 return depth==0
before=json.loads((O/'before.json').read_text()); allowed={str(chapter.relative_to(P)),str(bib.relative_to(P))}
# before.json uses paths relative to project root for this task.
changed=[name for name,h in before.items() if sha(P/name)!=h]
for name in changed:
 if name not in allowed:errors.append('unauthorized source change '+name)
tex=chapter.read_text(); b=bib.read_text(); keys=re.findall(r'@\w+\s*\{\s*([^,\s]+)',b)
cited=[k.strip() for batch in re.findall(r'\\cite\{([^}]+)\}',tex) for k in batch.split(',')]
if len(keys)!=len(set(keys)):errors.append('duplicate bibliography key')
if set(cited)!=set(keys):errors.append('unused/missing bibliography entries')
if not balanced(tex) or not balanced(b):errors.append('unbalanced braces')
if re.findall(r'\\section\{([^}]+)\}',tex)!=['研究背景与意义']:errors.append('unexpected section')
if '\\subsection' in tex or '\\begin{figure}' in tex or '\\begin{equation}' in tex:errors.append('unexpected subheading/figure/formula')
original=(O/'ch01_before.tex.txt').read_text();prefix='\n'.join(original.splitlines()[:3]);assert tex.startswith(prefix)
body=tex.split('\\label{sec:research_background}',1)[1].strip();paras=body.split('\n\n')
plain=re.sub(r'\\cite\{[^}]+\}','',body);cjk=len(re.findall(r'[\u3400-\u9fff]',plain));count=len(re.sub(r'\s','',plain))
if len(paras)!=6 or not 1800<=count<=2500:errors.append('paragraph/length outside task scope')
for phrase in ['显著提升','充分证明','全面优于','首先','其次','最后','P52','P53']:
 if phrase in body:errors.append('review phrase '+phrase)
cat={x['key']:x for x in json.loads((R/'docs/writing_audit/ch01_1_1_sources/literature_catalog.json').read_text())};keymap=json.loads((O/'citation_key_map.json').read_text())
# Check year and DOI against approved catalog, independent of the writer's string templates.
for w,key in keymap.items():
 start=b.index('{'+key+',');end=b.find('\n% W',start);entry=b[start:] if end<0 else b[start:end];e=cat[w]
 if re.search(r'year\s*=\s*\{'+e['year']+r'\}',entry) is None:errors.append('year mismatch '+w)
 if e['doi'] and e['doi'] not in entry:errors.append('DOI mismatch '+w)
 if not e['doi'] and re.search(r'\bdoi\s*=',entry):errors.append('unverified DOI '+w)
source_errors=[]
for x in json.loads((R/'docs/writing_audit/ch01_1_1_sources/internal_evidence.json').read_text()):
 if sha(P/x['path'])!=x['sha256']:source_errors.append(x['path'])
errors+=['source integrity '+x for x in source_errors]
result={'task':'TH-CH01-1.1-V2','date':'2026-11-19','stage':'3; draft delivered awaiting user review, not approved/frozen','errors':errors,'paragraphs':len(paras),'CJK_characters':cjk,'characters_including_punctuation_and_Latin_excluding_cites_and_spaces':count,'unique_citations':len(set(cited)),'bibliography_entries':len(keys),'changed_thesis_source_files':changed,'other_protected_files_unchanged':len(before)-len(changed),'compiled':False,'bibliography_output_enabled':False,'source_hashes':{'chapter':sha(chapter),'bibliography':sha(bib)},'limitations':['No TeX/BibTeX compilation or PDF inspection','Main bibliography call remains commented per edit boundary','Reference author rendering awaits authorized compilation']}
(O/'static_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
