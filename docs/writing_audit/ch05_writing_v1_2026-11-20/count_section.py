from pathlib import Path
import importlib.util,json,re
p=Path('scripts/count_thesis_words.py');spec=importlib.util.spec_from_file_location('existing_wordcount',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
O=Path('docs/writing_audit/ch05_writing_v1_2026-11-20').resolve();c=m.Counter(O/'wordcount');s=(O/'section_5_3.tex').read_text();v=c.run(s,'section_5_3');c.run(s,'section_5_3',verbose=True);(O/'wordcount/result.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));print(v)

parts={}
for i,match in enumerate(re.finditer(r"\\begin\{(figure|table)\}.*?\\end\{\1\}",s,re.S),1):
 parts[f"{match[1]}_{i}"]=c.run(match[0],f"float_{i}")
(O/"wordcount/float_counts.json").write_text(json.dumps(parts,ensure_ascii=False,indent=2))
