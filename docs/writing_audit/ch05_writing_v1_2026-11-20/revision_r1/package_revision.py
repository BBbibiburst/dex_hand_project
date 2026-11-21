from pathlib import Path
import json,re,hashlib,difflib,subprocess,ast
O=Path(__file__).resolve().parent;B=O.parent;T=O/'section_5_3.tex';tex=T.read_text();old=(B/'section_5_3.tex').read_text();base=json.loads((O/'baseline.json').read_text());w=json.loads((O/'wordcount/result.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Reuse only the pure text converters, not the previous packager's top-level actions.
tree=ast.parse((B/'package_section.py').read_text());pure=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['fig','table']],type_ignores=[]);ns={'re':re};exec(compile(pure,'<reading_conversion_only>','exec'),ns)
r=re.sub(r'^%.*\n','',tex,flags=re.M);r=re.sub(r'\\section\{([^}]+)\}',r'## 5.3 \1',r);r=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',ns['fig'],r,flags=re.S);r=re.sub(r'\\begin\{table\}.*?\\end\{table\}',ns['table'],r,flags=re.S);r=r.replace('\\begin{equation}','\n$$').replace('\\end{equation}','$$\n');r=re.sub(r'\\label\{[^}]+\}','',r)
for key,title,prefix in [('fig:ch05_gmm_physical_pipeline','方法流程图','图'),('fig:ch05_gmm_candidate_selection','候选关系图','图'),('tab:ch05_probe_dictionary','字典结构表','表')]:r=r.replace(prefix+'\\ref{'+key+'}',title).replace('\\ref{'+key+'}',title)
(O/'SECTION_5_3_READING_DRAFT.md').write_text('# 第五章5.3 R1阅读稿\n\n状态：定点技术修订，等待审核；未编译、未冻结。原稿保留。以下由修订TeX直接转换。\n\n'+r.replace('~',' ').strip()+'\n')
labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex);assert len(labels)==len(set(labels)) and not set(refs)-set(labels)
stack=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',tex):
 if m[1]=='begin':stack.append(m[2])
 else:assert stack and stack.pop()==m[2]
assert not stack
brace=0
for i,c in enumerate(tex):
 if c in '{}' and (i==0 or tex[i-1]!='\\'):brace+=1 if c=='{' else -1;assert brace>=0
assert brace==0
assert 1800<=w['total']<=2000
assert re.findall(r'\\label\{([^}]+)\}',old)==labels
assert 'N(\\boldsymbol q_{pj}' in tex and '未减去目标质心' in tex and '未除以目标协方差范数' in tex
assert tex.count('\\begin{equation}')==3 and tex.count('\\begin{figure}')==2 and tex.count('\\begin{table}')==1
# Check the metadata facts that were inspected, without executing any historic code.
a=json.loads((O/'artifact_presence.json').read_text());assert a['dictionary_rows']==480 and a['unique_first_saved_state_fingerprints']==160 and a['rows_per_fingerprint_distribution']=={'3':160};assert a['logged_candidate_attempts']==3593 and a['failed_statistics_gate_attempts']==2153
changed=[p for p,h in base['protected'].items() if sha(p)!=h];extra=json.loads((O/'extra_inputs.json').read_text());inputs={**base['inputs'],**extra};bad=[p for p,h in inputs.items() if sha(p)!=h];head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert not changed and not bad and head==base['HEAD']
(O/'changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),tex.splitlines(True),fromfile='section_5_3_V1.tex',tofile='revision_r1/section_5_3.tex')))
(O/'WORDCOUNT_REVIEW.md').write_text(f'''# 5.3 R1实际TeXcount

原稿：1830；修订稿：**{w['total']}**，符合1800–2000目标。增加内容限于坐标原点、混合统计、误差分母/空帧、指纹及提前停止的技术澄清，没有新增实验结果或额外小节。

|项目|实测|
|---|---:|
|正文/流程框/表格文字|{w['text']}|
|标题|{w['headings']}|
|图表题|{w['captions_footnotes']}|
|合计|{w['total']}|
|行内数学（不折算正文）|{w['inline_math']}|
|独立公式（不折算正文）|{w['display_math']}|

使用现有scripts/count_thesis_words.py::Counter、项目原TeXcount中文模式和原texcount_rules.tex，没有更改统计规则。节片段不在正式include树，故调用同一Counter统计独立文件，而非伪造第五章整章结果。

命令：`PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r1/count_section.py`。

原始输出在wordcount/raw/section_5_3.txt；完整解析在wordcount/wordcount_audit.html；分项在wordcount/result.json及float_counts.json。未编译，PDF页数UNKNOWN。
''')
check={'status':'R1定点复核完成，等待用户及ChatGPT审核','wordcount':w,'labels':labels,'missing_refs':[],'duplicate_labels':[],'environments_balanced':True,'braces_balanced':True,'original_responsibility_formula_retained':True,'relative_e_denominator_not_invented':True,'figures':2,'tables':1,'equations':3,'no_compile':True,'frozen':False}
(O/'static_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
integrity={'task':'CH05-5.3-R1','HEAD_before':base['HEAD'],'HEAD_after':head,'protected_before':base['protected'],'changed_protected_files':changed,'read_inputs':inputs,'changed_read_inputs':bad,'original_section_and_reading_unchanged':True,'chapter1_to4_unchanged':True,'main_bib_template_unchanged':True,'production_code_data_weights_unchanged':True,'Test_Holdout_payloads_accessed':False,'new_training':False,'GMM_fit':False,'model_inference':False,'MuJoCo_run':False,'new_pairs':False,'numeric_episode_payloads_decoded':False,'failed_Train_Validation_example_hashed_only':True,'TeXcount':w['total'],'compiled':False,'frozen':False,'commit_push':False,'no_5_4':True,'revised_section_sha256':sha(T)}
(O/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r1/check_artifacts.py','PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r1/count_section.py','PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r1/package_revision.py'],'read_only':'source text, metadata, path existence, Hash; no historic script executed'},ensure_ascii=False,indent=2))
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
print('PASS',w['total'],'protected',len(base['protected']),'inputs',len(inputs),'original draft unchanged')
