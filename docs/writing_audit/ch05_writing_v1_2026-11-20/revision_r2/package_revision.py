from pathlib import Path
import re,json,hashlib,subprocess,difflib,ast
import pymupdf as fitz
O=Path(__file__).resolve().parent;B=O.parent;R1=B/'revision_r1';old=(R1/'section_5_3.tex').read_text();tex=(O/'section_5_3.tex').read_text();base=json.loads((O/'baseline.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Exact same equation, table, figure data and labels; modifications are local explanatory prose.
for env in ['equation','table','figure']:
 pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
 assert re.findall(pat,old,re.S)==re.findall(pat,tex,re.S),env
assert re.findall(r'\\label\{([^}]+)\}',old)==re.findall(r'\\label\{([^}]+)\}',tex)
# Convert the source using the existing pure reading converters only.
tree=ast.parse((B/'package_section.py').read_text());module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fig','table']],type_ignores=[]);ns={'re':re};exec(compile(module,'<pure-reading-functions>','exec'),ns)
r=re.sub(r'^%.*\n','',tex,flags=re.M);r=re.sub(r'\\section\{([^}]+)\}',r'## 5.3 \1',r)
r=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',ns['fig'],r,flags=re.S);r=re.sub(r'\\begin\{table\}.*?\\end\{table\}',ns['table'],r,flags=re.S);r=r.replace('\\begin{equation}','\n$$').replace('\\end{equation}','$$\n');r=re.sub(r'\\label\{[^}]+\}','',r)
for key,title,prefix in [('fig:ch05_gmm_physical_pipeline','方法流程图','图'),('fig:ch05_gmm_candidate_selection','候选关系图','图'),('tab:ch05_probe_dictionary','字典结构表','表')]:r=r.replace(prefix+'\\ref{'+key+'}',title).replace('\\ref{'+key+'}',title)
(O/'SECTION_5_3_READING_DRAFT.md').write_text('# 第五章5.3 R2阅读稿\n\n状态：R2定点修订及临时编译完成，等待用户PDF审核；未冻结。R1及原稿保留。\n\n'+r.replace('~',' ').strip()+'\n')
w=json.loads((O/'wordcount/result.json').read_text());assert 1800<=w['total']<=2000
log=(O/'build/section_5_3_preview.log').read_text();aux=(O/'build/section_5_3_preview.aux').read_text();assert not re.search(r'Overfull|undefined|Undefined control sequence',log)
labels={k:{'number':n,'page':p} for k,n,p in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)}
assert labels['sec:ch05_gmm_pseudopairs']['number']=='5.3'
for prefix,n in [('fig:',2),('tab:',1),('eq:',3)]:assert {v['number'] for k,v in labels.items() if k.startswith(prefix)}=={f'5-{i}' for i in range(1,n+1)}
assert not set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex))-set(labels)
doc=fitz.open(O/'section_5_3_preview.pdf');fonts={};pages=[]
for i,p in enumerate(doc):
 t=p.get_text();assert len(t)>100 and '\ufffd' not in t;pages.append({'page':i+1,'chars':len(t)})
 for f in p.get_fonts():
  if f[0]:fonts[str(f[0])]={'name':f[3],'embedded':bool(doc.extract_font(f[0])[3])}
assert all(f['embedded'] for f in fonts.values())
art=json.loads((O/'artifact_presence.json').read_text());assert art['failed_statistics_gate_attempts']==2153 and all(v==2153 for v in art['failed_artifact_presence'].values())
changed=[p for p,h in base['protected'].items() if sha(p)!=h];extra=json.loads((O/'extra_inputs.json').read_text());bad=[p for p,h in extra.items() if sha(p)!=h];head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert not changed and not bad and head==base['HEAD']
(O/'changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),tex.splitlines(True),fromfile='revision_r1/section_5_3.tex',tofile='revision_r2/section_5_3.tex')))
(O/'WORDCOUNT_REVIEW.md').write_text(f'''# 5.3 R2实际TeXcount

R1：1998；R2：**{w['total']}**，不超过2000。复用项目原TeXcount 3.1.1中文模式、Counter及原规则。正文/流程框/表格文字{w['text']}，标题{w['headings']}，图表题{w['captions_footnotes']}；行内数学{w['inline_math']}、独立公式{w['display_math']}另报、不折算正文。

统计对象是section_5_3.tex，不包含临时预览用的第五章标题。未删减公式、表格/图示内容或指标定义；仅在局部补明审核要求，并压缩同段重复的来源/独立性表述。

命令：`PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r2/count_section.py`。

原始输出、解析HTML和分项见wordcount/。正式主文件include未改，不将本节统计作为第五章整章统计。
''')
(O/'R2_REVIEW.md').write_text('''# 5.3 R2定点修订核验

## 五项任务清单

|项目|状态|依据及处理|
|---|---|---|
|锚点仍受坐标不一致影响|完成|matching.py第6–8行的Real质心加GMM混合均值只进入d评价；expand.py第44–47行责任度仍在固定q求值。正文明确锚点未消除原点差异，不作平移一致逆映射主张|
|早停、argmin、最终准入顺序|完成|expand.py第79–82行：追加候选→score.passed即break→min全部candidates→所选score.passed且不重复才good。正文明确可能找到通过者却在最终选择后拒收|
|失败候选确实保存|完成|inverse_physics.py第77–79行保存Episode/条件/接触记录；expand.py第77–79行保存评分日志。再次只读核查日志范围内2153条统计未通过记录，三类文件均存在。改“也应保留”为已实现事实|
|责任度非物理归属概率|完成|仅为当前统计模型的分解权重，正文明确非物理接触归属概率；不改变实际q公式|
|TeXcount不超2000|完成|实测1999，原始引擎输出保留|

路径根：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/`。所有代码只读，未导入执行。

## 候选逻辑的准确结论

提前停止发生在候选评价之后、最终argmin之前。argmin包含此前未通过门槛的候选，最终门槛作用于其选出者。因此用户指出的“通过者代价较高、未通过者代价较低，最终拒收”在代码逻辑上可能发生。此处没有假定门槛通过候选必然代价最低，也没有把搜索改成仅在可行集合选优。

本轮没有重评分、改变选择或统计该边界情况的实际发生频数，影响为UNKNOWN；不能把逻辑可能性称为已证实的主要失败原因。重复不在score.passed gate内，重复的通过者亦可能导致早停后拒收，原实现保留。

## 失败记录的检查范围

只读inverse_attempts.csv及expansion_attempts.jsonl：pilot124、扩展3469，共3593条Train/Validation记录；其中2153条统计未通过。三类文件均存在，且一个失败实例Hash与R1检查来源绑定。未解码Episode数值数组，不作物理回放。范围不升级为全历史执行总数；R1已记录与C4历史3597口径未逐条对齐的4条差异，仍为UNKNOWN。

## 修订边界

三个公式、两张流程图及字典表与R1逐字一致；混合权重、协方差、s/d/e门槛、空帧与数值下限处理保留。只改责任度解释、锚点局限、最终准入逻辑和失败记录事实，并压缩同段重复叙述。未增加新的混合协方差公式、结果或文献。

原稿和R1全部保留，R2另存。临时编译后等待用户审核，不标冻结，不进入5.4。
''')
(O/'LAYOUT_REVIEW.md').write_text(f'''# 5.3 R2临时编译检查

状态：**编译成功，等待用户PDF审核，未冻结**。

- PDF：section_5_3_preview.pdf，**{len(doc)}页**。仅输入5.3；临时入口设置第五章上下文和section计数2，显示5.3。第一至四章及正式thesis.tex均未输入或改写。
- 使用既有hithesis、Fandol和项目本地XeLaTeX/latexmk，原样加载latexmkrc与latexmk.projectrc。无新cite，本节不添加参考文献页、不改reference.bib；不把未运行BibTeX称为新文献检查。
- 已查看全3页缩略图，并放大检查第2页责任度公式与字典表、第3页流程图及新增说明。
- 两图、一表、三公式均正常显示；临时编号为图5-1/5-2、表5-1、式5-1至5-3，实际aux已核对。后续整章合并时图表/公式编号可能随前两节浮动体改变，引用使用label自动解析。
- 无Overfull、未解析引用、空白页、重叠或孤立标题；公式下标、希腊字母、分式及argmin均显示正常。流程框及字典表未越出版心。
- 中文文字可提取，字体已嵌入；页眉页脚和独立页码正常。第3页章末自然留白，未为压页改变全局格式。
- 保留模板数学字体尺寸替代与未打印glossary等提示，不修改模板。完整控制台日志为build.log，最终引擎日志与SyncTeX在build/。

本轮仅临时预览，不将其标为最终冻结版。没有实机/仿真/训练，没有commit/push，未启动5.4。
''')
check={'wordcount':w,'compiled_pages':len(doc),'labels':labels,'missing_refs':[],'overfull':[],'undefined':[],'underfull':re.findall(r'Underfull[^\n]*',log),'fonts':fonts,'pages':pages,'equations_figures_table_unchanged':True,'frozen':False}
(O/'static_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
integrity={'task':'CH05-5.3-R2','HEAD_before':base['HEAD'],'HEAD_after':head,'protected_before':base['protected'],'changed_protected_files':changed,'checked_inputs':extra,'changed_inputs':bad,'R1_and_original_preserved':True,'main_bib_template_ch1_to4_unchanged':True,'formulas_figures_table_identical_to_R1':True,'new_model_training':False,'GMM_fit':False,'simulator_run':False,'Test_Holdout_access':False,'C0_changed':False,'new_experimental_result':False,'compiled':True,'frozen':False,'next_5_4_started':False,'commit_push':False,'section_sha256':sha(O/'section_5_3.tex'),'pdf_sha256':sha(O/'section_5_3_preview.pdf'),'TeXcount':w['total']}
(O/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r2/check_artifacts.py','PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r2/count_section.py','bash docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r2/build_preview.sh > docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r2/build.log 2>&1','PYTHONPATH=thesis/.tools/python python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r2/package_revision.py'],'build_log':'保留本轮初次及最终成功构建日志；最终引擎日志在build/','PDF_review':'全页缩略图+第2/3页放大查看'},ensure_ascii=False,indent=2))
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
print('PASS R2',w['total'],'pages',len(doc),'protected unchanged',len(base['protected']))
