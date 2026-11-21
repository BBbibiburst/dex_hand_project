from pathlib import Path
import re,json,hashlib,difflib,shutil
import pymupdf as fitz
R=Path.cwd();O=R/'docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=R/'thesis/body/ch04_multimodal_data.tex';tex=src.read_text();before=(O/'ch04_before.tex').read_text()
(O/'changes.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),tex.splitlines(True),fromfile='ch04_V1.tex',tofile='thesis/body/ch04_multimodal_data.tex')))
shutil.copyfile(src,O/src.name)
B=O/'build/chapter04_preview_with_references';aux=(B/'chapter04_preview_with_references.aux').read_text();log=(B/'chapter04_preview_with_references.log').read_text();blg=(B/'chapter04_preview_with_references.blg').read_text()
labels={k:{'number':n,'page':p} for k,n,p in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)}
keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',(R/'thesis/reference.bib').read_text()))
used=re.findall(r'\\cite\{([^}]+)\}',tex);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex)
wc=json.loads((O/'wordcount/THESIS_WORD_COUNT.json').read_text());ch=next(x for x in wc['chapters'] if x['number']==4);ch['actual_count']=ch['metrics']['total']
d=fitz.open(O/'chapter04_preview_with_references.pdf');body=fitz.open(O/'chapter04_preview.pdf')
pages=[];font={}
for i,p in enumerate(d):
 text=p.get_text();pages.append({'page':i+1,'characters':len(text),'replacement_character':'\ufffd' in text})
 for f in p.get_fonts():
  if f[0]:font[str(f[0])]={'name':f[3],'embedded':bool(d.extract_font(f[0])[3])}
protected=json.loads((O/'protected_before.json').read_text());changed=[p for p,h in protected.items() if sha(R/p)!=h]
audit=json.loads((R/'docs/writing_audit/ch04_material_audit_v1_2026-11-20/integrity.json').read_text());inputs=audit['audited_input_sha256'];bad=[p for p,h in inputs.items() if sha(Path(p))!=h]
check={'status':'第四章R1临时编译完成，等待用户与ChatGPT PDF审核；未冻结','body_pages':len(body),'with_references_pages':len(d),'texcount':ch,'citation_keys':used,'missing_bib_keys':[k for k in used if k not in keys],'missing_refs':[k for k in refs if k not in labels],'compiled_labels':labels,'overfull':re.findall(r'Overfull[^\n]*',log),'underfull':re.findall(r'Underfull[^\n]*',log),'undefined':re.findall(r'[^\n]*undefined[^\n]*',log),'bibtex_warnings':re.findall(r'Warning--[^\n]*',blg),'fonts':font,'pages':pages,'changed_protected':changed,'changed_audited_inputs':bad,'body_export':'BibTeX解析后从含文献预览提取相同正文页，保留引用编号','frozen':False}
assert not changed and not bad and not check['missing_bib_keys'] and not check['missing_refs'] and not check['overfull'] and not check['undefined']
for prefix,n in [('fig:',5),('tab:',6),('eq:',4)]:
 got=[v['number'] for k,v in labels.items() if k.startswith(prefix)]
 assert set(got)=={f'4-{i}' for i in range(1,n+1)},(prefix,got)
(O/'static_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
(O/'TECHNICAL_REVIEW.md').write_text('''# 第四章R1定点技术复核

状态：临时编译完成，等待PDF审核；不冻结。

## A 时间字段与式4-2

直接核对 `source/data/production_recorder.py`：`_before`（146–187行）在物理推进前从`env.data.time`读取timestamp；`_after`（192–200行）在物理推进后读取next_timestamp。`step`、`step_physics`（202–214行）分别包围物理执行；`OnlineRecorder.append`（232–244行）仅提交任务记录器接受的pending行，未接受的执行不自动成为正式数据行。

因此 t_i 是第i条被接受记录的执行前采样时刻，t_i^+ 是这一步结束时刻，t_{i+1} 是下一条被接受记录的执行前时刻。在连续执行、每步均被接受且没有额外物理推进/重置时，后两者衔接。跳步、未接受记录、异常中断或额外推进不能假设衔接。`validate_online_arrays`第73–76行分别检查相邻timestamp间隔和单行next_timestamp−timestamp；这是两个独立的时间合同检查，并非直接对每一对next_timestamp与后继timestamp作单独断言。require_causal路径另检查命令与qpos/qvel连续性（77–83行）。本轮未重新读取或认证历史全部时间数组。

保留原式4-2，只补充三个符号的定义与连续性前提；不将固定周期等同于缺口可忽略。

## B 队列状态与图4-4

`source/data/production_queue.py`不是Python Enum定义，而是在SQL及逻辑分支中使用真实状态字符串：PENDING、RUNNING、VALIDATING、EXPORTED、DUPLICATE、QUARANTINED。

register第315–348行：要求有效所有权与VALIDATING，外部verification.passed为True；来源或content重复进入DUPLICATE，无重复则插入episodes并进入EXPORTED。fail及租约恢复（203–224、300–312行）根据重试条件回到PENDING或进入QUARANTINED。EXPORTED确实存在，且不等同于success=True；合法失败是否计账仍取决于用途与冻结规则。

因此保留EXPORTED名称，仅将图和文字改为“登记：EXPORTED/DUPLICATE；异常隔离：QUARANTINED”，避免误读为成功登记有三个并列出口。长状态使用mbox保持完整。

## C 外部引用

现有reference.bib有已核验的 `mandlekar2022demonstrations`（CoRL/PMLR164），可支持开头“完成示教不自动等于适合训练”的一般背景，本轮仅复用该条目增加一处引用。它不作为LeRobot字段或本项目统计的依据。

逐项查阅现有30条BibTeX条目，没有LeRobot项目自身条目。该工具的字段、版本和转换合同继续以内部审计与实际产物为依据，不以其他数据集论文替代官方软件来源。建议后续获授权时为实际使用版本补充官方软件引用，记录版本/commit与URL；本轮不检索新题录，不新增或修改reference.bib，不猜作者和DOI。

## 修改边界

仅第四章正文的时间定义、队列出口说明、图内对应标注及一处已有文献引用改变。原六节结构、全部历史数值、质量边界和公式主体均保留。代码、原始数据、第一至三章、主文件、BibTeX、模板及配置Hash复核未变，详见static_checks.json。旧V1审计与阅读稿保留为历史版本，以本目录修订源及PDF为R1依据。
''')
(O/'LAYOUT_REVIEW.md').write_text(f'''# 第四章R1临时编译与排版检查

状态：**等待用户与ChatGPT PDF审核，未冻结**。

## 构建

使用现有hithesis、Fandol与本地XeLaTeX/BibTeX/latexmk；原样加载latexmkrc及latexmk.projectrc，未修改正式thesis.tex。两份临时入口仅输入第四章，chapter预置为3，故显示第4章。

正文预览 **{len(body)}页**；含参考文献 **{len(d)}页**，正文1–11页、参考文献第12页。页码是独立预览页码，不是整篇论文的最终页码。正文版由BibTeX解析后的同一正文页导出，避免省略bibliography引起问号引用。构建日志完整保留了初次正文入口未加文献时的警告与修正后的构建过程；最终两入口引擎日志均无未解析引用。

## PDF检查

已查看全部12页缩略图，并放大检查第3页时间公式、第7页队列状态与式4-4、第9页版本/精度表、第10页Sink图。

- 图4-1至4-5全部出现；流程框不溢出，长英文队列状态完整显示。Sink柱状图清晰，零值标注为0，无伪造非零高度。
- 表4-1至4-6均在版心内。4.5库存表在第8页、版本与Sink对照表在第9页，数值与限制列完整保留。未缩放整表或删减内容。表内长字段存在常规自动折行，正式排版时可再做细节润饰，不影响当前阅读。
- 式4-1至4-4由实际aux确认编号，式4-2在第3页正常显示；无公式越界或断失。引用[1]及所有ref目标正常解析。
- 中文Fandol、英文与数学字体可嵌入并可提取中文文本；未检出替代字符。页眉页脚与页码正常，无空白页或孤立标题。
- 第11页仅为本章小结末段，章末存在较大自然留白；第12页为单条参考文献。未为缩短页数改变模板、字体、页边距或删正文。待整篇论文合并后，参考文献会按正式主文件统一组织。

## 非阻断提示

最终日志有{len(check['underfull'])}项Underfull hbox，无Overfull。模板保留数学字体尺寸替代及未打印glossary提示。BibTeX有{len(check['bibtex_warnings'])}项缺字段警告，具体见static_checks.json；未虚构缺失元数据。

本轮不作正式冻结，不commit/push，不进入第五章。
''')
sections=[{'id':x['id'],'title':x['title'],'actual_count':x['actual_count']} for x in ch.get('sections',[])]
(O/'WORDCOUNT_REVIEW.md').write_text(f'''# 第四章R1实际TeXcount

修改前：7442。修改后：**{ch['actual_count']}**，增加{ch['actual_count']-7442}。增加内容仅为时间定义、连续性边界和队列路径说明，未扩充章节。

使用项目TeXcount 3.1.1中文规则，实际输出见wordcount/及count.log。正文、标题、图表标题与表格文字纳入既定统计；数学公式另计。仍在7200–7700范围。

```json
{json.dumps(sections,ensure_ascii=False,indent=2)}
```

正式主文件旧六章include未改；统计针对V2第四章独立源文件，不将旧骨架章节当成本章。
''')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['bash docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v1/build_preview.sh','./scripts/count_thesis_words.sh --chapter 4 --output-dir docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v1/wordcount','PYTHONPATH=thesis/.tools/python python3 docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v1/package_revision.py'],'compiled':True,'frozen':False},ensure_ascii=False,indent=2))
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted([src]+[x for x in O.rglob('*') if x.is_file() and x.name!='SHA256SUMS'])))
print('OK',len(body),len(d),ch['actual_count'],'protected=',len(protected),'audited_inputs=',len(inputs))
