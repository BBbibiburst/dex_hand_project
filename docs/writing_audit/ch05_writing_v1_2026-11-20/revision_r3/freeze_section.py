from pathlib import Path
import json,re,hashlib,difflib,shutil,subprocess,ast
import pymupdf as fitz
O=Path(__file__).resolve().parent;B=O.parent;R2=B/'revision_r2';F=Path('thesis/releases/ch05_5_3_2026-11-21');assert not F.exists(),'Freeze destination already exists; do not overwrite'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();base=json.loads((O/'baseline.json').read_text());tex=(O/'section_5_3.tex').read_text();old=(R2/'section_5_3.tex').read_text()
# Prove that the text, numbers and equations remain precisely R2's reviewed content.
normalized=tex.replace(' \\par\\vspace*{2.5mm}\n','').replace('且\\mbox{分量数}大于1时','且分量数大于1时')
normalized=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',lambda m:m[0].replace('\\\\[2pt]','\\\\[4pt]'),normalized,flags=re.S)
assert normalized==old
assert tex.count('\\par\\vspace*{2.5mm}')==2
w=json.loads((O/'wordcount/result.json').read_text());assert w['total']==1999
log=(O/'build/section_5_3_preview.log').read_text();aux=(O/'build/section_5_3_preview.aux').read_text();assert not re.search(r'Overfull|undefined|Undefined control sequence',log)
labels={k:{'number':n,'page':p} for k,n,p in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)}
assert labels['sec:ch05_gmm_pseudopairs']['number']=='5.3'
assert labels['eq:ch05_pair_selection']['page']=='2'
assert labels['eq:ch05_gmm_prior']['page']=='1' and labels['eq:ch05_gmm_responsibility']['page']=='2'
doc=fitz.open(O/'section_5_3_preview.pdf');assert len(doc)==3
texts=[p.get_text() for p in doc]
assert '分量数' in texts[1] and not texts[1].splitlines()[1].startswith('量数')
assert '且' in texts[0].splitlines()[-2]
fonts={}
for p in doc:
 assert '\ufffd' not in p.get_text()
 for f in p.get_fonts():
  if f[0]:fonts[str(f[0])]={'name':f[3],'embedded':bool(doc.extract_font(f[0])[3])}
assert all(x['embedded'] for x in fonts.values())
changed=[p for p,h in base['protected'].items() if sha(p)!=h];assert not changed;head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert head==base['HEAD']
# Reading text is content-identical to reviewed R2; change only the external version-status preface.
r=(R2/'SECTION_5_3_READING_DRAFT.md').read_text();r=r.replace('# 第五章5.3 R2阅读稿','# 第五章5.3 R3冻结初稿阅读稿',1).replace('状态：R2定点修订及临时编译完成，等待用户PDF审核；未冻结。R1及原稿保留。','状态：R2内容经用户审核通过；R3按本轮授权完成排版复核并冻结初稿。原稿、R1、R2保留。',1)
(O/'SECTION_5_3_READING_DRAFT.md').write_text(r)
(O/'changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),tex.splitlines(True),fromfile='revision_r2/section_5_3.tex',tofile='revision_r3/section_5_3.tex')))
(O/'WORDCOUNT_REVIEW.md').write_text(f'''# 5.3 R3最终TeXcount

R2：1999；R3：**1999**。正文/流程框/表格文字{w['text']}，节标题{w['headings']}，图表题{w['captions_footnotes']}；行内数学{w['inline_math']}及独立公式{w['display_math']}另报。仍使用项目原Counter、TeXcount中文模式及原规则。

正文及所有技术数据与R2逐字一致，只加入局部框内底部2.5mm留白、将框内额外行距4pt调为2pt，以及跨页位置的mbox。没有改全局字号、行距、页边距或统计规则。

命令：`PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r3/count_section.py`。原始输出及解析HTML在wordcount/。统计不包含临时第五章标题，不是第五章整章结果。
''')
(O/'LAYOUT_REVIEW.md').write_text('''# 5.3 R3排版检查与冻结依据

日期：2026-11-21。R2内容已获用户逐页审核通过；用户本轮要求完成两项R3排版调整、编译检查并在确认历史版本保留后冻结。**R3属于代理完成的最终排版复核，不伪称用户已经另行逐页审核了R3 PDF。**

## 两项调整

1. 两张流程图最后一行之后加入par与vspace*{2.5mm}，统一增加底部留白。为避免单纯增高图框把第二页末公式推到第三页，仅将这两个图框中的额外换行间距由4pt调为2pt；字号及全局段落行距不变，图的内容不变。
2. 对“且分量数大于1时”中的“分量数”使用mbox，避免词内断行及跨页拆分。原稿内容未改。

## 三页结果

- 第1页：第5章标题及5.3标题正常，图5-1底部有足够留白；式5-1正常。页末止于“且”，不拆开“分量数”。
- 第2页：开头完整显示“分量数大于1时”；式5-2、表5-1清楚，式5-3仍在本页下方，未被挤到下一页。
- 第3页：图5-2同样增加底部留白，指标解释和局限完整。末尾自然留白保留，没有为压页删减正文。

已实际查看三页渲染图。最终日志无Overfull、未解析引用或编译错误；保留模板数学字体尺寸替代、未打印glossary等非阻断提示。字体均嵌入、中文可提取，页眉页脚正常。公式、表格与图中文字均未改。

临时页码为1–3，节号5.3；图表及公式在独立预览中按第5章编号，合并完整章节后可自动重排。正式主文件include仍保持原样。

## 保留与冻结

原section_5_3.tex、原阅读稿、revision_r1/和revision_r2/全部保留并核对Hash未变；第一至第四章、主文件、参考文献、模板和实验产物均未改。冻结目录为thesis/releases/ch05_5_3_2026-11-21/。

按用户本轮授权冻结5.3初稿。没有新增技术审计或实验，不commit/push；5.4等待用户与ChatGPT指定任务书，不自动写作。
''')
F.mkdir(parents=True)
for src,dest in [(O/'section_5_3.tex','section_5_3.tex'),(O/'section_5_3_preview.pdf','section_5_3.pdf'),(O/'SECTION_5_3_READING_DRAFT.md','READING_DRAFT.md'),(O/'changes.diff','changes_R2_R3.diff'),(O/'WORDCOUNT_REVIEW.md','WORDCOUNT_REVIEW.md'),(O/'LAYOUT_REVIEW.md','LAYOUT_REVIEW.md'),(O/'build.log','build.log'),(O/'build/section_5_3_preview.log','xelatex.log'),(O/'wordcount/result.json','wordcount.json')]:shutil.copyfile(src,F/dest)
# Frozen source entry uses the archived file rather than a mutable working fragment.
entry=(O/'section_5_3_preview.tex').read_text().replace('../docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r3/section_5_3.tex','releases/ch05_5_3_2026-11-21/section_5_3.tex');(F/'preview_entry.tex').write_text(entry)
freeze={'section':'5.3 GMM统计先验引导的仿真触觉伪配对构建','date':'2026-11-21','state':'FROZEN_DRAFT','authorization':'用户本轮确认R2内容审核通过，要求仅做两项R3排版调整、编译核查并在保留历史版本后冻结','R3_review':'agent layout verification after user-approved R2 content, not a separate claimed user R3 PDF approval','pages':3,'TeXcount':1999,'section_sha256':sha(F/'section_5_3.tex'),'pdf_sha256':sha(F/'section_5_3.pdf'),'source_technical_content_identical_to_R2':True,'original_R1_R2_retained':True,'HEAD':head,'global_main_unchanged':True,'chapter5_complete':False,'next_section_started':False,'commit_push':False}
(F/'FREEZE_STATUS.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
(F/'README.md').write_text('''# 第五章5.3冻结初稿

冻结日期：2026-11-21。内容为用户审核通过的R2，R3只修两张图底部留白和“分量数”不可断行；经独立编译及三页排版检查后，按用户本轮授权冻结。

- section_5_3.pdf：3页独立预览，节号5.3。
- section_5_3.tex：冻结正文片段；技术内容与R2相同。
- READING_DRAFT.md：完整阅读稿。
- preview_entry.tex：冻结片段的独立编译入口，使用现有thesis模板。
- WORDCOUNT_REVIEW.md / wordcount.json：TeXcount1999。
- LAYOUT_REVIEW.md / changes_R2_R3.diff：检查与两项修改。
- build.log / xelatex.log：实际构建记录。
- FREEZE_STATUS.json / SHA256SUMS：冻结范围及完整性。

正式thesis.tex及include未改。原稿、R1、R2和R3工作目录均保留。这里只冻结5.3初稿，不代表第五章完成；5.4待具体任务书，不自动写作。

复编时从项目thesis目录加载原latexmkrc及latexmk.projectrc，指定本目录preview_entry.tex，输出到新的临时目录，不覆盖冻结PDF；实际本轮构建脚本保存在docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r3/build_preview.sh。
''')
(F/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(F.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
check={'labels':labels,'fonts':fonts,'no_overfull':True,'no_undefined_refs':True,'semantic_normalization_matches_R2':True,'first_page_last_line':texts[0].splitlines()[-2],'second_page_first_line':texts[1].splitlines()[1],'pages':3,'TeXcount':1999,'reviewed_images':['page_01.png','page_02.png','page_03.png']}
(O/'static_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
integrity={'task':'CH05-5.3-R3','date':'2026-11-21','protected_before':base['protected'],'changed_protected_files':changed,'HEAD_before':base['HEAD'],'HEAD_after':head,'content_and_equations_unchanged':True,'old_versions_preserved':True,'global_main_bib_template_ch1to4_unchanged':True,'new_training_simulation':False,'Test_Holdout_access':False,'compiled':True,'frozen':True,'freeze_scope':'5.3初稿，仅内容已获用户R2审核且R3代理排版核查通过','freeze_directory':str(F),'5_4_started':False,'commit_push':False}
(O/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r3/count_section.py','bash docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r3/build_preview.sh','PYTHONPATH=thesis/.tools/python python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r3/freeze_section.py'],'logs':'build.log保留首次纯加底部留白和最终局部调整构建；最终引擎日志在build/'},ensure_ascii=False,indent=2))
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
print('FROZEN',F,'pages3/count1999; protected',len(base['protected']),'unchanged')
