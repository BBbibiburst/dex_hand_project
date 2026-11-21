from pathlib import Path
import re,json,hashlib,shutil,difflib
import pymupdf as fitz
R=Path.cwd();O=R/'docs/writing_audit/ch03_writing_v1_2026-11-20/compile_review';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=R/'thesis/body/ch03_grasp_demonstrations.tex';tex=src.read_text();old=(O/'chapter_before.tex').read_text()
d=fitz.open(O/'build/ch03_preview.pdf');text=[p.get_text() for p in d];ref=next(i for i,t in enumerate(text) if t.startswith('哈尔滨工业大学硕士学位论文\n参考文献'))
shutil.copyfile(O/'build/ch03_preview.pdf',O/'chapter03_preview.pdf')
b=fitz.open();b.insert_pdf(d,from_page=0,to_page=ref-1);b.save(O/'chapter03_body_preview.pdf');b.close()
(O/'changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),tex.splitlines(True),fromfile='chapter_v1_before.tex',tofile='thesis/body/ch03_grasp_demonstrations.tex')))
log=(O/'build/ch03_preview.log').read_text();blg=(O/'build/ch03_preview.blg').read_text();aux=(O/'build/ch03_preview.aux').read_text();bib=(R/'thesis/reference.bib').read_text()
keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib));used=sorted({k.strip() for x in re.findall(r'\\cite\{([^}]+)\}',tex) for k in x.split(',')})
labels={m[0]:{'number':m[1],'page':m[2]} for m in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)}
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
fontrecords={}
pages=[]
for i,p in enumerate(d):
 for f in p.get_fonts():
  if f[0] not in fontrecords:
   data=d.extract_font(f[0]);fontrecords[f[0]]={'font':f[3],'embedded':bool(data[3])}
 words=p.get_text('words');outside=[w[:4] for w in words if w[0]<-1 or w[1]<-1 or w[2]>p.rect.width+1 or w[3]>p.rect.height+1]
 pages.append({'page':i+1,'text_characters':len(text[i]),'blank':len(text[i].strip())<25,'outside_page_words':outside,'replacement_characters':text[i].count('\ufffd'),'header_present':'哈尔滨工业大学硕士学位论文' in text[i]})
 p.get_pixmap(matrix=fitz.Matrix(.65,.65)).save(O/f'page_{i+1:02d}.png')
for i in [3,5,7,9]:d[i].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(O/f'detail_{i+1:02d}.png')
oldhash=json.loads((O/'protected_before.json').read_text());changed=[p for p,h in oldhash.items() if sha(R/p)!=h];assert changed==['thesis/body/ch03_grasp_demonstrations.tex'],changed
check={'status':'第三章V1临时编译完成，等待用户PDF审核','missing_bib_keys':sorted(set(used)-keys),'used_bib_keys':used,'missing_reference_labels':sorted(set(refs)-set(labels)),'compiled_labels':labels,'undefined_or_multiply_defined_warning':bool(re.search('undefined|multiply defined',log,re.I)),'overfull_boxes':re.findall(r'Overfull[^\n]*',log),'underfull_boxes':re.findall(r'Underfull[^\n]*',log),'bibtex_warnings':[l for l in blg.splitlines() if l.startswith('Warning--')],'pdf_pages':len(d),'body_pages':ref,'reference_pages':len(d)-ref,'fonts':list(fontrecords.values()),'pages':pages,'changed_protected_files':changed,'formal_main_first_two_chapters_bib_unchanged':True,'frozen':False}
assert not check['missing_bib_keys'] and not check['missing_reference_labels'] and not check['undefined_or_multiply_defined_warning'] and not check['overfull_boxes']
assert all(f['embedded'] for f in fontrecords.values())
assert not any(p['blank'] or p['outside_page_words'] or p['replacement_characters'] for p in pages)
(O/'citation_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
wc=json.loads((O/'wordcount/THESIS_WORD_COUNT.json').read_text())['chapters'][2]['actual_count']
report=f'''# 第三章临时编译与排版检查

状态：**第三章V1临时编译完成，等待用户PDF审核。** 未冻结、未提交，不进入第四章。

## 构建与页数

按当前hithesis、fontset=fandol使用项目本地XeLaTeX + BibTeX + latexmk。临时入口ch03_preview.tex仅引入本章并设置chapter计数为2，正文显示第3章，正式thesis.tex及include关系保持不变。

chapter03_preview.pdf共**{len(d)}页**，其中正文**{ref}页**、参考文献**{len(d)-ref}页**；独立正文副本为chapter03_body_preview.pdf。页码从1开始，只是独立预览页码，不是整篇论文最终页码。完整构建日志为build.log，最终引擎/BibTeX日志在build/。

## 图表与公式

- 6图均正常排出，无文字越出页面或流程框。图3-6明确是待补真实案例画面，没有生成或插入虚构图片。
- 6表均在版心内，未缩小到额外字号；Sugar Box五列结果表在第10页完整显示。表格沿用原small字号，没有通过压缩文字或缩放整表解决问题。
- 历史批量结果表中FINAL_VERIFIED最初发生标识符中间断行。本轮唯一源文件修改为该单元格显式换行并用mbox保持标识符完整；数据、Strict阈值、评分系数和正文段落全部未变。详见changes.diff。
- 四个公式正常显示：式3-1候选合同；式3-2世界目标与安装旋转补偿，转置标记清楚；式3-3评分中位置系数保持20；式3-4插值与smoothstep，上下标完整。未发现需要重新解释的技术问题。
- 图号3-1至3-6、表号3-1至3-6、式号3-1至3-4已从实际aux解析；引用目标全部存在，无未解析引用。

## 页面与字体检查

已检查全页缩略图，并放大检查公式页、任务阶段表及第10页两张实验表。中文使用嵌入的FandolSong/FandolHei，英文和数学字体正常；PDF中文字可提取，未检出替代字符。页眉为硕士论文模板，页脚数字连续。

无空白页、无正文/表格越界、无孤立小节标题。第12页为本章小结结束页，存在自然章末留白；参考文献单独起页，不将其误判为空白插页。未为减少页数改变页边距、字号或段落内容。

## 保留的非阻断警告

最终日志有{len(check['underfull_boxes'])}项Underfull hbox排版提示，未发生Overfull。保留模板数学字体尺寸替代及未打印glossary提示。BibTeX有{len(check['bibtex_warnings'])}项缺字段警告，逐条见citation_check.json；没有补造地址或访问日期，也未修改reference.bib。引用条目均已正常生成。

## 修改范围及保护

仅修改第三章表格中一个单元格的换行。第一、二章正文和冻结PDF、正式主文件、参考文献与模板的前后Hash一致；构建配置原样加载、未修改，历史代码及数据未修改。排版调整后的TeXcount仍为**{wc}**，只作保持内容规模的核对，不扩写。

本轮提供PDF供用户审核，不自动冻结、不commit/push、不启动第四章或任何实验。
'''
(O/'layout_review.md').write_text(report)
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['bash docs/writing_audit/ch03_writing_v1_2026-11-20/compile_review/build_preview.sh','./scripts/count_thesis_words.sh --chapter 3 --output-dir docs/writing_audit/ch03_writing_v1_2026-11-20/compile_review/wordcount','PYTHONPATH=thesis/.tools/python python3 docs/writing_audit/ch03_writing_v1_2026-11-20/compile_review/package_review.py'],'original_main_modified':False,'no_freeze':True},indent=2)+'\n')
paths=[src]+[p for p in O.rglob('*') if p.is_file() and p.name!='SHA256SUMS']
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted(paths)))
print('PASS:',len(d),'pages,',ref,'body; TeXcount',wc,'; labels and fonts resolved; one layout-only change.')
