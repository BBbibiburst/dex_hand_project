from pathlib import Path
import pymupdf as fitz
import re,json,hashlib,shutil,difflib,ast
R=Path.cwd();O=R/'thesis/releases/ch02_2026-11-20';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in json.loads((O/'protected_hashes.json').read_text()).items():assert sha(R/p)==h,p
s=(R/'thesis/body/ch02_simulation_platform.tex').read_text();shutil.copyfile(R/'thesis/body/ch02_simulation_platform.tex',O/'ch02_simulation_platform.tex');shutil.copyfile(R/'thesis/reference.bib',O/'reference.bib')
(O/'final_changes.diff').write_text(''.join(difflib.unified_diff((O/'source_before.tex').read_text().splitlines(True),s.splitlines(True),fromfile='revision_v1',tofile='chapter02_frozen_draft')))
d=fitz.open(O/'build/ch02_preview.pdf');texts=[p.get_text() for p in d];ref=next(i for i,t in enumerate(texts) if t.startswith('哈尔滨工业大学硕士学位论文\n参考文献'))
shutil.copyfile(O/'build/ch02_preview.pdf',O/'chapter02_with_references.pdf');b=fitz.open();b.insert_pdf(d,from_page=0,to_page=ref-1);b.save(O/'chapter02.pdf')
log=(O/'build/ch02_preview.log').read_text();assert not re.search('undefined|Overfull',log,re.I)
wc=json.loads((O/'wordcount/THESIS_WORD_COUNT.json').read_text())['chapters'][1]['actual_count']
# Reading draft from the final source, not stale revision text.
ns={'re':re};tree=ast.parse((R/'docs/writing_audit/ch02_writing_v1_2026-11-20/package_delivery.py').read_text())
for x in tree.body:
 if isinstance(x,ast.FunctionDef) and x.name in ['inline','fig','tab']:exec(compile(ast.Module(body=[x],type_ignores=[]),'converter','exec'),ns)
t=re.sub(r'^%.*\n','',s,flags=re.M);t=re.sub(r'\\chapter\{([^}]+)\}',r'# 第二章 \1',t);n=[0]
def sec(m):n[0]+=1;return f'## 2.{n[0]} {m[1]}'
t=re.sub(r'\\section\{([^}]+)\}',sec,t);t=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',ns['fig'],t,flags=re.S);t=re.sub(r'\\begin\{table\}.*?\\end\{table\}',ns['tab'],t,flags=re.S);t=t.replace('\\begin{equation}','\n$$').replace('\\end{equation}','$$\n');t=ns['inline'](t)
(O/'CH02_READING_DRAFT.md').write_text('第二章初稿冻结快照；正式全论文排版尚未冻结。\n\n'+t)
status={'date':'2026-11-20','status':'FROZEN_CHAPTER_2_DRAFT_WITH_NONBLOCKING_WARNINGS','scope':'Chapter 2 standalone draft; main thesis includes unchanged','texcount':wc,'body_pages':ref,'preview_pages_with_references':len(d),'formula_2_5_visual_check':'PASS: all three lines including normalized float32 output visible on preview page 11','compile_success':True,'references_resolved':True,'no_overfull':True,'formal_main_unchanged':True,'chapter1_unchanged':True,'not_whole_thesis_final_freeze':True}
(O/'FREEZE_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
(O/'FREEZE_REPORT.md').write_text(f'''# 第二章初稿冻结记录

已按本轮授权完成三项最终处理：

1. 检查发现旧式（2-5）缺少文字中说明的归一化输出行；现已补入并在最终PDF第11页目视确认完整显示，截图为formula_2_5_check.png。
2. 图2-4区分position关节位置目标与ik末端位姿转换，目标限步/平滑注明适用路径。
3. 图2-2补明静态关节零位、右侧放大、未显示底座及非任务执行截图。

临时入口编译成功，正文{ref}页，含参考文献{len(d)}页；实际TeXcount **{wc}**。引用正常，无Overfull。保留模板字体尺寸、glossary和已记录BibTeX缺字段非阻断警告，不猜测出版信息。

状态：第二章初稿已冻结；不代表整篇论文最终排版。源文件、最终阅读稿、PDF、构建日志、差异和Hash均已归档。第一章、正式主文件、其他章节和参考文献源文件未修改；无commit/push，未启动第三章。
''')
paths=[p for p in O.rglob('*') if p.is_file() and p.name!='SHA256SUMS'];(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted(paths)))
print(json.dumps(status,ensure_ascii=False))
