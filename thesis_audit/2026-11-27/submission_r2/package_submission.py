from pathlib import Path
import json,hashlib,zipfile,re,shutil,sys
r=Path('/home/zmy/dex_hand_project');th=r/'thesis';audit=r/'thesis_audit/2026-11-27/submission_r2';release=th/'releases/draft_2026-11-27_r2';release.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((th/'releases/draft_2026-11-27/SOURCE_MANIFEST.json').read_text());paths={th/n for n in old['sources']};paths.update((th/'figures/revision_20261127').iterdir())
wc=json.loads((audit/'wordcount/THESIS_WORD_COUNT.json').read_text());paths.update(r/n for n in wc['actual_include_graph']['files'])
for p in list(paths):
 if p.suffix=='.tex':
  for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',p.read_text()):
   q=th/name
   if not q.is_file():raise ValueError('Missing graphics '+name)
   paths.add(q)
paths={p for p in paths if p.is_file()};manifest={str(p.relative_to(th)):sha(p) for p in sorted(paths)}
with zipfile.ZipFile(release/'thesis_source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in manifest:z.write(th/name,'thesis/'+name)
with zipfile.ZipFile(release/'thesis_source.zip') as z:
 assert z.testzip() is None
 for name,h in manifest.items():assert hashlib.sha256(z.read('thesis/'+name)).hexdigest()==h
for src,dest in [('thesis_draft.pdf','thesis_draft.pdf'),('build.log','build.log'),('texcount_files.txt','texcount_files.txt')]:shutil.copy2(audit/src,release/dest)
shutil.copy2(audit/'wordcount/THESIS_WORD_COUNT.json',release/'THESIS_WORD_COUNT.json')
sys.path.insert(0,str(th/'.tools/python'));import pymupdf
pdf=pymupdf.open(release/'thesis_draft.pdf');log=(audit/'build/thesis.log').read_text(errors='replace');assert not any(k in log for k in ['undefined','multiply defined','Missing character']);assert len(pdf)==95
f=(release/'SOURCE_MANIFEST.json');f.write_text(json.dumps({'entry':'thesis/thesis.tex','status':'integrated seven-chapter revised draft R2; not final freeze','source_files':len(manifest),'sources':manifest,'pdf_sha256':sha(release/'thesis_draft.pdf'),'pages':len(pdf),'body_texcount':wc['actual_compilation_source_body']['total']},ensure_ascii=False,indent=2))
(release/'SUBMISSION_REVIEW.md').write_text('''# 完整论文初稿R2交付\n\n包含Stage B1跨章节逻辑与术语修订、19处流程/结构图重绘、当前MuJoCo模型与712触觉区域静态渲染、图2-6三类策略四个代表网格、图3-6历史成功画面，以及5段实现对应伪代码。未新增实验或改变评价结论。\n\nPDF为当前正式入口独立XeLaTeX四遍+BibTeX重编，95页；原93页发行包完整保留。源码ZIP含全部现有入口依赖、模板、参考文献和新图表资产，逐文件Hash验证通过。备份TeX时保存thesis_source.zip，解压后的thesis/thesis.tex为入口；正常编译仍需相应TeX发行版及字体。\n\n已知遗留：第一章两处轻微Overfull，部分文献citedate/address/year字段警告，缺少两条Nut失败直接录像，历史版本链和独立Root的原有限制。MuJoCo图为静态仿真渲染，不是真实硬件照片或新任务执行结果。\n\n这是新版初稿提交，不代表论文正式冻结。\n''')
(release/'WORDCOUNT.md').write_text(f"# 实际TeXcount\n\n当前主文件实际接入正文合计：{wc['actual_compilation_source_body']['total']}。PDF：95页。\n\n使用项目现有TeXcount规则，正文、标题、图注/脚注/表格内容按既定工具口径计数，数学不加权；未调整规则或对新增算法作手工估算。各文件原始输出见texcount_files.txt，完整结果见THESIS_WORD_COUNT.json。5段算法另按LaTeX环境数量登记，不将环境数量混入图表64项统计。\n")
(release/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(release.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
before=json.loads((audit/'before_submission.json').read_text());assert all(sha(r/p)==h for p,h in before['old_release'].items());assert all(sha(r/p)==h for p,h in before['active_modified_files'].items())
(audit/'SUBMISSION_VALIDATION.md').write_text(f"# 新版论文提交前核验\n\n- 当前主入口独立复编95页，未定义引用、重复标签及缺字警告0。\n- 源码包{len(manifest)}个文件，ZIP CRC及逐文件SHA256均通过。\n- 正文TeXcount {wc['actual_compilation_source_body']['total']}，统计规则未改变。\n- 原发行包所有Hash一致。编译及打包期间活动正文未再修改。\n- 新发行路径：thesis/releases/draft_2026-11-27_r2。\n- 仅提交论文源码、图片及相关审计报告，其他实验程序/数据和构建缓存不提交。\n")
print('PACKAGE COMPLETE',len(manifest),wc['actual_compilation_source_body']['total'])
