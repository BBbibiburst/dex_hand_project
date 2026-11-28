from pathlib import Path
import json,hashlib,difflib,re,sys
r=Path(__file__).resolve().parents[3];d=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((d/'baseline.json').read_text());changes=[];comparisons={}
for p,h in b['source_hashes'].items():
 f=r/p;now=sha(f);comparisons[p]={'before':h,'after':now,'unchanged':h==now}
 if now!=h:
  a=(d/'before'/p).read_text(encoding='utf-8-sig').splitlines(True);z=f.read_text(encoding='utf-8-sig').splitlines(True)
  changes+=difflib.unified_diff(a,z,fromfile='before/'+p,tofile=p)
(d/'changes.diff').write_text(''.join(changes))
protected={p:x for p,x in comparisons.items() if p not in ['thesis/front/cover.tex','thesis/front/abstract.tex','thesis/THESIS_PERSONAL_INFORMATION.csv','thesis/body/ch01_introduction.tex','thesis/body/ch02_simulation_platform.tex','thesis/body/ch03_grasp_demonstrations.tex']}
assert all(x['unchanged'] for x in protected.values()),[p for p,x in protected.items() if not x['unchanged']]
assert sha(r/'thesis_draft_commented.pdf')==b['commented_pdf_sha256']
log=(d/'build/thesis.log').read_text(errors='replace')
checks={'undefined_reference_or_citation':bool(re.search(r'(?:Reference|Citation).*undefined|There were undefined references',log)), 'missing_character': 'Missing character:' in log,'overfull_hbox':re.findall(r'Overfull \\hbox \(([^\n]+)',log),'build_success':'Output written on' in log}
(d/'build_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
sys.path.insert(0,str(r/'thesis/.tools/python'));import pymupdf
pdf=pymupdf.open(d/'thesis_annotation_correction_r2.pdf')
(d/'READING_DRAFT.md').write_text('# 当前R2预览逐页阅读稿\n\n直接从本轮PDF提取，换行沿用PDF；源变更以changes.diff为准。\n\n'+'\n\n'.join(f'## PDF第{i+1}页\n\n'+p.get_text() for i,p in enumerate(pdf)))
(d/'VALIDATION.md').write_text(f'''# 构建与统计\n\n- 现有模板，独立输出目录；4轮XeLaTeX及1轮BibTeX完成。\n- 本轮PDF {len(pdf)}页；上一轮W1为94页，增加4页，包含摘要变化、两张新增图及浮动分页影响。\n- 项目原TeXcount工具真实正文合计54737，上一轮54441，差+296；没有更改计数规则。\n- 原工具仍显示旧的“当前六章”字样，且按旧标题匹配将第五章标MISSING；实际include图已含第五章各片段，其字数包含在正文总计中。报告页数栏也读取旧PDF，不能用它代替本轮98页；这些历史工具限制未擅自修改。逐文件实际结果见wordcount/。\n- 无未解析引用、无Missing character。既有窄表英文名称存在两处Overfull hbox：5.79674pt、2.0989pt；另有rsfs字体尺寸替代及glossaries未打印警告，不标记为零警告。\n- 已实际查看新增图完整输出及PDF第34、36页，两图完整，页眉页码及图题正常；并渲染第2、3页供检查封面。没有声称重新人工逐页审核98页。\n- 第五至第七章、第四章、主文件和参考文献与本轮备份Hash一致。历史release未作为写入目标；具体核对范围见integrity.json。\n\n新增图编号由LaTeX自动生成，本预览为图2-10与图2-11。原章内后续标签使用自动引用；没有手改历史冻结PDF编号。\n''')
files={str(p.relative_to(r)):sha(p) for p in d.iterdir() if p.is_file() and p.name!='integrity.json'}
files.update({str(p.relative_to(r)):sha(p) for p in (r/'thesis/figures/annotation_r2').iterdir() if p.is_file()})
source_files=['outputs/tactile_p45_mask_ablation_2026-11-18/A/training_summary.json','outputs/tactile_p48_unified_four_route_audit_2026-11-19/code/generate.py','outputs/tactile_p48_unified_four_route_audit_2026-11-19/metrics_swd.csv','outputs/tactile_p49_round_frozen_adversarial_2026-11-19/code/train.py']
(d/'integrity.json').write_text(json.dumps({'source_comparison':comparisons,'protected_checked':protected,'protected_checked_all_unchanged':True,'original_annotated_pdf_unchanged':True,'original_annotated_pdf_sha256':b['commented_pdf_sha256'],'artifacts':files,'F0_sources_current_hashes':{p:sha(r/p) for p in source_files},'execution':{'new_training':False,'model_inference':False,'simulation':False,'read_test_holdout_arrays':False,'video_decode':'five historical P32 task videos only','commit_push':False,'freeze':False},'scope_note':'Only source_comparison entries have pre/post comparison. Figure sources have current hashes and read-only processing, not a fabricated pre-run snapshot of all experimental assets.'},ensure_ascii=False,indent=2)+'\n')
print('finalized',len(pdf),'pages',len([x for x in comparisons.values() if not x['unchanged']]),'modified source files',checks)
