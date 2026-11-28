from pathlib import Path
import json,hashlib,difflib,re,sys
r=Path(__file__).resolve().parents[3];d=Path(__file__).parent;H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();base=json.loads((d/'baseline.json').read_text());compare={};diff=[]
allowed={'thesis/body/ch01_introduction.tex','thesis/body/ch02_simulation_platform.tex','thesis/body/ch03_grasp_demonstrations.tex','thesis_audit/2026-11-28/experiment_strengthening/MASTER_PLAN.md'}
for rel,old in base.items():
 p=r/rel;after=H(p);compare[rel]={'before':old,'after':after,'unchanged':old==after}
 if after!=old:
  assert rel in allowed,rel
  diff+=difflib.unified_diff((d/'before'/rel).read_text(encoding='utf-8-sig').splitlines(True),p.read_text(encoding='utf-8-sig').splitlines(True),fromfile='before/'+rel,tofile=rel)
(d/'changes.diff').write_text(''.join(diff))
# Preserve displayed mathematical definitions and all existing data tables.
content={}
for rel in sorted(x for x in allowed if x.endswith('.tex')):
 a=(d/'before'/rel).read_text();b=(r/rel).read_text();entry={}
 for env in ['equation','table','algorithm']:
  pat=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
  entry[env+'_blocks_unchanged']=re.findall(pat,a,re.S)==re.findall(pat,b,re.S)
 assert all(entry.values()),(rel,entry)
 content[rel]=entry
j=json.loads((d/'wordcount/THESIS_WORD_COUNT.json').read_text());graph=j['actual_include_graph']['files'];active='\n'.join((r/p).read_text() for p in graph if p.endswith('.tex'))
labels=re.findall(r'\\label\{([^}]+)\}',active);dupes=sorted({x for x in labels if labels.count(x)>1});assert not dupes,dupes
log=(d/'build/thesis.log').read_text(errors='replace');check={'equations_tables_algorithms':content,'duplicate_labels':dupes,'undefined_ref_cite':bool(re.search(r'(?:Reference|Citation).*undefined|There were undefined references',log)),'missing_character':'Missing character:' in log,'overfull_hbox':re.findall(r'Overfull \\hbox \(([^\n]+)',log),'build_completed':'Output written on' in log}
assert check['build_completed'] and not check['undefined_ref_cite'] and not check['missing_character']
(d/'static_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
sys.path.insert(0,str(r/'thesis/.tools/python'));import pymupdf
pdf=pymupdf.open(d/'thesis_annotation_batch2.pdf')
(d/'READING_DRAFT.md').write_text('# 本批完整PDF阅读稿\n\n从最终PDF直接提取，保留分页。方法图中文采用字形轮廓嵌入，图中文字不一定可文本提取，应同时查看PDF图页。\n\n'+'\n\n'.join(f'## PDF第{i+1}页\n\n'+p.get_text() for i,p in enumerate(pdf)))
(d/'VALIDATION.md').write_text(f'''# 本批构建与核对\n\n最终独立预览为{len(pdf)}页，正文TeXcount实际54294，较上一批54737减少443。原始统计输出位于wordcount/。本批仍使用原TeXcount规则；旧标题匹配造成的第五章MISSING提示及旧PDF页数字段不代表当前正文缺章或本轮PDF页数。\n\n4轮XeLaTeX和1轮BibTeX完成，未改主文件、模板、参考文献。无未解析引用、无LaTeX Missing character；既有绪论窄表仍有5.79674pt与2.0989pt两处英文宽度警告，另有原rsfs尺寸替代与glossaries提示。未声称零警告。\n\n已实际查看最终PDF第25、30、42页，对应控制图、触觉处理算子图和历史轨迹图；并查看三个独立图的PNG。首次发现合入PDF后的中文缺字，已修改矢量图字形嵌入方式并重新编译复查，最终中文显示完整，页眉、页码和图题正常。不是仅凭编译成功判定图正常，也未声称逐页人工复审98页。\n\n第一至第三章的equation、table、algorithm环境与本批备份逐块一致。新增轨迹使用历史保存字段，不改变原始实验数据或Strict阈值。第四至第七章、前置文件、身份CSV、主文件及Bib与本批备份Hash一致。当前图号自动为图2-4、图2-8、图3-5，后续图号由LaTeX顺延；无重复label或未解析交叉引用。\n\n上一批R2 PDF继续保留，未覆盖发布版、原批注稿或冻结产物。详细保护范围与Hash见integrity.json。\n''')
old=r/'thesis_audit/2026-11-28/annotation_correction_r2';ih=json.loads((old/'integrity.json').read_text());oldpdf=old/'thesis_annotation_correction_r2.pdf';assert H(oldpdf)==ih['artifacts'][str(oldpdf.relative_to(r))]
sourcepaths=['source/control/arm.py','source/control/composite.py','source/control/end_effectors.py','source/envs/rl_env.py','source/sensors/tactile/signal_processing.py','assets/grippers/dex_hand/dex_hand.xml','configs/robot_profiles/rm75b_dex_hand.json']
traj=json.loads((d/'trajectory_source.json').read_text())
for p,h in traj['source_files'].items():assert H(r/p)==h
art={str(p.relative_to(r)):H(p) for p in d.iterdir() if p.is_file() and p.name!='integrity.json'};art.update({str(p.relative_to(r)):H(p) for p in (r/'thesis/figures/annotation_batch2').iterdir() if p.is_file()})
(d/'integrity.json').write_text(json.dumps({'source_comparison':compare,'protected_all_unchanged':all(v['unchanged'] for k,v in compare.items() if k not in allowed),'previous_R2_pdf_unchanged':True,'read_only_source_current_hashes':{p:H(r/p) for p in sourcepaths},'trajectory_inputs_rechecked':traj['source_files'],'artifacts':art,'execution':{'training':False,'model_inference':False,'simulation':False,'read_test_holdout_arrays':False,'experiment_metric_recompute':False,'historical_record_plotting':True,'physical_tasks':False,'commit_push':False,'freeze':False},'scope':'Pre/post hash comparison covers source_comparison only; experimental source current hashes are not fabricated pre/post whole-dataset checks.'},ensure_ascii=False,indent=2)+'\n')
print('DONE',len(pdf),'pages; protected unchanged; equation/table/algorithm blocks unchanged; no unresolved references')
