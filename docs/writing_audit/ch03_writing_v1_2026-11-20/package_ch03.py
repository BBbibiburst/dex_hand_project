"""Package stage-3 chapter draft; no compilation, training or simulation."""
from pathlib import Path
import ast,re,json,hashlib,difflib,shutil
R=Path.cwd();O=R/'docs/writing_audit/ch03_writing_v1_2026-11-20';A=R/'docs/writing_audit/ch03_material_audit_v1_2026-11-20';P=R/'thesis/body/ch03_grasp_demonstrations.tex'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tex=P.read_text();bib=(R/'thesis/reference.bib').read_text()
shutil.copyfile(P,O/P.name)
(O/'changes.diff').write_text(''.join(difflib.unified_diff([],tex.splitlines(True),fromfile='/dev/null',tofile=str(P.relative_to(R)))))
keys=re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib);cites=sorted({x.strip() for m in re.findall(r'\\cite\{([^}]+)\}',tex) for x in m.split(',')})
labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
alllabels=[x for q in (R/'thesis/body').glob('*.tex') for x in re.findall(r'\\label\{([^}]+)\}',q.read_text())]
stack=[];errors=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',tex):
 if m[1]=='begin':stack.append(m[2])
 elif not stack or stack.pop()!=m[2]:errors.append(m[0])
assert not stack and not errors
assert set(cites)<=set(keys)
assert set(refs)<=set(alllabels)
assert all(alllabels.count(x)==1 for x in labels)
figs=[]
for i,m in enumerate(re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',tex,re.S),1):
 figs.append({'index':i,'label':re.search(r'\\label\{([^}]+)\}',m[0])[1],'caption':re.search(r'\\caption\{([^}]+)\}',m[0])[1],'placeholder':'图位' in m[0]})
tables=[{'label':re.search(r'\\label\{([^}]+)\}',m[0])[1],'caption':re.search(r'\\caption\{([^}]+)\}',m[0])[1]} for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',tex,re.S)]
protected=json.loads((O/'protected_before.json').read_text());changed=[p for p,h in protected.items() if not (R/p).is_file() or sha(R/p)!=h];assert not changed,changed
source_index=json.loads((A/'source_index.json').read_text())
assert all(sha(R/p)==x['sha256'] for p,x in source_index['current_source_files'].items())
assert all(sha(R/p)==h for p,h in source_index['historical_json_hashes'].items())
# Reuse converter definitions only, without executing an old packaging script.
ns={'re':re}
for x in ast.parse((R/'docs/writing_audit/ch02_writing_v1_2026-11-20/package_delivery.py').read_text()).body:
 if isinstance(x,ast.FunctionDef) and x.name in ['inline','fig','tab']:exec(compile(ast.Module(body=[x],type_ignores=[]),'reading_conversion','exec'),ns)
t=re.sub(r'^%.*\n','',tex,flags=re.M);t=re.sub(r'\\chapter\{([^}]+)\}',r'# 第三章 \1',t);n=[0]
def sec(m):n[0]+=1;return f'## 3.{n[0]} {m[1]}'
t=re.sub(r'\\section\{([^}]+)\}',sec,t)
t=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',ns['fig'],t,flags=re.S);t=re.sub(r'\\begin\{table\}.*?\\end\{table\}',ns['tab'],t,flags=re.S)
t=t.replace('\\begin{equation}','\n$$').replace('\\end{equation}','$$\n');t=ns['inline'](t)
(O/'CH03_READING_DRAFT.md').write_text('第三章V1正文完成，等待用户审核。由最终TeX生成的完整阅读稿；引用及交叉引用保留键，尚未编译或冻结。\n\n'+re.sub(r'\n{3,}','\n\n',t).strip()+'\n')
wc=json.loads((O/'wordcount/THESIS_WORD_COUNT.json').read_text());c=wc['chapters'][2];m=c['metrics'];assert 8000<=c['actual_count']<=9000
budget=[650,1700,1350,1550,1450,1300,420]
lines=['# 第三章实际TeXcount','',''+wc['engine'],'','统计源：`thesis/body/ch03_grasp_demonstrations.tex`。实际章题采用本任务书的“面向灵巧手操作的抓取候选生成与专家示教构建”；统计工具摘要仍显示V2旧规划题名，不代表正文题名未更新。没有修改规划或主文件。','','|节|实际|本轮目标|差值|','|---|---:|---:|---:|']
for x,b in zip(c['sections'],budget):lines.append(f"|{x['id']} {x['title']}|{x['actual_count']}|{b}|{x['actual_count']-b:+d}|")
lines +=['',f"全章 **{m['total']}**，符合8000–9000。相对任务书中心8420变化{m['total']-8420:+d}。计数=text {m['text']} + 标题 {m['headings']} + 图表题/脚注 {m['captions_footnotes']}；章标题另计，小节合计不含章标题。",'',f"行内公式{m['inline_math']}、独立公式环境{m['display_math']}，单独统计不折算正文。表格文字和流程图框内文字按现有规则计入text，题注计入caption类别；无脚注、代码块。TeXcount规则自测结果：{wc['tests_passed']}。",'', '已执行--chapter 3和--files，命令输出见count_chapter.log/count_files.log，原始解析见wordcount/raw与wordcount_audit.html。--files仅显示正式主文件的旧六章include图；新章是V2规划路径统计，并未接入正式主文件。未编译PDF，页数UNKNOWN。','', '初次写作9023，超过上限后删去相邻重复说明；没有通过公式、重复内容或不存在的结果填充。各节按论证需要调整，未强制机械对齐目标。']
(O/'WORDCOUNT_REVIEW.md').write_text('\n'.join(lines)+'\n')
(O/'EVIDENCE_CITATION_MAP.md').write_text('''# 论点、文献与内部证据对应

本轮不新增或修改BibTeX；复用以下已核验键：

|键|正文用途|边界|
|---|---|---|
|zurbrugg2025graspqp|3.2官方可微抓取指标来源|本项目仅适配，不宣称原创GraspQP|
|zurbrugg2026dexevolve|3.2进化优化原方法来源|本项目为DexEvolve-style，不冒充完整Isaac复现|
|mandlekar2022demonstrations|3.5人类示教与质量背景|不是本项目任务性能依据|
|mandlekar2023mimicgen|3.5自动构造示教的背景|不声称复现MimicGen|

上述元数据沿用reference.bib与第一章已审核文献目录；未扩展检索、未补造DOI。

|正文论点|内部代码/结果|等级与范围|
|---|---|---|
|3.1候选坐标、六驱动及合同|contracts.py::GraspCandidate、executor.py::candidate_world_pose|L2指定函数|
|3.2几何指标及约束|graspqp_adapter.py::refine_candidates_with_graspqp、hand_surrogate.py|L2已核验适配函数，不代表完整算法消融|
|3.2进化细化|dexevolve.py的变异、交叉、评价及档案；budget.py|L2当前实现，历史预算不回填|
|3.2格点与条件后续编辑|templates.py、batch_grasp_edit.py；Top100 summary|L2实现与清单统计，RL2条为历史状态|
|3.3坐标与两目标评分|executor.py::candidate_world_pose/rank_candidates_for_scene|L2函数；位置残差系数实际20|
|3.4插值与阶段|executor.py::_run_pose_segment/execute_grasp|L2函数；不保证全局无碰撞或无分支跳变|
|3.4严格与普通回放|strict_replay.py、profiles.py、grasp_pipeline/replay.py|L2函数；当前tail穿透判据与全程记录分开|
|3.5五任务逻辑|task_transfer.py/stack_transfer.py/push_transfer.py/nut_transfer.py|L2所审计函数/片段；前缀复用与重执行分开|
|3.6批次结果|outputs下dex_hand_top100_v2、pick_place_top100_final、stack_task_valid_v24、push_top100_final、nut_assembly_top100_final各summary.json|L2条目重计；内含成功是历史声明，不是本轮L3|
|3.6 Sugar十条Strict表|ch03_material_audit的historical_results_check.json::source_a_ten_rows及其逐条Strict原件|L2指定文件，表中数值来自对应final_lift和tail_max_angular_speed，显示保留5位小数|
|3.6九十次汇总失败|outputs/dp_p1_2026-10-11/generation_formal_summary.json|L1汇总，未逐条认证90源轨迹|
|3.6 Lift单例、Nut retain局部对照|five_task_p32_b_2026-11-03/lift_sugar_strict.json及Nut结果；docs/progress/2026-11-03_p32_b_online_recorder.md|指定结果L2及报告解释L1，不称跨seed稳定提升|

源函数行号、完整Hash与历史原件路径见审核通过的ch03_material_audit_v1_2026-11-20/source_index.json及CH03_EVIDENCE_MATRIX.md。没有借用14/127未知批次，无L3。

## 任务书与代码差异

任务书3.3评分示例的位置残差系数为2，审核材料和实际executor.py为20。正文式eq:ch03_reachability_score采用代码中的20，未修改任务书、实现或历史结果。原始评分是20*maximum_position_error+0.6*maximum_orientation_error+0.08*travel。

## 图表和公式

6图：整体流程、候选模块、可达筛选、轨迹与Strict、多任务关系为5张实现示意框图；第6张典型成功/失败案例为待补真实画面图位。没有伪造实验曲线或图像。拟用Lift及Nut对应历史视频/manifest；失败视频若无对应原件，不制作虚假截图。

6表：候选参数、Strict阈值、任务阶段、历史批次、Sugar10条、90次失败汇总。4个公式环境：候选合同、目标变换、可达评分、插值。无新增未实现的控制理论。
''')
(O/'WRITING_REVIEW.md').write_text('''# 第三章V1写作自审

状态：**第三章V1正文完成，等待用户审核。** 未编译、未冻结，不自动进入第四章。

1. 七节按候选生成—约束筛选—轨迹—任务—实验组织。章节采用任务书正式标题，文件沿用V2规范body/ch03_grasp_demonstrations.tex，不改主文件。
2. 第三方GraspQP/DexEvolve标明来源和适配身份；未将格点、代码量或脚本量作为理论创新。
3. 可达评分位置系数遵循实际20，而非任务书示例2，差异见EVIDENCE_CITATION_MAP.md。
4. 预检限于pregrasp/grasp两个目标；不写全路点连续无碰撞保证；普通Replay、生成阶段success及FINAL状态严格区分。
5. 旧action与ctrl语义、PickPlace前缀恢复、Stack筛选域、Push专属接触、Nut旧/新几何分别说明；详细数据关联留第四章。
6. 全部批次是已有结果，缓存、variant、Root、Episode区分。Sugar10条只支持局部扰动；90尝试明确L1；二者不累计为新总量。
7. 14/127批次及UltraDexGrasp未被补造；没有新盲测、实机或统一成功率的声明。
8. 全部引用键存在，标签不重复，交叉引用有目标，环境配对完整；结果见static_checks.json。静态检查不能替代排版检查。
9. 第一、二章源与冻结PDF、主文件、参考文献及全部既有受保护文件Hash保持不变；源实现及所索引历史结果也未改变。

## 剩余问题

- 典型案例图仍待补原始画面；本轮不生成假截图。五张流程示意为实现逻辑，不是新实验结果。
- 未编译，图表浮动、表格宽度和最终编号待用户审核后检查，尤其Sugar五列参数表。
- 公平算法对照和完整历史版本绑定不足，已限定论文结论，不以这些不足阻塞当前初稿。
- 统计工具摘要仍沿用规划题名，但读取本轮新文件；正文题名按用户本任务授权，未改V2文件。

本轮不commit/push、不运行训练/仿真/服务器，不修改Strict标准。等待用户与ChatGPT内容审核。
''')
check={'status':'第三章V1正文完成，等待用户审核','source':str(P.relative_to(R)),'texcount':m['total'],'figures':figs,'tables':tables,'equations':len(re.findall(r'\\begin\{equation\}',tex)),'citation_keys':cites,'missing_citations':sorted(set(cites)-set(keys)),'duplicate_bib_keys':[k for k in set(keys) if keys.count(k)>1],'missing_reference_targets':sorted(set(refs)-set(alllabels)),'duplicate_labels':[k for k in labels if alllabels.count(k)>1],'environment_errors':errors,'changed_protected_files':changed,'first_two_chapters_and_frozen_pdfs_unchanged':True,'reference_bib_unchanged':True,'no_compilation':True,'no_training_or_simulation':True}
(O/'static_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['./scripts/count_thesis_words.sh --chapter 3 --output-dir docs/writing_audit/ch03_writing_v1_2026-11-20/wordcount','./scripts/count_thesis_words.sh --files --output-dir docs/writing_audit/ch03_writing_v1_2026-11-20/wordcount','python3 docs/writing_audit/ch03_writing_v1_2026-11-20/package_ch03.py'],'latex_compiled':False},indent=2)+'\n')
(O/'README.md').write_text('''# 第三章V1交付

完整正文：thesis/body/ch03_grasp_demonstrations.tex；本目录同名文件为交付快照。

CH03_READING_DRAFT.md提供完整中文阅读稿；WORDCOUNT_REVIEW.md提供实际总量与分节统计；EVIDENCE_CITATION_MAP.md说明证据、文献、图表及任务书评分差异；WRITING_REVIEW.md列明自审和剩余问题；changes.diff为新增源文件完整差异；static_checks.json与SHA256SUMS记录检查。

状态：第三章V1正文完成，等待用户审核。无PDF编译、无冻结，不进入第四章。
''')
paths=[P]+[p for p in O.iterdir() if p.is_file() and p.name!='SHA256SUMS']
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted(paths)))
print('Packaged:',m['total'],'words;',len(figs),'figures;',len(tables),'tables;',check['equations'],'equations; existing files unchanged.')
