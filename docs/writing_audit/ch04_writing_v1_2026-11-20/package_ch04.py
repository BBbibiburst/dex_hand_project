"""Package the chapter and evidence/static checks, without compiling LaTeX."""
from pathlib import Path
import ast,re,json,hashlib,difflib,shutil
R=Path.cwd();O=R/'docs/writing_audit/ch04_writing_v1_2026-11-20';A=R/'docs/writing_audit/ch04_material_audit_v1_2026-11-20';P=R/'thesis/body/ch04_multimodal_data.tex'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tex=P.read_text();data=json.loads((A/'dataset_checks.json').read_text());bib=(R/'thesis/reference.bib').read_text()
protected=json.loads((O/'protected_before.json').read_text());changed=[p for p,h in protected.items() if sha(R/p)!=h];assert not changed,changed
inputs=json.loads((A/'integrity.json').read_text())['audited_input_sha256'];assert all(sha(R/p)==h for p,h in inputs.items())
shutil.copyfile(P,O/P.name);(O/'changes.diff').write_text(''.join(difflib.unified_diff([],tex.splitlines(True),fromfile='/dev/null',tofile=str(P.relative_to(R)))))
keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib));cites=sorted({v.strip() for x in re.findall(r'\\cite\{([^}]+)\}',tex) for v in x.split(',')})
labels=re.findall(r'\\label\{([^}]+)\}',tex);alllabels=[x for p in (R/'thesis/body').glob('*.tex') for x in re.findall(r'\\label\{([^}]+)\}',p.read_text())];refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
stack=[];errors=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',tex):
 if m[1]=='begin':stack.append(m[2])
 elif not stack or stack.pop()!=m[2]:errors.append(m[0])
assert not stack and not errors
# Basic brace validation, ignoring comments and escaped literal braces.
clean=re.sub(r'(?m)^%.*$','',tex);depth=0
for m in re.finditer(r'(?<!\\)[{}]',clean):
 depth+=1 if m[0]=='{' else -1
 assert depth>=0
assert depth==0
figs=[]
for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',tex,re.S):
 f={'caption':re.search(r'\\caption\{([^}]+)\}',m[0])[1],'label':re.search(r'\\label\{([^}]+)\}',m[0])[1],'image_files':re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',m[0]),'placeholder':'图位' in m[0]};figs.append(f)
for f in figs:
 for p in f['image_files']:assert (R/'thesis'/p).is_file(),p
tables=[{'caption':re.search(r'\\caption\{([^}]+)\}',m[0])[1],'label':re.search(r'\\label\{([^}]+)\}',m[0])[1]} for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',tex,re.S)]
assert not set(cites)-keys and not set(refs)-set(alllabels)
assert all(alllabels.count(x)==1 for x in labels)
# Verify all quantitative anchors against the approved audit, not model estimates.
assert [(x['actual_episode_indices'],x['actual_frames']) for x in data['lerobot']]==[(5,1550),(30,14310),(1,340)]
v4=[x for x in data['npz_cases'] if x['kind']=='packaged_v4_development'];assert len(v4)==5 and sum(x['frames'] for x in v4)==2727
assert len({x['lineage_hash'] for x in v4})==5 and len({x['asset_hash'] for x in v4})==4 and len({x['content_hash'] for x in v4})==5
assert all(all(x['registered_file_hashes_match'].values()) and not x['missing_registered_files'] for x in v4)
oldpick=next(x for x in data['npz_cases'] if x['path'].endswith('pick_tomato_rgb_00'))
assert oldpick['pre_action_command_mismatch_row_count']==373 and abs(oldpick['pre_action_command_max_error']-.1)<1e-6
assert all(data['fixed_export_numeric_check'][k]==0 for k in ['action_vs_source_ctrl_max_abs','state_command_vs_previous_ctrl_max_abs','tactile_vs_previous_source_row_max_abs'])
assert abs(data['historical_v4_comparison']['legacy_replay']['max_qpos_component_error']-1.628897)<1e-6
assert data['historical_v4_comparison']['exact_replay']['max_qpos_component_error']==0
# Reuse reading converters only, without executing old packaging side effects.
ns={'re':re}
for f in ast.parse((R/'docs/writing_audit/ch02_writing_v1_2026-11-20/package_delivery.py').read_text()).body:
 if isinstance(f,ast.FunctionDef) and f.name in ['inline','fig','tab']:exec(compile(ast.Module(body=[f],type_ignores=[]),'reading_converter','exec'),ns)
read=re.sub(r'^%.*\n','',tex,flags=re.M);read=re.sub(r'\\chapter\{([^}]+)\}',r'# 第四章 \1',read);n=[0]
def sec(m):n[0]+=1;return f'## 4.{n[0]} {m[1]}'
read=re.sub(r'\\section\{([^}]+)\}',sec,read);read=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',ns['fig'],read,flags=re.S);read=re.sub(r'\\begin\{table\}.*?\\end\{table\}',ns['tab'],read,flags=re.S)
read=ns['inline'](read.replace('\\begin{equation}','\n$$').replace('\\end{equation}','$$\n'))
(O/'CH04_READING_DRAFT.md').write_text('第四章V1正文完成，等待用户与ChatGPT审核。由最终TeX自动转换；公式及引用目标保留，未编译或冻结。\n\n'+re.sub(r'\n{3,}','\n\n',read).strip()+'\n')
wc=json.loads((O/'wordcount/THESIS_WORD_COUNT.json').read_text());ch=wc['chapters'][3];m=ch['metrics'];assert 7200<=m['total']<=7700
budget=[1050,1400,950,1600,2075,425]
lines=['# 第四章实际TeXcount','','引擎：'+wc['engine'],'','源文件：`thesis/body/ch04_multimodal_data.tex`。正文统计采用现有中文规则，未以模型估算替代。','','|小节|实际统计量|本轮中心目标|差值|','|---|---:|---:|---:|']
for s,b in zip(ch['sections'],budget):lines.append(f"|{s['id']} {s['title']}|{s['actual_count']}|{b}|{s['actual_count']-b:+d}|")
lines+=['',f"全章 **{m['total']}**，符合本轮7200–7700目标；相对7500中心为{m['total']-7500:+d}。统计量=text {m['text']} + 标题 {m['headings']} + 图表题/脚注 {m['captions_footnotes']}。章标题另计，不包含在小节合计中。",'',f"表格文字及流程框内文字按现有规则纳入text；图表标题计入caption。行内公式{m['inline_math']}、独立公式环境{m['display_math']}另报，不折算正文；无脚注或代码块。",'', '已执行--chapter 4和--files，记录见count_chapter.log/count_files.log。--files按正式主文件旧include结构列库存，新第四章按V2规划路径统计；没有修改include关系。未编译，PDF页数UNKNOWN。', '', '现有脚本总体规划章目标仍显示7000–8000，本轮交付另以用户收紧的7200–7700检查，未修改预算工具或规划。TeXcount自测全部通过。各节篇幅按实际内容调整，未为了单节落点扩写重复论述。']
(O/'WORDCOUNT_REVIEW.md').write_text('\n'.join(lines)+'\n')
(O/'EVIDENCE_CITATION_MAP.md').write_text('''# 第四章论点、证据与图表对应

本章以已审实现和指定数据核验为依据，没有新增或强行插入外部文献，不修改reference.bib。内部来源均可由材料审计目录source_index.json、dataset_checks.json及integrity.json追溯函数、字段与SHA256。正文不以代码文件数量作为贡献。

|小节|论点/对象|代码或数据依据|证据范围|
|---|---|---|---|
|4.1|state39、两种action13、tactile712及对象状态|source/imitation/observation.py；production_recorder.py；三LeRobot和11NPZ字段|L2所列合同与字段，不等同实机标定|
|4.2|执行前采集、实际动作与缓存|production_recorder.py::OnlineEnvironment/recorder_type|L2当前实现，不回填旧数据|
|4.2|RGB与控制降频对应|production_capture.py及production_video.py|L2实现，默认/速度配置不冒充服务器实测|
|4.3|pre-action索引与低层ctrl|collect_generated_lerobot.py::recorded_pre_action_indices/_export_one；P2固定340行|L2指定数值字段，非全部模态认证|
|4.3|P31移位与语义修复的区别|five_task_dataset.py::write_causal_source|L2代码；移位不自动转换IK动作|
|4.4|结构、时序、来源与隔离|production_recorder.py、production_artifacts.py、production_queue.py|L2机制，非全库实际运行保证|
|4.4/4.5|旧PickPlace373转移及旧Stack缺字段|dataset_checks.json::npz_cases及所列源NPZ|L2本轮材料复核；不推断唯一原因，不声称已修复|
|4.5|三LeRobot的5/1550、30/14310、1/340及非零率|dataset_checks.json::lerobot，全部所选数值Parquet|L2限定目录，RGB未解码，独立Root未知|
|4.5|五v4包2727行、5 lineage、4 asset、5 content|dataset_checks.json::npz_cases，五包episode.json及登记文件|L2逐项字段/文件Hash检查，不认证独立Root或正式验收|
|4.5|三个指定340行映射最大差0|dataset_checks.json::fixed_export_numeric_check|L2；只比较action、state命令区和指定触觉|
|4.5|Sink两路径误差1.628897/0|outputs/five_task_p38_exact_replay_2026-11-03/comparison.json|L1历史回放成绩，读取源辅助字段为L2，无本轮L3|
|4.5|Real/Natural冻结角色与规模|outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/data_contract.csv|L1历史合同声明，未读Real全库或Test/Holdout数组|

## 图表来源

图4-1数据载体、图4-2采样时序、图4-3多速率采集、图4-4质量准入均为根据实际代码组织的流程示意，不是实验截图。图4-5由comparison.json中两条已存误差绘制：旧路径1.6288973093、完整状态+float64路径0，图中保留六位显示。没有补造曲线，零值直接标注，不以极小正数替代。

plot_sink_comparison.py仅生成统计图；输入Hash和原字段保存在figure_provenance.json。图像资产为thesis/figures/ch04/sink_historical_precision.pdf及PNG。所有5张图均已提供，无待补图位；未运行物理仿真。

六表依次为字段合同、记录版本、质量门槛、LeRobot库存、旧/v4数据对照及Sink历史结果。任务标签是lift:物体，不将task_index计作多任务数量；三个库不能加成36个独立Root。

## 四个公式的实现对应

- eq:ch04_causal_pairing：在线pre-action或连续旧post-action行的对应，明确成立条件，不能无条件修复隐藏重置。
- eq:ch04_time_interval：0.05s控制/转移时间合同，表示浮点容差而非逐位时间相等。
- eq:ch04_mapping_error：最大绝对误差，对应审计脚本np.max(abs(...))。
- eq:ch04_duplicate：production_queue.register查询lineage相等或content相等，逻辑为OR，不是AND。

没有引入不存在的同步估计、语义对齐或无泄漏保证。
''')
(O/'WRITING_REVIEW.md').write_text('''# 第四章V1写作自审

状态：**第四章V1正文完成，等待用户与ChatGPT审核。** 未编译、未冻结，不进入第五章。

- 六节按合同—采样关联—转换—分层质量—实际验证—局限组织，未重复第二章传感器原理和第三章抓取/Strict算法。
- state39是受控执行器位置、速度、上一命令；IK13与Joint13区分，object_state可含目标site，processed触觉不写成Pa或真实压力。
- 当前修复代码、单样本重导出和已验证v4开发包分别表述。旧PickPlace373条关系错误与旧Stack缺字段保留，不推断唯一原因、不宣称已经修复。
- 三组LeRobot、v4五包、Sink历史回放和冻结Real/Natural合同分别注明统计单位与证据来源，不合并为新物理规模。Real合同为L1，不包装为本轮Real全库核验。
- 映射0误差限定到对应340行和指定字段；Hash匹配不等于Root独立；视频存在不等于本轮完整解码同步验收。
- 未宣称服务器全量规模、全局无泄漏或正式生产指标通过。未使用重复索引时长累计新增物理小时。
- 5图、6表、4公式；一张图采用历史数值，其他为实现示意，不是新实验图像。公式和数值锚点静态检查见static_checks.json。
- 未新增外部引用，无BibTeX修改；所有已有源文件、第一至第三章及冻结PDF、主文件、模板、数据和代码Hash保持不变。

## 待人工审核及后续排版问题

本轮不编译，公式宽度、流程框高度、浮动体位置与表格断行仍需内容审核后检查。英文图轴标注可在最终图表统一时处理，不因此重新生成实验数据。

4.3和4.6略低于中心目标，4.4略高，但全章满足7200–7700；保留实际论证密度，不重复扩充来机械配平。文中UNKNOWN与历史声明限制应在审核时保留。

本轮仅创建第四章及必要图表、交付文件，未运行训练、仿真或服务器任务，未重建数据集，不commit/push。
''')
checks={'status':'第四章V1正文完成，等待用户与ChatGPT审核','source_path':str(P.relative_to(R)),'texcount':m['total'],'sections':[{k:x[k] for k in ['id','title','actual_count']} for x in ch['sections']],'figures':figs,'tables':tables,'equation_labels':re.findall(r'\\label\{(eq:[^}]+)\}',tex),'citation_keys':cites,'missing_citation_keys':sorted(set(cites)-keys),'missing_reference_targets':sorted(set(refs)-set(alllabels)),'duplicate_labels':[x for x in labels if alllabels.count(x)>1],'environment_errors':errors,'braces_balanced':True,'data_numeric_anchors_match_audit':True,'changed_existing_protected_files':changed,'source_data_code_hashes_match_material_audit':True,'formal_main_unchanged':True,'reference_bib_unchanged':True,'chapters_1_2_3_unchanged':True,'compiled':False,'frozen':False}
(O/'static_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['OPENBLAS_NUM_THREADS=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch04_writing_v1_2026-11-20/plot_sink_comparison.py','./scripts/count_thesis_words.sh --chapter 4 --output-dir docs/writing_audit/ch04_writing_v1_2026-11-20/wordcount','./scripts/count_thesis_words.sh --files --output-dir docs/writing_audit/ch04_writing_v1_2026-11-20/wordcount','python3 docs/writing_audit/ch04_writing_v1_2026-11-20/package_ch04.py'],'no_compilation':True},indent=2)+'\n')
(O/'README.md').write_text('''# 第四章V1交付

正式新增源文件：thesis/body/ch04_multimodal_data.tex。本目录同名文件是本轮快照。

CH04_READING_DRAFT.md为完整阅读稿，WORDCOUNT_REVIEW.md为真实统计；EVIDENCE_CITATION_MAP.md记录证据、图表与公式；WRITING_REVIEW.md说明事实边界和剩余问题；changes.diff只包含新增章节；static_checks.json及SHA256SUMS保存检查。

状态：第四章V1正文完成，等待审核。未编译、未冻结，不修改主文件include关系，不进入第五章。
''')
paths=[P]+[x for x in O.iterdir() if x.is_file() and x.name!='SHA256SUMS']+list((R/'thesis/figures/ch04').iterdir())
(O/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(R)}\n' for p in sorted(set(paths)) if p.is_file()))
print('Packaged',m['total'],'words;',len(figs),'figures;',len(tables),'tables;',len(checks['equation_labels']),'equations. Protected files unchanged.')
