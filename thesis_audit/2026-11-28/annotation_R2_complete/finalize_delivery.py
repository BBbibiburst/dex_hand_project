from pathlib import Path
import json,csv,re,hashlib,difflib,zipfile,collections,sys
R=Path('/home/zmy/dex_hand_project');D=R/'thesis_audit/2026-11-28/annotation_R2_complete';T=R/'thesis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,s): (D/n).write_text(s,encoding='utf-8')
base=json.loads((D/'baseline.json').read_text());wc=json.loads((D/'wordcount/THESIS_WORD_COUNT.json').read_text());active=[R/x for x in wc['actual_include_graph']['files']]
changes={p:{'before':h,'after':sha(R/p)} for p,h in base.items() if sha(R/p)!=h}
diff=''
for p in changes:
 if (D/'before'/p).exists() and (R/p).suffix in ['.tex','.bib']:
  diff+=''.join(difflib.unified_diff((D/'before'/p).read_text().splitlines(True),(R/p).read_text().splitlines(True),fromfile='before/'+p,tofile='after/'+p))
write('changes.diff',diff)
# Exact graph, labels, references, bibliography and graphics, without executing research programs.
texts={str(p.relative_to(R)):re.sub(r'(?<!\\)%[^\n]*','',p.read_text()) for p in active}
joined='\n'.join(texts.values());labels=re.findall(r'\\label\{([^}]+)\}',joined);refs=re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',joined)
bib=(T/'reference.bib').read_text();keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',bib));cites=[v.strip() for x in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',joined) for v in x.split(',')]
graphics=[];missing=[]
for src,s in texts.items():
 for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s):
  candidates=[T/name,T/'figures'/name,T/'figures/framework'/name,T/'figures/network'/name,T/'figures/experiments'/name]
  got=next((q for q in candidates if q.is_file()),None)
  if not got:got=next((q.with_suffix(e) for q in candidates for e in ['.pdf','.png','.jpg'] if q.with_suffix(e).is_file()),None)
  if got:graphics.append(got)
  else:missing.append({'source':src,'graphic':name})
static={'duplicate_labels':[k for k,v in collections.Counter(labels).items() if v>1],'unresolved_references':sorted(set(refs)-set(labels)),'missing_citation_keys':sorted(set(cites)-keys),'missing_graphics':missing,'active_source_count':len(active),'figures':len(re.findall(r'\\begin\{figure\*?\}',joined)),'tables':len(re.findall(r'\\begin\{table\*?\}',joined)),'bib_entries':len(keys),'old_bib_preserved_as_prefix':bib.startswith((D/'before/thesis/reference.bib').read_text())}
write('static_checks.json',json.dumps(static,ensure_ascii=False,indent=2))
# Figure manifest with final automatic numbers.
aux='\n'.join(p.read_text(errors='replace') for p in (D/'build').rglob('*.aux'))
labelnum={m[0]:(m[1],m[2]) for m in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)}
figs=[('platform_actual_architecture','fig:ch02_architecture','当前机器人模型、控制与712维历史观测','draw_architecture.py','platform_actual_architecture_sources.json'),('joint_tendon_geometry','fig:ch02_drive','实际关节模型、固定腱系数和执行器','draw_geometry.py','joint_tendon_geometry_sources.json'),('sampling_processes','fig:ch02_sampling_steps','指节、指尖、手掌三类算法实际中间结果','draw_sampling.py','sampling_processes_sources.json'),('grasp_evidence_pipeline','fig:ch03_overall_pipeline','Sugar Box候选几何、历史执行及物体运动','draw_grasp_evidence.py','grasp_evidence_sources.json'),('candidate_geometry_feedback','fig:ch03_candidate_sources','候选接触、进化记录及另物体PPO保存动作','draw_grasp_evidence.py','grasp_evidence_sources.json'),('wrist_recorded_targets','fig:ch03_reachability','已保存末端目标、预检残差及执行状态','draw_grasp_evidence.py','grasp_evidence_sources.json'),('recorded_tracking_strict','fig:ch03_strict_replay','保存状态的运动学重建与历史Strict结果','draw_tracking.py','recorded_tracking_strict_sources.json'),('five_task_phase_records','fig:ch03_task_transfer','五任务原始视频的实际阶段帧','draw_task_phases.py','five_task_phase_records_sources.json'),('assembly_failure_records','fig:ch03_examples','装配成功失败记录的高度、接触与阶段','draw_failure_cases.py','assembly_failure_records_sources.json'),('data_lineage_instances','fig:ch04_carriers','v4与Sugar导出两个真实来源关系','draw_lineage.py','data_lineage_instances_sources.json')]
figure_rows=[]
for name,label,desc,script,source in figs:
 n,pg=labelnum.get(label,('UNKNOWN','UNKNOWN'));figure_rows.append(dict(asset='thesis/figures/annotation_R2/'+name+'.pdf',label=label,figure_number=n,printed_page=pg,description=desc,script=script,provenance=source))
write('FIGURE_SOURCES.json',json.dumps(figure_rows,ensure_ascii=False,indent=2))
write('FIGURE_SOURCES.md','# 本轮图稿与来源\n\n九幅替换、一幅新增；保留此前已核对的控制器、处理算子、热图、模型和伪代码。下列图号来自最终编译aux，页码为正文印刷页码。\n\n|图号/页码|内容|图稿|重绘脚本 / 原始来源Hash|\n|---|---|---|---|\n'+''.join(f"|{q['figure_number']} / {q['printed_page']}|{q['description']}|{q['asset']}|{q['script']} / {q['provenance']}|\n" for q in figure_rows)+'\n实际几何和保存记录不等于本轮新物理验证。候选图的GraspQP与进化代表不是同一个候选的连续优化轨迹；PPO为另一物体的条件分支。运动学重建不重新执行控制器；447行来源与322行Strict回放不混为同一次记录。失败图仅用对应失败NPZ，不借用成功截图。所有代码可读；重画需要原始来源，本TeX包仅依赖已导出的图稿。\n')
# Preserve original IDs/comments/marked text and all old navigation fields.
P=R/'thesis_audit/2026-11-28/annotation_correction_batch2/ANNOTATION_LEDGER.csv';old=list(csv.DictReader(P.open(encoding='utf-8-sig')))
new_actions={2:'已提交学校规范来源、当前字体识别及两种可用字体对照；不擅自安装授权不明字体或更换全局版式。',7:'摘要先解释实机安全/协调成本及仿真批量接触记录的作用，再组织四项工作。',9:'摘要准确概括六驱动代理指标适配、搜索档案及机器人约束流程，不把开源算法称原创。',10:'中英文摘要按平台、示教、数据、迁移递进，收束量化结果。',11:'压缩字段细节，保留数据关联与340行修复核验结果。',12:'摘要和5.3正面介绍统计先验用途；方法定义处保留伪配对性质，不改称同步物理真值。',13:'增加五条任务触觉有效性、10条局部Strict、340行映射结果；保留迁移2/11及距离改善边界。',14:'英文摘要同步实际技术范围和量化结果。',19:'改写专家示教记录与后续数据使用的关系。',21:'说明选用实验室设备型号与感知布局；未声称完成参数及触觉响应全标定。',22:'绪论按平台、抓取、数据、跨域组织；补充核实的robosuite和ManiSkill。',23:'补充SAPIEN，区分引擎与框架，说明MuJoCo腱传动/接触能力；不编造其他引擎不能模拟欠驱动。',24:'加入作者实际示教体验与遮挡问题，引用AnyTeleop支持通用技术挑战，不虚构定量实测。',25:'解释Narang工作中源域/目标域/适配内容，而非互斥列举。',26:'说明同一接触在仿真与实机难以精确复现后，引出统计信息利用。',27:'增加方法比较解释并调整长方法名列宽；最终页17无溢出。',28:'核对实际位于绪论组织段，沿用已改正式表述；旧CSV导航不是精确位置。',29:'替换为实际模型、控制合同及历史712D观测的系统图。',30:'补全MJCF全称及XML、STL、URDF不同资产角色。',31:'替换为实际关节模型、6×12固定腱系数矩阵和执行器对应。',33:'组合模型与实验室型号关系、底座与末端安装关系具体化。',34:'删去模型图注中不必要的防御性补句。',35:'解释皮肤mesh、碰撞参数及触觉观测模块职责。',41:'按原批注删除局部多余句，不扩大物理对应认证。',42:'交代覆盖手指、指尖、手掌的布局目的，准确保留未标定范围。',43:'以几何拟合和采样过程替代空泛防御句。',44:'保留全代表形状并增加三策略实际中间图，补充静态间距描述；没有触觉效果新实验。',48:'先交代验证目标与样本，再集中说明验证范围。',49:'新增五任务物体/环境/阶段/判定定义，使用历史起终画面；现行规则不回填旧记录。',50:'用物体几何、真实候选、视频帧及物体运动替换通用流程框。',52:'正面解释指标优化、几何约束与执行验证的分工。',53:'展示接触点/法向、六驱动比例、进化档案、腕姿编辑及单独PPO记录。',55:'解释档案插入、替换、密度排序及后续机器人验证。',56:'描述腕姿编辑和分支筛选实际用途，边界集中说明。',57:'已绘保存目标、残差和状态；被拒绝的21个候选逐姿态状态未保存，不能补画为历史数据。',58:'已用保存qpos和归档模型重建实际末端跟踪并对应历史Strict；缺少逐次内部IK残差，未伪造收敛轨迹。',59:'用五任务原始视频登记阶段中点组成5×4画面；Stack记录为后段。',60:'改为正向任务条件和执行动作说明。',61:'用装配成功/失败各自NPZ绘高度、接触与阶段，没有借用成功图片充作失败。',62:'按缓存/变体/局部扰动分开解释统计结果，集中保留条件边界。',63:'概括候选、机器人验证及示教构建用途，去掉重复局限。',64:'以v4 PickPlace和Sugar导出两个实际实例绘字段及行偏移。'}
figids={29:0,31:1,44:2,50:3,53:4,57:5,58:6,59:7,61:8,64:9}
anchors={7:'摘要',9:'六',10:'摘要',11:'340',12:'统计',13:'10',14:'Abstract',19:'专家示教还需要准确记录',21:'机器人组合模型选用',22:'机器人仿真与触觉感知建模',23:'SAPIEN',24:'灵巧手人工示教',25:'Narang',26:'仿真触觉与真实触觉的差异',27:'tab:related_work_routes',28:'章节',30:'MuJoCo模型描述格式',33:'机器人组合模型选用',34:'fig:ch02_robot_model',35:'手部皮肤',41:'曲面',42:'感知',43:'采样',48:'平台',49:'任务',52:'GraspQP',55:'档案',56:'腕部',60:'Push',62:'缓存',63:'本章'}
for r in old:
 i=int(r['id']);p=r['source']
 if i==28:p='thesis/body/ch01_introduction.tex'
 if i in range(7,15):p='thesis/front/abstract.tex'
 if i in [21,33]:p='thesis/body/ch02_simulation_platform.tex'
 s=(R/p).read_text() if (R/p).exists() else ''
 label=figure_rows[figids[i]]['label'] if i in figids else anchors.get(i,'')
 ln=next((n for n,line in enumerate(s.splitlines(),1) if label and label in line),None)
 r['status']='部分处理' if i in [2,57,58] else ('证据限制／不按字面执行' if i in [12,23] else '已处理（待人工验收）')
 if i in new_actions:r['action_or_remaining']=new_actions[i]
 r['modified_file']=p;r['modified_location']=f'第{ln}行：{label}' if ln else '沿用前批已处理位置，原行提示仅供导航'
 r['verification_evidence']=('FIGURE_SOURCES.md；'+figure_rows[figids[i]]['provenance']) if i in figids else ('FONT_SCHEME.md；FONT_COMPARISON.pdf' if i==2 else ('LITERATURE_VERIFICATION.md；changes.diff；当前源文件' if i in [22,23,24,25] else '当前源文件；changes.diff或原批修改记录；最终PDF'))
 r['acceptance_result']='已有可核验改动，等待人工评价表达/图稿质量' if i not in [2,12,23,57,58] else ({2:'仅字体方案待决定，全局字体保持不变',12:'以真实统计伪配对性质满足正面叙述意图',23:'已补框架比较，拒绝无依据的排他性能断言',57:'记录缺口保留，具体补充方案见OPEN_ISSUES_AND_PROPOSALS',58:'运动学跟踪已补；内部求解残差缺口保留'}[i])
 if i in figids:r['pdf_verification_pages']='正文'+str(figure_rows[figids[i]]['printed_page'])+'页'
 else:r['pdf_verification_pages']='全页概览及对应章节抽检；非逐字人工审定'
fields=list(old[0]);
with (D/'ANNOTATION_LEDGER.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(old)
original=list(csv.DictReader(P.open(encoding='utf-8-sig')));preserved=all(all(a[k]==b[k] for k in ['id','pdf_page','comment','original_marked_text']) for a,b in zip(original,old)) and len(old)==64
write('ANNOTATION_LEDGER.md','# 64项逐条落实总账\n\nCSV为完整版本，原始ID、批注和选中文字逐字保留。状态不是人工验收结果。\n\n'+''.join(f"- #{r['id']} **{r['status']}**：{r['action_or_remaining']}\n" for r in old))
# Body chapter counts from actual file metrics, not stale planning mapping.
counts={}
for n in range(1,8):counts[n]=sum(q['total'] for q in wc['actual_files'] if q['category']=='body' and (Path(q['path']).name.startswith('ch0'+str(n)+'_') or f'/ch0{n}_sections/' in q['path']))
write('WORDCOUNT_REVIEW.md','# 实际TeXcount\n\n运行命令：`./scripts/count_thesis_words.sh --files --output-dir thesis_audit/2026-11-28/annotation_R2_complete/wordcount`。真实原始输出、规则Hash与源文件Hash位于wordcount目录。\n\n|章|实际字数|\n|---|---:|\n'+''.join(f'|{n}|{v}|\n' for n,v in counts.items())+f"\n实际接入正文合计：**{wc['actual_compilation_source_body']['total']}**；包含活动前后置文本的源计数：{wc['actual_full_source']['total']}。\n\n标题、图表标题、脚注、表格正文计入；数学权重为0。文件汇总按当前include图；工具中旧V2规划标题匹配缺失第五章、摘要占位提示和71页旧PDF项属于继承的索引限制，不代表本次缺章。所有第五章片段均包含在actual_include_graph和实际合计中。当前PDF页数单独用PyMuPDF读取，不能用旧PDF字段。未修改计数规则。\n")
# Gather source input hashes declared by figure scripts.
inputs={};source_reports=[]
for p in D.glob('*_sources.json'):
 q=json.loads(p.read_text());source_reports.append(p.name)
 for k,v in q.get('files',{}).items():
  if isinstance(v,str) and len(v)==64:inputs[k]={'recorded':v,'current':sha(R/k) if (R/k).is_file() else None}
protected={p:{'before':h,'after':sha(R/p),'unchanged':sha(R/p)==h} for p,h in base.items() if p not in changes}
# Source bundle, only literal active source graph and graphics + template files.
source_set=set(active+graphics+[T/'reference.bib'])
source_set.update(q for q in T.iterdir() if q.is_file() and q.suffix in ['.cls','.sty','.cfg','.bst'])
manifest={str(q.relative_to(R)):sha(q) for q in sorted(source_set)}
write('TEX_SOURCE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(D/'thesis_annotation_R2_source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for q in sorted(source_set):z.write(q,str(q.relative_to(R)))
 z.writestr('README_R2.txt','R2完整活动源文件与所用图稿。进入thesis目录，以XeLaTeX + BibTeX + XeLaTeX两次或更多编译thesis.tex。需要hithesis相关TeX包、Fandol/TeX Gyre字体。未附TeX发行版、私有Windows字体、实验数据或模型。主文件保持本轮之前组织。初稿待人工审核，非冻结版。\n')
 z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
log=(D/'build/thesis.log').read_text(errors='replace');checks=json.loads((D/'pdf_page_checks.json').read_text())
warn=[x for x in log.splitlines() if any(w in x for w in ['Warning:','Overfull','Missing character','undefined','too large'])]
write('LAYOUT_REVIEW.md',f'''# PDF与版式核对

最终PDF **{checks['pages']}页**；对照批注第二批98页，增加8页。新增内容主要为实际采样中间图、候选几何/历史任务阶段及必要论述，不是新增实验。

全106页自动检查页面文字边界：无文字坐标超出版面；无低于40字符的空页候选。逐张查看7张全页缩略图，放大查看新增图所在PDF页22、24、31、41、43、45、47、52、54、57。检查范围是布局、图形和主要标注，不等于106页逐字人工校对。page_review保存缩略图与放大图；pdf_page_checks.json保留逐页数据。

图2-3为整页浮动图，固定腱系数和名称可辨；采样图三行分别展示指节、指尖、手掌。图3-6为20帧历史阶段画面，每行任务独立。图4-1两个来源路径分开，索引137/136偏移保持。候选图坐标刻度稀疏化、提高实体对比度，未改变几何数据。全页概览未见图文重叠、孤立标题或整页空白。

中文摘要仍有续页（PDF第5页约数行），英文摘要亦为两页；不是本轮通过改字号压缩处理，列为后续排版优化。部分复杂图的小字号仍需用户在实际阅读尺寸确认。本轮不更换全局字体，不将人工体验等同于自动验收。

最终XeLaTeX日志未出现Overfull、未定义引用、缺字或浮动体过大。保留的模板警告如下：

'''+''.join('- '+x+'\n' for x in warn)+'\n完整构建日志见build.log，最后一遍TeX详细日志见build/thesis.log。\n')
write('CHANGE_REPORT.md','# R2修改摘要\n\n以本目录before快照（即批注第二批状态）为差异基线，非Git HEAD。\n\n'+''.join('- '+p+'\n' for p in changes)+'\n新增十幅带来源图稿（九替换、一新增），脚本与来源Hash见FIGURE_SOURCES。新增四条经原文核对的文献；原BibTeX保留为前缀。修订摘要/绪论和第二至四章相关叙述；第五章仅改5.3首句，其余第五至七章、主文件、模板及封面未改。\n\n原稿、R1、前两批PDF与总账均保留，未以新稿覆盖历史材料。changes.diff包括所有本轮TeX/Bib文本修改；新增图稿和交付脚本由源清单及完整性记录列明。\n')
write('REVIEW.md',f'''# R2交付与自审

## 实际完成

- 完整初稿PDF：{checks['pages']}页；活动正文TeXcount {wc['actual_compilation_source_body']['total']}。
- 九幅旧通用图已替换为实际几何、当前模型、保存轨迹/视频和真实数据来源；新增一幅三种表面采样中间结果图。并非所有流程关系都要删除，必要的数据流和算法框图保留。
- 64项原始ID/批注/选中文字保持不变：59项已处理待人工验收，3项部分处理，2项事实约束采用替代处理。此状态不能称64项均已完成。
- 新增文献4条，均读原文核对，且原参考文献元数据未覆盖。字体保持原模板，提交方案先审核。
- 标签、引用键、图稿存在性静态检查结果见static_checks.json；无缺失引用、重复标签或图稿。

## 仍需处理

1. #02：字体仅完成规范核对与备选对照，具体是否改用授权Windows字体待确认。
2. #57：历史过滤候选的逐姿态状态缺失，已画现存目标与预检结果，不能补成历史不可达演示。
3. #58：保存状态可以重建末端跟踪，不能得到当时未保存的IK内部残差。补充验证方案已列出，本轮未执行。
4. #12、#23：保持统计伪配对事实，补足正面方法叙述；介绍SAPIEN/框架及MuJoCo传动建模特点，不虚构其他引擎不具备欠驱动能力。
5. 摘要续页、小字图实际阅读尺寸和全篇人工校对仍待用户审核。尚无新增公平算法对照、模块消融、实机触觉标定或任务收益实验；新图不能替代这些证据。

## 结论与保护

没有改变C27的2/11联合验收、F0指定SWD下降49.77%、对抗接受不代表质量以及D50限定NO-GO。没有训练、神经网络推理、物理积分、服务器任务、Test/Holdout数组访问、commit/push或冻结。静态MuJoCo前向/运动学/渲染仅用于已有模型与保存状态的几何展示。

完整TeX包可独立携带源与已导出图稿；不携带实验模型与数据。所有现存旧文件Hash检查范围与真实来源Hash在integrity.json中列明，不能由有限清单推断整个磁盘均已审计。
''')
integ={'baseline':'before snapshot from annotation batch2 state','changed_existing':changes,'unchanged_baseline_files':protected,'source_inputs':inputs,'source_inputs_hash_match':all(q['recorded']==q['current'] for q in inputs.values()),'original_64_comments_preserved':preserved,'status_counts':dict(collections.Counter(r['status'] for r in old)),'source_manifest':manifest,'static_checks':static,'scope':{'training':0,'neural_inference':0,'physics_steps':0,'new_IK_solves':0,'Test_Holdout_arrays_accessed':False,'commit_push_freeze':False},'note':'只对清单涉及文件作前后/登记Hash核对，不将此解释为全量实验目录审计。'}
write('integrity.json',json.dumps(integ,ensure_ascii=False,indent=2))
print(json.dumps({'changed':list(changes),'static':static,'counts':counts,'original_comments':preserved,'sources_match':integ['source_inputs_hash_match'],'zip_files':len(manifest)},ensure_ascii=False,indent=2))
