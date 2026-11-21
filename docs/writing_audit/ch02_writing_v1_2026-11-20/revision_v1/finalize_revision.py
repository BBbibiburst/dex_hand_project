from pathlib import Path
import ast,re,json,hashlib,difflib,shutil
import pymupdf as fitz
R=Path.cwd();O=R/'docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1';P=O.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tex=(R/'thesis/body/ch02_simulation_platform.tex').read_text();old=(O/'chapter_before.tex').read_text()
shutil.copyfile(R/'thesis/body/ch02_simulation_platform.tex',O/'ch02_simulation_platform.tex')
(O/'technical_changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),tex.splitlines(True),fromfile='chapter_before.tex',tofile='thesis/body/ch02_simulation_platform.tex')))
# Reuse only conversion function definitions; do not execute the V1 packager or overwrite V1 records.
src=ast.parse((P/'package_delivery.py').read_text());ns={'re':re}
for f in src.body:
 if isinstance(f,ast.FunctionDef) and f.name in ['inline','fig','tab']:
  exec(compile(ast.Module(body=[f],type_ignores=[]),'reading_conversion','exec'),ns)
read=re.sub(r'^%.*\n','',tex,flags=re.M);read=re.sub(r'\\chapter\{([^}]+)\}',r'# 第二章 \1',read)
i=[0]
def section(m):i[0]+=1;return f'## 2.{i[0]} {m[1]}'
read=re.sub(r'\\section\{([^}]+)\}',section,read)
read=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',ns['fig'],read,flags=re.S)
read=re.sub(r'\\begin\{table\}.*?\\end\{table\}',ns['tab'],read,flags=re.S)
read=read.replace('\\begin{equation}','\n$$').replace('\\end{equation}','$$\n');read=ns['inline'](read)
(O/'CH02_READING_DRAFT.md').write_text('第二章修订V1完整阅读稿。保留公式及引用键供审核，正式版式见同目录PDF预览；本稿不代表正式冻结。\n\n'+re.sub(r'\n{3,}','\n\n',read).strip()+'\n')
log=(O/'build/ch02_preview.log').read_text(errors='replace');blg=(O/'build/ch02_preview.blg').read_text(errors='replace')
shutil.copyfile(O/'build/ch02_preview.pdf',O/'chapter02_preview_with_references.pdf')
d=fitz.open(O/'build/ch02_preview.pdf');texts=[p.get_text() for p in d]
refpage=next(i for i,t in enumerate(texts) if t.startswith('哈尔滨工业大学硕士学位论文\n参考文献'))
body=fitz.open();body.insert_pdf(d,from_page=0,to_page=refpage-1);body.save(O/'chapter02_preview.pdf');body.close()
fontrecords={}
for page in d:
 for f in page.get_fonts():
  if f[0] not in fontrecords:
   fdata=d.extract_font(f[0]);embedded=bool(fdata[3]);method='font program extraction';glyphs=0
   if not embedded and d.xref_get_key(f[0],'Subtype')[1]=='/Type3':
    ref=d.xref_get_key(f[0],'CharProcs')[1];match=re.match(r'(\d+) 0 R',ref)
    if match:
     charprocs=d.xref_object(int(match[1]));glyphs=sum(d.xref_is_stream(int(x)) for x in re.findall(r'(\d+) 0 R',charprocs));embedded=glyphs>0;method='embedded Type3 CharProcs streams'
   fontrecords[f[0]]={'xref':f[0],'name':f[3],'embedded_bytes':len(fdata[3]),'embedded':embedded,'verification':method,'embedded_type3_glyphs':glyphs}
page_records=[]
for n,page in enumerate(d):
 t=texts[n];words=page.get_text('words');outside=[w[:4] for w in words if w[0]<-1 or w[1]<-1 or w[2]>page.rect.width+1 or w[3]>page.rect.height+1]
 page_records.append({'pdf_page':n+1,'text_characters':len(t),'replacement_characters':t.count('\ufffd'),'blank':len(t.strip())<25,'words_outside_page':outside,'header_present':'哈尔滨工业大学硕士学位论文' in t})
 page.get_pixmap(matrix=fitz.Matrix(.65,.65)).save(O/f'page_{n+1:02d}.png')
for index in [3,8,9]:d[index].get_pixmap(matrix=fitz.Matrix(1.35,1.35)).save(O/f'detail_page_{index+1:02d}.png')
(O/'pdf_checks.json').write_text(json.dumps({'preview_pages':len(d),'body_pages':refpage,'reference_pages':len(d)-refpage,'chapter_number':2,'printed_body_page_range':[1,refpage],'temporary_standalone_pagination':True,'pages':page_records,'fonts':list(fontrecords.values()),'all_fonts_embedded':all(f['embedded'] for f in fontrecords.values()),'undefined_reference_or_citation':bool(re.search(r'undefined|multiply defined',log,re.I)),'overfull_boxes':re.findall(r'Overfull[^\n]*',log),'underfull_boxes':re.findall(r'Underfull[^\n]*',log),'bibtex_warnings':[l for l in blg.splitlines() if l.startswith('Warning--')],'no_formal_freeze':True},ensure_ascii=False,indent=2)+'\n')
protected=json.loads((O/'protected_before.json').read_text());changed=[p for p,h in protected.items() if sha(R/p)!=h]
assert changed==['thesis/body/ch02_simulation_platform.tex'],changed
assert not re.search(r'undefined|multiply defined',log,re.I)
assert not re.search(r'Overfull',log)
(O/'integrity.json').write_text(json.dumps({'protected_existing_file_count':len(protected),'changed_existing_files':changed,'only_authorized_chapter_modified':True,'first_chapter_frozen_pdf_unchanged':True,'main_include_unchanged':True,'reference_bib_unchanged':True,'no_training_no_physics_no_server_no_git':True},indent=2)+'\n')
newwc=json.loads((O/'wordcount/THESIS_WORD_COUNT.json').read_text());oldwc=json.loads((P/'wordcount/THESIS_WORD_COUNT.json').read_text());newch=newwc['chapters'][1];oldch=oldwc['chapters'][1]
rows=['# 修订前后实际TeXcount','','|小节|修订前|修订后|差值|','|---|---:|---:|---:|']
for a,b in zip(oldch['sections'],newch['sections']):rows.append(f"|{b['id']} {b['title']}|{a['actual_count']}|{b['actual_count']}|{b['actual_count']-a['actual_count']:+d}|")
rows+=['',f"全章：{oldch['actual_count']} → **{newch['actual_count']}**，变化{newch['actual_count']-oldch['actual_count']:+d}；8000–9000预算内。章标题计数另占16，不包含在小节合计中。",'',f"新统计细分：text {newch['metrics']['text']}、标题 {newch['metrics']['headings']}、图表题/脚注 {newch['metrics']['captions_footnotes']}；数学公式环境{newch['metrics']['display_math']}个，单独统计不折算正文。表格文字按现有规则纳入text。",'','真实命令及终端输出见count_chapter.log/count_files.log。--files仍按正式主文件include图列旧第二章；新章使用--chapter 2的V2规划路径统计。当前独立预览不改变正式主文件计数。']
(O/'WORDCOUNT_COMPARISON.md').write_text('\n'.join(rows)+'\n')
(O/'TECHNICAL_REVIEW.md').write_text('''# 定点技术核查与修订

## A IK的两个姿态机制

重新核对arm.py::__init__/bind/_solve_ik和composite.py::create_robot_controller及指定profile。use_nullspace默认True，nullspace_gain=0.1；未提供nullspace_posture时，bind使用0.5*(ctrl_low+ctrl_high)，当前七关节范围对称，故q0为七维零向量。posture_weight默认0关闭的是额外的加权姿态残差行，不是零空间项；大于0时添加diag(sqrt(weight*joint_weights))及对应姿态误差。

SVD使用full_matrices=False；伪逆为V diag(s/(s²+lambda²)) Uᵀ，投影I-Jinv*J。自适应阻尼实际参与SVD路径。SVD失败时使用Jᵀ(JJᵀ+damping²I)⁻¹e，此分支不增加零空间项。默认posture_weight=0且达到位姿容差时提前退出，因此不能说每步必定继续优化姿态。位置profile不调用IK；其未传入覆盖上述默认值的选项。正文已补齐这些区别，不误写默认姿态项失效。

## B 处理链逐行核验

signal_processing.py::process实际顺序：float64副本 → max(raw-deadzone,0) → clip[0,saturation] → saturation*(value/saturation)^exponent → IIR → Patch内crosstalk → 可选Gaussian → patch结果clip → 可选noise并clip → normalize时除saturation → float32。

IIR保存的是串扰前_filtered状态；后处理值不反馈进IIR。reset(size)建立零向量，reset(None)置None，两种首次调用路径不同。当前profile的crosstalk=.12、Gaussian=0、noise=0、normalize=True，正文公式末行明确写为float32(clip(C_c(v),0,s)/s)，没有把除s放到IIR或串扰之前。通用可选链在文字中保留，公式末行明确限定当前profile。

## C XML和参数单位

逐项读取当前arm/hand actuator节点和tendon/fixed，实际字段保存在actuator_xml_check.json；Builder仅覆盖solver/timestep/iterations/cone，没有更改执行器ctrlrange/kp/forcerange或腱系数。表中数值与XML相符，无需更换数值。

采用模型SI约定：arm控制角rad、输出Nm、kp为Nm/rad；固定腱模型控制长度m、输出N、kp为N/m，腱系数具有模型长度/角度换算意义。不是已测实物力/刚度；引擎按dampratio生成的实际阻尼数值未经本轮编译模型读取，仍UNKNOWN，不填造数值。表头缩短为“输出限幅”以避免不自然换行。

## D 历史统计

只改变显示：8.405%、15.296%、18.263%、0.495%、12.718%。对应原比例和NPZ SHA仍在上一阶段model_and_episode_audit.json中；本轮不重算或生成实验数据，也不改原统计档案。段落中的Push/Stack描述同步改为百分比。

## E 组合几何

新增plot_assembled_geometry.py。纯解析XML并对STL应用刚体变换，复现Builder将arm根变换设为arm_mount、将hand根平移置零以及在right_hand后追加XYZ(-90,-90,0)度旋转的过程。所有转动关节取零，保留body/geom原四元数，执行mesh scale；重复引用同mesh同变换的可视几何去重。不调用MuJoCo编译、forward或step。未画底座，只画机械臂与手部，图题明确静态关节零位。37个mesh实例及输入Hash见geometry_provenance.json。

原图位替换为实际离线组合图及同姿态手部放大，不将该几何图视为动态或碰撞正确性实验。其他段落保持原结构，只作上述技术说明和显示修正。
''')
checks=json.loads((O/'pdf_checks.json').read_text())
(O/'BUILD_ISSUES.md').write_text('''# 第二章临时编译检查

XeLaTeX + BibTeX + latexmk按现有hithesis配置构建成功。临时入口ch02_preview.tex位于revision_v1，正式thesis.tex及include关系未改。正文标第2章，独立预览页码从1开始；图2-1至图2-9、表2-1至表2-7、式2-1至式2-5均正常解析。

- chapter02_preview.pdf：第二章正文14页。
- chapter02_preview_with_references.pdf：完整预览15页，末页为参考文献。
- 中文FandolSong/FandolHei字体嵌入，正文可提取中文；图片中的Type3字形按CharProcs流检查，不把extract_font无法导出Type3误判为缺字体。字体清单和逐页检查见pdf_checks.json。
- 最终日志无未定义引用、无重复label提示、无Overfull。执行器表、712维Patch表及触觉公式已放大检查，未发现越界或裁切。
- 无空白页；第14页为小结结束页，留白较多，不是内部漏排或空白插页。没有通过改页边距、字号或硬插内容压缩篇幅。
- 页眉使用硕士论文模板，页脚为独立预览页码；9图/7表均有题注。全页缩略图和重点页截图保留，供人工复核。

## 保留的非阻断问题

1. 模板数学字体尺寸替代警告（最大约0.41063pt）和未打印glossary警告；未改模板。
2. BibTeX共7条警告：原文献缺address/citedate，两个官方持续更新页面缺year。未猜测网页出版年或修改原文献元数据；引用本身已正常生成。
3. 日志3个Underfull hbox，来自参考文献行分配；无正文Overfull。属于非致命排版提示。
4. 关节零位使总览中的机械臂近似竖直，手部细节通过右图放大；不是任务工作姿态。其余几何图保持原有离线数据，英文坐标注记可留待最终图表统一时处理。

本次仅为预览和修订检查，不宣布正式冻结，不进入第三章，不进行物理实验或Git提交。
''')
(O/'README.md').write_text('''# 第二章定点修订与预览 V1

- ch02_simulation_platform.tex：修订后完整LaTeX快照，正式新章文件同步更新。
- CH02_READING_DRAFT.md：完整阅读稿。
- chapter02_preview.pdf：14页正文预览。
- chapter02_preview_with_references.pdf：15页含参考文献预览。
- WORDCOUNT_COMPARISON.md：实际修订前后字数；完整输出在wordcount/。
- TECHNICAL_REVIEW.md：A–E逐项核查；technical_changes.diff为完整修改差异。
- BUILD_ISSUES.md、pdf_checks.json：编译和PDF检查及未解决警告。
- build/与build_console.log/build_final_console.log：完整构建产物和日志。
- actuator_xml_check.json、geometry_provenance.json：参数与离线模型图来源。
- protected_before.json、integrity.json、SHA256SUMS：修改保护与Hash。

状态：等待用户审核，未正式冻结；未修改第一章、主文件、其他章节或BibTeX。
''')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['python3 docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1/prepare_revision.py','PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1/plot_assembled_geometry.py','python3 docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1/revise_text.py','bash docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1/build_preview.sh','./scripts/count_thesis_words.sh --chapter 2 --output-dir docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1/wordcount','./scripts/count_thesis_words.sh --files --output-dir docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1/wordcount','PYTHONPATH=thesis/.tools/python python3 docs/writing_audit/ch02_writing_v1_2026-11-20/revision_v1/finalize_revision.py'],'note':'prepare_revision.py captures before state and must not be rerun over this archived revision; revise_text.py is a one-time patch, not an idempotent transformation'},indent=2)+'\n')
# Include final inputs, PDF, report and reproducible scripts; leave large build caches outside checksum list except core logs.
paths=[q for q in O.iterdir() if q.is_file() and q.name!='SHA256SUMS']
paths +=[R/'thesis/body/ch02_simulation_platform.tex',R/'thesis/figures/ch02/revision_v1/assembled_robot.pdf',O/'build/ch02_preview.log',O/'build/ch02_preview.blg']
(O/'SHA256SUMS').write_text(''.join(f'{sha(q)}  {q.relative_to(R)}\n' for q in sorted(set(paths))))
print('Final revised words',newch['actual_count'],'pages',refpage,'+',len(d)-refpage,'refs; only current chapter changed.')
