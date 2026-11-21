"""Package the chapter draft and static checks, without compiling LaTeX."""
from pathlib import Path
import re,json,hashlib,difflib,shutil
R=Path.cwd();O=R/'docs/writing_audit/ch02_writing_v1_2026-11-20'
p=R/'thesis/body/ch02_simulation_platform.tex';tex=p.read_text();bib=(R/'thesis/reference.bib').read_text()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((O/'protected_before.json').read_text())
# Recover the exact original prefix for the append-only bibliography diff.
marker=bib.index('% CH02: official documentation')
candidates=[bib[:marker-i] for i in range(5)]
oldbib=next(x for x in candidates if hashlib.sha256(x.encode()).hexdigest()==before['thesis/reference.bib'])
(O/'reference_before.bib').write_text(oldbib)
shutil.copyfile(p,O/'ch02_simulation_platform.tex')
diff=''.join(difflib.unified_diff([],tex.splitlines(True),fromfile='/dev/null',tofile='thesis/body/ch02_simulation_platform.tex'))
diff+=''.join(difflib.unified_diff(oldbib.splitlines(True),bib.splitlines(True),fromfile='thesis/reference.bib.before',tofile='thesis/reference.bib'))
(O/'changes.diff').write_text(diff)
keys=re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib);cites=sorted({k.strip() for v in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',tex) for k in v.split(',')})
labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
missingkeys=sorted(set(cites)-set(keys));missingrefs=sorted(set(refs)-set(labels))
alllabels=[]
for q in (R/'thesis/body').glob('*.tex'):alllabels+=re.findall(r'\\label\{([^}]+)\}',q.read_text())
duplicatelabels=sorted({k for k in alllabels if alllabels.count(k)>1})
# Check nested environment pairing without executing TeX.
stack=[];env_errors=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',tex):
 if m[1]=='begin':stack.append(m[2])
 elif not stack or stack.pop()!=m[2]:env_errors.append(m.group())
if stack:env_errors.append('unclosed:'+','.join(stack))
figures=[]
for i,m in enumerate(re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',tex,re.S),1):
 s=m[0];figures.append({'order':i,'caption':re.search(r'\\caption\{([^}]+)\}',s)[1],'label':re.search(r'\\label\{([^}]+)\}',s)[1], 'image_files':re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s),'placeholder':'图位' in s})
tables=[{'caption':re.search(r'\\caption\{([^}]+)\}',m[0])[1],'label':re.search(r'\\label\{([^}]+)\}',m[0])[1]} for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',tex,re.S)]
changed=[q for q,h in before.items() if not (R/q).exists() or sha(R/q)!=h]
assert changed==['thesis/reference.bib'],changed
assert not missingkeys and not missingrefs and not duplicatelabels and not env_errors
assert all((R/'thesis'/f).is_file() for x in figures for f in x['image_files'])
# Human reading draft with the complete prose, equations, tables and figure descriptions.
def inline(s):
 s=re.sub(r'\\cite\{([^}]+)\}',lambda m:'〔文献：'+m[1]+'〕',s)
 s=re.sub(r'\\(?:eqref|ref)\{([^}]+)\}',lambda m:'〔'+m[1]+'〕',s)
 s=re.sub(r'\\label\{[^}]+\}','',s)
 s=s.replace('\\_','_').replace('~',' ')
 return s
read=re.sub(r'^%.*\n','',tex,flags=re.M)
read=re.sub(r'\\chapter\{([^}]+)\}',r'# 第二章 \1',read)
counter=[0]
def sec(m):counter[0]+=1;return f'## 2.{counter[0]} {m[1]}'
read=re.sub(r'\\section\{([^}]+)\}',sec,read)
def fig(m):
 s=m[0];caption=re.search(r'\\caption\{([^}]+)\}',s)[1];imgs=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s)
 if imgs:return '\n\n'+ '\n'.join(f'![{caption}](/home/zmy/dex_hand_project/thesis/{x})' for x in imgs)+'\n\n'
 s=re.sub(r'\\caption\{[^}]+\}|\\label\{[^}]+\}','',s)
 s=re.sub(r'\\(?:begin|end)\{(?:figure|minipage)\}(?:\[[^\]]*\])?(?:\{[^}]+\})?','',s)
 s=re.sub(r'\\parbox(?:\[[^\]]*\])*\{[^}]+\}','',s)
 s=re.sub(r'\\centering|\\fbox|\\phantom\{[^}]*\}','',s)
 s=re.sub(r'\\\\(?:\[[^\]]*\])?',' → ',s).strip().strip('{}').strip()
 return '\n\n**图：'+caption+'**\n\n'+s+'\n\n'
read=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',fig,read,flags=re.S)
def tab(m):
 s=m[0];cap=re.search(r'\\caption\{([^}]+)\}',s)[1]
 body=re.search(r'\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}',s,re.S)[1]
 rows=[inline(x.replace('\\hline','').strip()) for x in body.split('\\\\')];rows=[x for x in rows if x]
 cells=[[c.strip() for c in row.split('&')] for row in rows];n=max(map(len,cells))
 return '\n\n**表：'+cap+'**\n\n'+'\n'.join(['|'+'|'.join(cells[0])+'|','|'+'|'.join(['---']*n)+'|']+['|'+'|'.join(row)+'|' for row in cells[1:]])+'\n\n'
read=re.sub(r'\\begin\{table\}.*?\\end\{table\}',tab,read,flags=re.S)
read=read.replace('\\begin{equation}','\n$$').replace('\\end{equation}','$$\n')
read=inline(read)
(O/'CH02_READING_DRAFT.md').write_text('本稿由第二章TeX自动转换，供内容审核；引用与交叉引用保留键，正式编号待编译。示意框只表示代码接口，几何图来源为离线STL，不是物理实验截图。\n\n'+re.sub(r'\n{3,}','\n\n',read).strip()+'\n')
wc=json.loads((O/'wordcount/THESIS_WORD_COUNT.json').read_text());ch=wc['chapters'][1]
target=[750,1350,1350,1800,1050,1750,450]
lines=['# 第二章实际字数与统计口径','','引擎：'+wc['engine']+'.', '', '实际执行：`./scripts/count_thesis_words.sh --chapter 2 --output-dir docs/writing_audit/ch02_writing_v1_2026-11-20/wordcount`及相同目录的`--files`。两次均成功，内置测试全部通过。','', '|节|实际TeXcount|建议目标|差值|','|---|---:|---:|---:|']
for s,t in zip(ch['sections'],target):lines.append(f"|{s['id']} {s['title']}|{s['actual_count']}|{t}|{s['actual_count']-t:+d}|")
m=ch['metrics'];lines+=['',f"全章 **{m['total']}** = text {m['text']} + 标题 {m['headings']} + 图表题/脚注 {m['captions_footnotes']}。比中心8500多{m['total']-8500}，在8000–9000内。各节合计不含章标题16，故加总比全章少16。",'',f"数学公式单独统计：行内{m['inline_math']}、独立{m['display_math']}；不计正文目标。表格文字和结构图框内文字按当前规则纳入text，图表标题纳入captions计数。本章无脚注、无代码块。2.1较目标多181，包含模块表与架构图文字；未为逐节精确落点重复扩写。",'', '`--files`只列当前主文件实际include图，因此仍显示旧第二章16，而不会自动把新文件纳入编译。新第二章的8703来自V2规划路径的`--chapter 2`，不是当前PDF已更新。主文件未改，未编译，当前第二章PDF页数UNKNOWN。','', '完整JSON/CSV/raw输出和解析审计见wordcount/；后续审核以新章节文件和planned_2统计为准。']
(O/'WORDCOUNT_REVIEW.md').write_text('\n'.join(lines)+'\n')
checks={'missing_citation_keys':missingkeys,'duplicate_bib_keys':sorted({k for k in keys if keys.count(k)>1}),'missing_reference_labels':missingrefs,'duplicate_body_labels':duplicatelabels,'environment_errors':env_errors,'figures':figures,'tables':tables,'equation_count':len(re.findall(r'\\begin\{equation\}',tex)),'cited_keys':cites,'protected_changed':changed,'bibliography_original_prefix_preserved':True,'first_chapter_unchanged':True,'main_unchanged':True,'compiled':False,'frozen':False}
(O/'static_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
(O/'FIGURES_AND_FORMULAS.md').write_text('''# 图表与公式来源

图按TeX实际顺序编号：9个figure，7个table，5个equation环境。正式章号和页面未编译。

|图|内容|来源及状态|
|---|---|---|
|1|五层架构|builder/factory/rl_env/production_recorder实际接口；TeX框图|
|2|组合模型|明确待补图位；XML/STL已定位，未借用异版本截图|
|3|驱动拓扑|当前arm/hand XML；TeX关系图|
|4|多速率控制|arm.py、rl_env.py、指定profile；TeX流程图|
|5|五指与手掌索引|fitting/layout.py；索引组织图，不冒充整手几何图|
|6|指尖STL与7×8采样|当前skin_0_2_p.STL及fingertip_ellipsoid_grid_points；离线PDF/PNG|
|7|局部采样坐标|相同采样点及生产_taxel_frame；外向修正前的三轴|
|8|处理链|signal_processing.py及profile；高斯/噪声注明关闭|
|9|验证层次|五条已有NPZ及材料核验JSON；不表示重新执行任务|

几何图可用`PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch02_writing_v1_2026-11-20/plot_geometry.py`复现。脚本仅读取STL并拟合/绘图，不创建或推进MuJoCo物理模型。坐标为mesh局部坐标，不能当作完整手部世界姿态或Real电极位置。

7表依次为模块、执行器、控制参数、布局参数、16Patch索引、信号阶段、五Episode统计；来源为材料核验目录model_and_episode_audit.json/source_inventory.json与对应原始文件。图表中的技术参数不是新增实验测量。

|公式标签|内容|代码依据|
|---|---|---|
|eq:ch02_tendon|固定腱关节线性组合|当前dex_hand.xml tendon/fixed|
|eq:ch02_ik|加权Jacobian的SVD阻尼伪逆及姿态项|source/control/arm.py::_solve_ik；自适应/限幅在正文解释|
|eq:ch02_bezier|张量积Bezier及加权ridge求解|fitting/bezier.py::fit_bezier_surface/evaluate_bezier_surface|
|eq:ch02_site_size|邻距和overlap决定box半宽|dex_hand.py::_neighbor_spacing/SimpleBoxTactileSensor._add_taxel|
|eq:ch02_processing|死区、饱和、幂响应及IIR|signal_processing.py::process；reset边界在正文说明|

未添加Hertz/FEM材料模型、显式推杆闭链或未实现控制律。第二章仅解释当前模拟读出，不声称实物传感器标定。
''')
(O/'SELF_REVIEW.md').write_text('''# 自审与交付状态

状态：阶段3草稿已完成，等待用户/ChatGPT审核；未编译、未冻结。

- 完整2.1–2.7，实际TeXcount 8703；当前include仍使用旧骨架，新文件未自动接入主文档。
- 采用当前12关节/6固定腱、原生touch、swept-shell+Bezier实际路径；不把工程集成表述为原创算法。
- 控制器默认、指定profile和历史结果分开，IK/位置伺服/物理积分分开。
- 五条历史sensor_tactile采用已核验统计，历史success不称本轮执行结果；UNKNOWN和Root独立性边界保留。
- 九图中6张为接口/拓扑框图、2张为实际STL离线几何图、1张组合几何图位。五指总体图目前是索引拓扑，尚非完整手部3D几何图；未用假截图填补。
- 7表、5公式；引用键、环境配对、交叉引用和图像文件存在性静态检查见static_checks.json。
- 原28个BibTeX条目的字节前缀完整保留，仅追加mujoco2026xml及mujoco2026computation两项已核验官方在线资料。键中2026标识核验年，不被写成文献出版年；无猜测DOI。
- 第一章、冻结PDF、主文件、旧章节、V2规划及模板均未改变。没有训练、仿真、服务器操作、P52/P53或Git提交。

## 剩余问题

1. 组合模型准确几何图尚待补；当前不修改主文件来渲染，也不借用旧模型图。
2. 当前版本与历史五任务记录的完整绑定仍UNKNOWN，按已授权的历史功能证据口径写入。
3. 本轮未编译，浮动体位置、公式宽度、字体和新增在线文献的实际排版尚未验收；留到内容审核后的阶段5。
4. 2.1含架构/模块图表而高于750目标；全章满足预算，不通过压缩证据边界机械对齐每节。
5. 几何拟合算法与采样图不提供真实Taxel物理映射、面积或压力标定，也未作为新实验结果。

请审核全文论证与边界，确认后再下达修改或编译任务；不会自行开始第三章。
''')
(O/'RUN_COMMANDS.json').write_text(json.dumps({'commands':['PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch02_writing_v1_2026-11-20/plot_geometry.py','./scripts/count_thesis_words.sh --chapter 2 --output-dir docs/writing_audit/ch02_writing_v1_2026-11-20/wordcount','./scripts/count_thesis_words.sh --files --output-dir docs/writing_audit/ch02_writing_v1_2026-11-20/wordcount','python3 docs/writing_audit/ch02_writing_v1_2026-11-20/package_delivery.py'],'no_latex_compilation':True,'no_physics_simulation':True},indent=2)+'\n')
# Hash source inputs, final draft, images, delivery and exact bibliography prefix.
paths=[p,R/'thesis/reference.bib']+list((R/'thesis/figures/ch02').glob('*'))+[x for x in O.iterdir() if x.is_file() and x.name!='SHA256SUMS']
inputs=[R/'assets/grippers/dex_hand/meshes/skin_0_2_p.STL',R/'source/sensors/tactile/fitting/fingertip.py',R/'source/sensors/tactile/fitting/bezier.py',R/'source/sensors/tactile/dex_hand.py']
(O/'SHA256SUMS').write_text(''.join(f'{sha(x)}  {x.relative_to(R)}\n' for x in sorted(set(paths+inputs)) if x.exists()))
print('Packaged draft:',ch['actual_count'],'words; figures',len(figures),'tables',len(tables),'equations',checks['equation_count'])
print('Protected files: only authorized append to reference.bib; original prefix verified.')
