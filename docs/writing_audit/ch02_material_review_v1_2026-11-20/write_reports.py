"""Render material-audit documents from the read-only inventory."""
from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent;R=O.parents[2]
A=json.loads((O/'model_and_episode_audit.json').read_text());S=json.loads((O/'source_inventory.json').read_text())
def put(n,t):(O/n).write_text(t.strip()+'\n')
def table(head,rows):return '|'+ '|'.join(head)+'|\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('|'+ '|'.join(map(str,row))+'|' for row in rows)+'\n'
chap=[('2.1','系统需求与总体架构',700,800,750,'可写：模块接口及数据流；不将接口存在当作全部场景验证。'),('2.2','RM75B机械臂与Dex Hand灵巧手模型',1300,1450,1350,'有条件可写：明确当前固定腱模型，历史机构版本不能混写。'),('2.3','灵巧手接触建模与控制方法',1300,1450,1350,'可写：当前代码与指定profile；历史各任务配置不能自动沿用。'),('2.4','复杂曲面触觉阵列建模',1700,1950,1800,'可写算法：真实网格派生布局；未执行新几何绘图、面积认证或物理标定。'),('2.5','712维触觉观测及处理流程',1000,1100,1050,'可写：native touch及指定processed链；不能称真实压力。'),('2.6','仿真平台功能验证',1600,1800,1750,'有条件可写：五条历史开发样本的数组核验；当前模型动态验证UNKNOWN。'),('2.7','本章小结',400,450,450,'可写：限定已核验范围，承接第三章。')]
base='''# 第二章整章材料核验（TH-CH02-V2）

日期：2026-11-20。阶段2，不是正文，不构成写作授权。全部路径相对项目根目录。

## 总体结论

当前材料能支撑平台、控制、阵列布局与读出流程的技术描述，以及指定历史Episode的有限功能证据。进入阶段3应为**有条件就绪**：先由用户确认如何在正文区分“当前固定腱模型实现”和“历史任务记录”，不能将二者自动视为同一冻结版本。没有启动仿真、训练、服务器操作、P52/P53，也未修改正式TeX、参考文献、模板或历史数据。

证据等级：L0只定位；L1已读报告/配置的声明；L2只对实际核对的函数、XML字段或指定数组有效，不向整个系统或全部数据外推。Python AST和XML解析不是运行时动态验证。

## 篇幅与实际正文状态

当前主文件仍使用六章骨架，`thesis/body/ch02_data_and_problem.tex`是旧占位；V2对应`body/ch02_simulation_platform.tex`尚未创建。2.1–2.7实际正文统计均为MISSING（尚无对应正式正文可测），不能把材料报告计入TeXcount。旧占位不代表V2第二章完成量。本轮不改include、不运行编译。以下预算总和8500，范围8000–9000。

'''
base+=table(['小节','标题','预算','建议落点','实际正文','准备情况'],[(n,t,f'{lo}–{hi}',v,'MISSING',state) for n,t,lo,hi,v,state in chap])
base+='''
## 2.1 架构核验

实际链路为机器人descriptor注册→builder加载arm/base/hand并安装传感器→任务factory→Composite控制器→MuJoCo物理子步→观测及任务评价→记录器。控制目录是`source/control/`，不是不存在的`source/controllers/`。`make_manipulation_env`通过注册表创建五种任务，关闭默认场景，由任务补入场景。五种factory入口和五条已有记录共同支持“已实现五类任务接口且存在开发执行产物”，不支持全部任务对所有资产均成功。

可以按模型、控制、感知、任务、记录五层组织论述；记录器和数据格式在此仅说明接口，详细lineage与转换留第四章。不能以模块框图声称实机部署。

## 2.2 机构与安装核验

`assets/robots/rm75b/rm75b.xml`有joint1–joint7七个转动关节及pos_joint1–7位置执行器。安装入口为`right_hand`与`right_hand_site`，descriptor的附加旋转为XYZ欧拉角(-90,-90,0)度，通过SciPy转四元数并由builder组合；还须保留父body已有变换，不能把附加角直接当世界姿态。

当前`assets/grippers/dex_hand/dex_hand.xml`为12个转动关节、6个固定腱及6个位置执行器；四指每指2关节，拇指4关节。不存在显式slide推杆、equality闭链或flex。名称`act_push_*`保留历史命名，但实际传动是tendon，不得写成当前XML中的推杆位移驱动。四指固定腱每个关节系数0.003；拇指对掌0.0085；拇指屈曲0.0035/0.004/0.004。被动弹簧和接触后关节分配形成模型中的欠驱动响应；不是硅胶材料形变验证。

XML的具体关节、范围、轴、全部执行器及腱系数见`model_and_episode_audit.json`。7+6=13是控制通道数，不是机器人13个物理关节。模型资产可定位XML、URDF和STL，但CAD原始出处、授权及相对实机几何精度UNKNOWN，不得补写厂商认证。

## 2.3 接触与控制核验

独立arm XML写PGS/50次，但实际builder `_configure_solver`覆盖为Newton/100次、椭圆摩擦锥、0.001s步长；正文描述组合构建路径应采用覆盖后值。当前profile control_dt=0.05s，对应每控制周期50个物理步（20Hz/1000Hz），而不是触觉传感器固有带宽。环境每步先apply_action，再physics子步，随后观测/评价；同一周期内保持执行器目标。

当前手部padgeom为mesh碰撞，condim=4，friction=(1.3,0.02,0.0002)，solref=(0.01,1)，solimp=(0.9,0.95,0.001,0.5,2)。这些不是任意接触最终统一参数：XML还有显式contact pair；物体/桌面有各自参数，需按配对和优先级解释。碰撞与采样使用同一命名skin mesh资产及geom变换，但MuJoCo碰撞表示与拟合曲面不能等同，不能声称逐三角面精确弹性接触。

ArmPositionIkController在position模式直接写关节目标；ik模式接受xyz+四元数，用mj_jacSite得到位置/旋转Jacobian，SVD阻尼伪逆并可叠加零空间姿态项，再限步、限幅和目标平滑。默认80次迭代、阻尼1e-3、自适应开、单步上限0.15rad、速度上限2rad/s、位置容差1e-4m、旋转容差1e-3rad；姿态权重0.35、位置权重1、零空间开且gain0.1，额外posture_weight默认0。它们是构造默认值，具体调用可覆盖，不代表每条历史轨迹均使用IK模式。

实际底层为MJCF position servo（kp及dampratio），不是发现了单独实现的通用力矩PD程序；只有在解释MuJoCo执行器时可讨论等效比例/阻尼作用。控制律和实际分支见公式档案。

## 2.4 曲面阵列核验

固定布局为每指4×8、4×8、7×8三个Patch，五指共600；手掌7×16共112，总计16 Patch、712个box site及712个标量touch传感器（由augment_spec在构建时添加，不是在原始XML中逐行写出）。通道按finger0–4、每指0/1/2、掌心、每Patch row-major顺序连接，区间见下表。手指编号的真实解剖名称及Real电气对应关系未认证，先保留索引。

segment入口先对STL外壳采样，以PCA、椭圆截面和外表面筛选构造规则样本，再拟合Bezier曲面。实际_regular_surface使用dense_rows=max(18,4r)、dense_cols=max(32,4c)，degree=(5,7)，正则5e-5，参数单元中心采样。

fingertip-ellipsoid名称不能当算法结论：当前函数转入swept-shell外壳识别和Bezier拟合，dense_rows=max(12,3r)、dense_cols=max(24,3c)，degree=(6,7)，正则8e-6，端点边界权重10；最终u来自[0.06,1]分格中心，避免指尖极点重复。部分旧docstring仍提理想椭球，与实际函数体不同，应以调用链为准。

mesh-uv用于掌心：PCA投影到二维参数域，在三角形内用重心插值回到3D；投影重叠时选择外表面、缺口处存在最近三角形回退。它不是无畸变全局曲面参数化，不能默认每格具有相同真实面积。

采样点经geom局部旋转/平移进入所属body坐标；局部切线由行列差分构造，Gram–Schmidt正交化，叉乘得法向，再以外向启发式修正方向。site半宽按相邻距离最小值×overlap/2并受最小半宽约束，厚度独立设置。site是有体积的感知区，不是独立碰撞Taxel；感知区可重叠，法向是几何构造量，不是已经标定的实物法向。

profile指定half_depth=0.004m、overlap=1.25、min_half_size=0.0005m；类默认分别0.0015m、1.0、0.0005m，不能混写。真实Taxel电极面积、覆盖率和几何误差尚未统计；box截面面积不能冒充物理电极面积。允许后续用现有STL/离线采样绘图，本轮只核验实现，没有运行模型或生成新物理样本。

'''
base+=table(['Patch','行×列','通道区间（零基、右开）','策略名'],[(x['name'],f"{x['rows']}×{x['cols']}",f"[{x['start']},{x['stop']})",x['kind']) for x in A['patches']])
base+='''
## 2.5 原生读出与处理核验

调用链：builder→tactile.augment_spec→SimpleBox._add_taxel创建box site及mjSENS_TOUCH→bind按sensor名称验证dim=1并保存sensor_adr→read_raw从sensordata地址取float32副本→TactileSignalProcessor.process→processed712。sensordata地址不同于sensor编号。历史P40五个编译模型的记录均为730个sensor，其中712原生touch、18执行器传感器，插件实例0、flex0；这是历史报告L1，非本轮加载模型结果。

原生touch非负标量累加感知区内相关接触的法向力；SI模型约定下可解释为模拟力N，但不除面积、不含三轴切向力，不能说是认证真实力或Pa。processed经过死区、截断、幂响应、IIR、Patch邻域传播、可选高斯/噪声及归一化后，不能保留N单位。重叠site可能重复计同一接触力，712总和不自动等于总外力。

指定profile启用deadzone=.002、saturation=1、exponent=.7、alpha=.45、crosstalk=.12、normalize=true；gaussian_sigma=0、noise_std=0，因此不能写本profile启用了高斯扩散或噪声。处理顺序及公式见公式档案。crosstalk使用8邻域、按源邻居数量分配；Gaussian辅助函数虽名reflect，实际np.pad采用edge，不能照函数名写镜像边界。滤波reset初态也须按实际reset(size)路径说明，不能笼统保证无启动瞬态。

touch_grid插件在历史本机注册列表中存在不代表生产模型使用它。历史本机MuJoCo为3.10.0，服务器当前版本及插件实例UNKNOWN。不能用本机历史版本代替服务器证据。

## 2.6 历史功能证据核验

本轮只读以下五份NPZ的sensor_tactile、timestamp、字段shape，并核对result/manifest。五份均float32[T,712]、finite、非负、非全零，时间增量约0.05s。以下非零率是对源数组重新统计，不是引用旧summary。

'''
base+=table(['任务案例','T','非零比例','finite/非负','历史success','当前模型版本一致性'],[(e['case'],e['fields'][e['tactile_key']]['shape'][0],f"{e['nonzero_fraction']:.9f}",'是/是',e['historical_success'],'UNKNOWN') for e in A['episodes']])
base+='''
五条数组同时含`tactile`及`sensor_tactile`，实际值并不完全相同，部分非零率也不同；本章合同统计明确采用production_recorder验证的sensor_tactile，不把两个字段混称同一读出时刻。其严格记录时序归因还需结合当时record调用位置，不在此凭差值推断。

Lift样本qpos[T,26]、qvel[T,25]、ctrl[T,13]与当前arm7+hand12加自由物体的维数相容，但**维数相容不能证明XML、几何、控制与信号参数同版本**。五条manifest未建立本轮当前XML完整hash对应，独立Root也UNKNOWN。历史result记录success=true、RGB解码检查及合同声明属L1；本轮没有视频重解码、执行器重新运行、任务物理重放或Strict复验。只能说历史记录存在有效格式触觉观测，不能说本轮验证了全部当前物理行为。

Lift的Strict单独记录可引用为历史声明，不能传播到另外四项任务；不同任务成功判据和抓取性能留第三章。数据全量去重、同步与LeRobot审计留第四章。NPZ非零不证明物理标定正确，功能验证不证明跨域迁移成功。

## 2.7 收束范围

可总结组合模型、控制/物理多速率、STL驱动的曲面布点、原生touch到processed的合同及五条历史功能样本。不要提前总结迁移实验，不将当前版本材料写成全量硬件/生产验收。承接第三章：在已有平台与观测基础上研究候选到可执行示教的约束。

## 待审核的关键事实边界

1. 当前XML固定腱、无显式推杆/闭链；正文是否以当前实现为主，并将历史机构作为版本说明？
2. 历史五任务样本与当前XML/配置一致性UNKNOWN；允许以历史功能记录形式使用，不补做实验。
3. 实物Taxel映射、面积、CAD来源精度和力标定UNKNOWN；不影响按仿真内部索引描述实现，但限制物理等价结论。
4. 8000–9000预算可由技术实现、图表解释和有限功能证据组织；2.6若不足，不以新增性能声明或重复数据审计凑字数。

完整hash、函数行号与NPZ字段见同目录JSON；论文现有第一章及引用文件受保护，完整性检查见integrity.json。
'''
put('CH02_MATERIAL_REVIEW.md',base)
entries=[('2.1','模块构建与任务factory','source/robots/builder.py','build_robot','L2函数'),('2.1','五任务入口','source/envs/manipulation/factory.py','make_manipulation_env / make_*_env','L2函数'),('2.2','arm7关节位置执行器','assets/robots/rm75b/rm75b.xml','worldbody/joint; actuator/position','L2 XML'),('2.2','hand12关节6腱；无equality/flex','assets/grippers/dex_hand/dex_hand.xml','worldbody/joint;tendon/fixed;actuator','L2 XML'),('2.2','附加安装旋转与末端site','source/robots/arms/rm75b.py','RM75B_ARM','L2配置代码'),('2.3','DLS、加权误差与零空间','source/control/arm.py','_solve_ik;_adaptive_damping;_smooth_joint_target','L2函数'),('2.3','手部目标与传动坐标','source/control/end_effectors.py','bind;apply_action;current_position','L2函数'),('2.3','每周期物理子步及先动作后观测','source/envs/rl_env.py','step;step_physics','L2函数'),('2.4','16patch/712行主序','source/sensors/tactile/fitting/layout.py','dex_hand_patch_layout','L2函数'),('2.4','外壳+Bezier','source/sensors/tactile/fitting/segment.py','_finger_segment_regular_surface_grid_points','L2函数'),('2.4','非理想椭球实现','source/sensors/tactile/fitting/fingertip.py','_fingertip_regular_surface_grid_points','L2函数'),('2.4','掌心PCA投影及重心插值','source/sensors/tactile/fitting/projection.py','_mesh_uv_grid_points_from_triangles','L2函数'),('2.4','Bezier正则拟合','source/sensors/tactile/fitting/bezier.py','fit_bezier_surface;evaluate_bezier_surface','L2函数'),('2.4/2.5','site构建、地址绑定、read_raw','source/sensors/tactile/dex_hand.py','augment_spec;bind;read_raw;_add_taxel','L2函数'),('2.5','processed顺序及邻域实现','source/sensors/tactile/signal_processing.py','process;_neighbor_crosstalk;_convolve_axis_reflect','L2函数'),('2.3/2.5','特定profile参数，不冒充所有实验参数','configs/robot_profiles/rm75b_dex_hand.json','control_dt;tactile_sensor_options','L2文件'),('2.6','合同字段与时间间隔规则','source/data/production_recorder.py','validate_online_arrays','L2函数')]
rows=[]
for sec,claim,p,fn,lev in entries:
 fs=S[p].get('functions',[]);matches=[f"{x['name']}:{x['line']}–{x['end']}" for x in fs if x['name'] in fn.split(';') or x['name'] in fn.split(' / ')]
 rows.append((sec,claim,p,fn+('；'+', '.join(matches) if matches else ''),S[p]['sha256'],lev))
put('CH02_EVIDENCE_MATRIX.md','# 论点—原始证据矩阵\n\n完整Hash如下；行号由AST提取。L2仅限列明范围，XML默认为语义解析而非运行时验证。\n\n'+table(['小节','论点','文件','函数/字段/行号','SHA256','等级'],rows)+'\n## 指定历史数组\n\n'+table(['文件','字段','SHA256','等级'],[(e['path'],'sensor_tactile / timestamp / shape',e['sha256'],'L2指定数组') for e in A['episodes']])+'\nresult/manifest路径及Hash见source_inventory.json；success/视频/Strict历史记录只作L1。未核验当前运行模型、CAD来源认证、真实Taxel电气映射，均UNKNOWN。\n')
figs=[('图2-1','模块架构与控制/数据流','builder、factory、rl_env、record代码','可据已核验接口画框图；不是实际截图'),('图2-2','组合模型与安装','arm/hand XML、STL、body变换','现有截图须先查来源；本轮不启动渲染仿真'),('图2-3','关节—腱—执行器关系','model_and_episode_audit.json','可离线画7关节+12手关节/6腱，不能画虚构推杆'),('图2-4','控制周期与物理子步','controller、env、profile','可离线画20Hz/1kHz限定配置流程'),('图2-5','曲面阵列总体布局','16个STL、geom变换、fitting调用','可离线计算；本轮未生成，不用旧异版本图冒充'),('图2-6','局部Taxel与box感知区','_taxel_frame/_add_taxel、STL','可离线绘制，标局部坐标与感知体积'),('图2-7','外壳拟合/Bezier/mesh-uv','segment/fingertip/projection','可离线可复现图，不声称物理面积积分'),('图2-8','raw→processed','signal_processing.py、profile','可画流程，Gaussian/noise标该profile关闭'),('图2-9','功能记录核验流程','五条NPZ/result/manifest','可画读取—shape—数值—时间检查；不是本轮执行任务流程')]
put('CH02_FIGURE_TABLE_PLAN.md','# 图表计划\n\n图表是阶段3候选，不是已生成的正式论文图。禁止AI合成截图或压力真值。\n\n'+table(['编号','目的','来源','可生成性/限制'],figs)+'''\n## 表格

- 表2-1 模块及功能：builder/control/sensors/tasks/recorder；不展开第四章存储设计。
- 表2-2 执行器：从JSON完整列7个arm、6个hand传动坐标/range/kp/force；物理关节另列，区分rad与腱长度。
- 表2-3 控制频率及参数：代码默认、profile、历史记录三列，未知不填默认。
- 表2-4 16 Patch的rows/cols/kind/start/stop：使用实际JSON，接口名与实际拟合策略并列。
- 表2-5 raw/processed/image层级：维度、dtype、变换、单位与限制；可视化色标不是采集幅值缩放。
- 表2-6 五条指定sensor_tactile：T、finite、非负、非零率、20Hz、历史success及证据等级。
- 表2-7 功能核验边界：历史记录存在/本轮数组检查/当前物理动态UNKNOWN/硬件标定UNKNOWN。

不必为满足编号重复制作热图。若展示触觉，使用各Patch原4×8、7×8、7×16方阵统一色标，不把712向量拉成长条。多时刻来自同一源Episode，标timestamp与字段名，不混用tactile和sensor_tactile。几何图标明仿真内部坐标，未认证Real对应关系。
''')
put('CH02_FORMULA_PLAN.md',r'''# 公式计划与实现约束

使用现有术语：raw为模拟法向力累加，processed为后处理标量；空间坐标单位m，关节rad；不能将下列量都称压力。这里只列可写公式，不写论文正文。

## F1 坐标与运动学（2.2/2.3）

\(p_b=R_{bg}p_g+t_{bg}\)，局部点、旋转3×3、平移3；依据dex_hand.py::_transform_points和builder安装变换。世界位姿由body链复合，可写\(T_{we}(q)=\prod T_i(q_i)\)为MJCF链的记号，不能虚构DH参数表。

\(\dot x=J(q)\dot q\)，arm q∈R⁷、末端速度六维；实际mj_jacSite获取平移和旋转Jacobian，位置m、旋转rad。不是本项目新算法。

## F2 实际IK（2.3）

\(J_w=[w_p J_p; w_r J_r]=U\Sigma V^T\)，\(e_w=[w_p e_p;w_r e_r]\)。代码默认权重1和0.35；旋转误差来自目标四元数乘当前共轭后转rotvec，不是Euler角相减。

\(J_\lambda^+=V\operatorname{diag}(\sigma_i/(\sigma_i^2+\lambda^2))U^T\)，\(\Delta q=J_\lambda^+e_w+(I-J_\lambda^+J_w)k_n(q_0-q)\)。第二项只在use_nullspace启用时使用；阻尼投影不是严格正交零空间，不能声称绝对不影响主任务。

更新\(q\leftarrow\operatorname{clip}(q+\operatorname{clip}(\Delta q,-s,s),q_{min},q_{max})\)，之后还有_smooth_joint_target速度约束与滤波；不得只写未受限伪逆当完整控制器。SVD失败分支退回\(J^T(JJ^T+\lambda^2I)^{-1}e\)。posture_weight>0时Jacobian/误差另增正则姿态行，默认0。

依据source/control/arm.py::_solve_ik/_adaptive_damping/_smooth_joint_target。自适应阻尼阈值和滤波应由代码表表达，无需再推导标准理论。

## F3 固定腱传动与伺服（2.2/2.3）

\(l_t=\sum_j c_{tj}q_j\)，当前MJCF tendon/fixed；六行系数见JSON。c采用模型长度/角度约定，不是测得的物理推杆尺寸。控制输入为腱位置目标而非手指所有关节独立目标。

可按MuJoCo position actuator文档解释\(f_a=k_p(u-l)-k_v\dot l\)，其中kv由dampratio等引擎规则给定，最终有force限幅。此为执行器层说明，不是自编通用PD控制器；未经编译不填实际kv数值，不把腱力与关节力矩混为同一单位。

## F4 Bezier拟合及参数采样（2.4）

\(S(u,v)=\sum_{i=0}^{d_u}\sum_{j=0}^{d_v}B_i^{d_u}(u)B_j^{d_v}(v)C_{ij}\)。控制点三维m、Bernstein参数无量纲。fit_bezier_surface构造设计矩阵A并解
\((A^TWA+\rho I)C=A^TWP\)，\(\rho=\eta\max(\mathrm{tr}(A^TWA)/n_c,10^{-12})\)。不是固定未缩放ridge；指尖有端点权重10，segment无该额外加权。

segment参数中心\(u_r=(r+1/2)/R,v_c=(c+1/2)/C\)；指尖u先在[0.06,1]分格再取中心。掌心用投影三角形重心权重\(p=\sum_i\beta_i v_i\)，\(\sum_i\beta_i=1\)，需交代多层投影及回退策略，不声称等真实面积。

## F5 朝向与感知区（2.4）

由列差分\(a\)归一化得\(e_x\)，行差分b作\(e_y=\mathrm{normalize}(b-(b^Te_x)e_x)\)，\(n=e_x\times e_y\)，\(R=[e_x,e_y,n]\)，再按实现外向启发式翻转。边界/退化情况按函数保护，不称精确网格法线。

\(h_x=\max(h_{min},\frac12\gamma d_c)\)，\(h_y=\max(h_{min},\frac12\gamma d_r)\)，d取对应邻距最小值，hz来自depth配置。box截面4hxhy只是感知区几何量，不是实物Taxel电极面积。

## F6 raw与processed（2.5）

\(r_j=\sum_{k\in\mathcal C_j}f^n_k\)。C由原生touch的body、site区域及引擎重投影规则决定；重叠C可能重复，不可把sum_j r_j当总力。原生无面积除法、无三轴切向分量。

\(a=\operatorname{clip}(r-d,0,s)\)，\(b=s(a/s)^p\)，\(v_t=\alpha b_t+(1-\alpha)v_{t-1}\)。初态按reset(size)路径，不能默认每段独立都从首帧初始化。

Patch串扰\(w_i=(1-c)v_i+c\sum_{j:i\in N(j)}v_j/|N(j)|\)，N为可用8邻域，注意按源节点邻居数归一化。可选高斯\(g_k\propto\exp[-k^2/(2\sigma^2)]\)，radius=max(1,ceil(3sigma))、一维核归一化、两轴顺序卷积、edge填充。指定profile sigma=0未启用。

有噪声时添加高斯噪声并clip，normalize时除s，最终float32。不能把Gaussian后处理写成原生接触力渲染，也不把normalized输出写N/Pa。

## 不应写入

未实现的Flex/FEM硅胶本构、Hertz接触压力推导、touch_grid插件核、真实Taxel面积积分、剪切三分量、虚构DH/推杆闭链方程、未标定Real/Sim通道等价、跨域网络损失。它们均非本章当前实现证据。
''')
# Reuse previously audited bibliographic metadata without changing reference.bib.
cat=json.loads((R/'thesis/docs/writing_audit/ch01_1_2_sources/literature_catalog.json').read_text())
selected=[x for x in cat if x['id'] in ['S06','S08','S09','S10','S25']]
refs='# 文献与官方技术资料核验\n\n日期2026-11-20。本轮不修改reference.bib。共9项：5篇已核验论文、3项MuJoCo官方文档主题及1项作者教学资料。论文复用1.2核验档案，不声称本轮全文重读；详细作者/题录/原PDF/阅读页码同步保存在reference_records.json。官方在线资料本轮核对相关段落，stable版本不能替代历史运行版本证明。\n\n'
for x in selected:
 refs+=f"## {x['id']} {x['title']}\n\n作者：{'; '.join(x['authors'])}。{x['year']}，{x['venue']}。DOI：{x['doi']}。\n\n原文：{x['primary_url']}；本地：`{x['local_pdf']}`。阅读范围：{x['reading_scope']['detail']}，PDF页{x['reading_scope']['pdf_pages']}（沿用已审核阅读记录，不标为本轮全篇阅读）。\n\n"
 refs+= {'S06':'用途：2.1/2.3物理引擎背景；实际版本参数仍以项目代码为准。','S08':'用途：2.4/2.5区别视觉触觉渲染和原生标量touch，不将其成像模型写成本项目实现。','S09':'用途：2.4/2.5传感器响应/形变/读出的区别，不能声称已复现GelSight模型。','S10':'用途：2.5物理单位与标定边界，仅简短背景，不把其FEM/3D接触力输出作为本项目能力。','S25':'用途：2.5传感器模型差异边界，可选引用；潜空间迁移详论已放1.2/第五章，不在本章重复。'}[x['id']]+'\n\n'
refs+='''## D01 MuJoCo Contributors：XML Reference，sensor/touch

官方链接：https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-touch 。在线持续更新资料，无固定期刊/DOI，访问2026-11-20。本轮阅读touch段：空间区域、body关联、软接触重投影、法向力累加及非负标量。对应2.5；配合本项目P40封存engine_sensor.c，不凭文档推断服务器实际版本。

## D02 MuJoCo Contributors：Computation，Soft contact model

https://mujoco.readthedocs.io/en/stable/computation/index.html 。本轮阅读contact模型概述及算法目的；对应2.3，引擎柔顺接触不是Flex软体。无固定出版年份/DOI，不编造。

## D03 MuJoCo Contributors：XML Reference，position actuator / fixed tendon / mesh

https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position 。本轮核对position的dampratio以及mesh相关属性；结合XML说明执行器层与几何语义。与D01同一文档的不同主题，正式BibTeX可合并为一个官方文档条目，不应重复充数。fixed tendon实际系数由本项目XML证明，未把第三方资料当模型来源。

## D04 Kevin M. Lynch、Frank C. Park：Modern Robotics教学资源，6.2 Numerical Inverse Kinematics

Cambridge University Press配套作者资料；教材2017，网页年份不擅填。官方链接：https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/ 。本轮阅读网页Transcript中Newton–Raphson、Jacobian伪逆、冗余和迭代初值的说明。仅支持2.3背景；实际SVD阻尼/零空间配置以arm.py为准，不将网页当本项目DLS实现证明。未阅读全文教材，不编造教材DOI。

## 题录和引用处理

S06/S08/S09/S10/S25沿用1.2记录，正式写作先复用现有bib中的对应标题条目；本轮不生成新的键或改元数据。官方资料仅记录URL和访问日期，是否合并条目由后续任务书授权。无需为凑6–12项引入未读的欠驱动机构论文。

RM75B和Dex Hand模型的厂商原始CAD出处、参数认证、手部结构论文与硬件版本仍UNKNOWN；此缺口由项目资产描述明确披露，不能用相似产品网页补证。近似拟合曲面和site几何是项目代码证据，不自动具有独立论文级新颖性。
'''
put('CH02_REFERENCE_REVIEW.md',refs)
(O/'reference_records.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
put('CH02_WRITING_READINESS.md','''# 第二章阶段2就绪结论

**结论：CONDITIONAL READY，等待用户和ChatGPT审核；不自动进入阶段3。**

1. 2.1/2.3/2.4/2.5：代码与配置材料足以起草限定版本的实现论证。阵列图尚未从当前资产生成，后续可独立离线绘制，不需要新增物理实验。
2. 2.2：当前XML结构已核对，必须采用固定腱12关节/6驱动描述；CAD来源、实机结构等价和几何误差UNKNOWN，不能写成厂家认证模型或显式推杆闭链。
3. 2.6：五条历史sensor_tactile已直接检查，可写历史记录的合同与数值有效性；不能说本轮重新执行任务或验证当前模型。各样本与当前XML的精确hash绑定UNKNOWN，五条独立Root身份UNKNOWN，Strict不外推。
4. 2.7：只在上述范围收束，无需新的研究结果。

## 审核需要决定的一个版本口径

建议正文用“当前实现”描述模型/算法，用“历史开发记录”报告五任务样本，并明确未认证二者配置完全一致。若要求本章宣称同一冻结系统的端到端动态验证，现有证据不足：先补当时模型与配置档案，不能自动启动新仿真。

## 尚存UNKNOWN

- 历史Episode与当前XML、fitting及processor配置完全同版本的证据。
- 机器人CAD制造商来源、授权/几何精度、真实安装标定。
- Real电气通道到仿真索引的认证映射、真实压力/力标定。
- Taxel实际表面积、局部面积误差与法向误差；感知box面积不是实物面积。
- 服务器当前MuJoCo/插件及生产配置；本轮不访问服务器。
- 五条记录的完整Root/reset关系及不同tactile字段时序归因。

上述缺口限制物理真实性、独立性和当前系统动态结论，不妨碍按已核验代码描述仿真内部实现。

## 篇幅

2.1 750、2.2 1350、2.3 1350、2.4 1800、2.5 1050、2.6 1750、2.7 450，总8500。实际正文为MISSING，未新建或修改第二章TeX，未将本报告当正文统计。2.6不以数据总量或额外性能比较填充。

## 停止状态

材料报告已交付；不写正文、不改bib/主文件/骨架、不编译、不运行训练/仿真/实机/服务器、不commit/push、不进入第三章或其他实验。后续须获得本轮材料审核及正式写作任务书。
''')
# Verify protected sources against the inventory captured before report generation.
checks=[]
for p,v in S.items():
 q=R/p
 checks.append({'path':p,'sha256_before':v['sha256'],'sha256_after':hashlib.sha256(q.read_bytes()).hexdigest(),'unchanged':v['sha256']==hashlib.sha256(q.read_bytes()).hexdigest()})
(O/'integrity.json').write_text(json.dumps({'checked_sources':checks,'all_unchanged':all(x['unchanged'] for x in checks),'no_simulation_or_training':True,'official_thesis_sources_modified':False,'note':'read-only snapshots cover listed source/config and historical metadata; NPZ SHA in model_and_episode_audit.json'},ensure_ascii=False,indent=2)+'\n')
print('Wrote six reports; source integrity:',all(x['unchanged'] for x in checks))
