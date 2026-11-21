# 第二章整章材料核验（TH-CH02-V2）

日期：2026-11-20。阶段2，不是正文，不构成写作授权。全部路径相对项目根目录。

## 总体结论

当前材料能支撑平台、控制、阵列布局与读出流程的技术描述，以及指定历史Episode的有限功能证据。进入阶段3应为**有条件就绪**：先由用户确认如何在正文区分“当前固定腱模型实现”和“历史任务记录”，不能将二者自动视为同一冻结版本。没有启动仿真、训练、服务器操作、P52/P53，也未修改正式TeX、参考文献、模板或历史数据。

证据等级：L0只定位；L1已读报告/配置的声明；L2只对实际核对的函数、XML字段或指定数组有效，不向整个系统或全部数据外推。Python AST和XML解析不是运行时动态验证。

## 篇幅与实际正文状态

当前主文件仍使用六章骨架，`thesis/body/ch02_data_and_problem.tex`是旧占位；V2对应`body/ch02_simulation_platform.tex`尚未创建。2.1–2.7实际正文统计均为MISSING（尚无对应正式正文可测），不能把材料报告计入TeXcount。旧占位不代表V2第二章完成量。本轮不改include、不运行编译。以下预算总和8500，范围8000–9000。

|小节|标题|预算|建议落点|实际正文|准备情况|
|---|---|---|---|---|---|
|2.1|系统需求与总体架构|700–800|750|MISSING|可写：模块接口及数据流；不将接口存在当作全部场景验证。|
|2.2|RM75B机械臂与Dex Hand灵巧手模型|1300–1450|1350|MISSING|有条件可写：明确当前固定腱模型，历史机构版本不能混写。|
|2.3|灵巧手接触建模与控制方法|1300–1450|1350|MISSING|可写：当前代码与指定profile；历史各任务配置不能自动沿用。|
|2.4|复杂曲面触觉阵列建模|1700–1950|1800|MISSING|可写算法：真实网格派生布局；未执行新几何绘图、面积认证或物理标定。|
|2.5|712维触觉观测及处理流程|1000–1100|1050|MISSING|可写：native touch及指定processed链；不能称真实压力。|
|2.6|仿真平台功能验证|1600–1800|1750|MISSING|有条件可写：五条历史开发样本的数组核验；当前模型动态验证UNKNOWN。|
|2.7|本章小结|400–450|450|MISSING|可写：限定已核验范围，承接第三章。|

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

|Patch|行×列|通道区间（零基、右开）|策略名|
|---|---|---|---|
|skin_0_0_p|4×8|[0,32)|segment|
|skin_0_1_p|4×8|[32,64)|segment|
|skin_0_2_p|7×8|[64,120)|fingertip-ellipsoid|
|skin_1_0_p|4×8|[120,152)|segment|
|skin_1_1_p|4×8|[152,184)|segment|
|skin_1_2_p|7×8|[184,240)|fingertip-ellipsoid|
|skin_2_0_p|4×8|[240,272)|segment|
|skin_2_1_p|4×8|[272,304)|segment|
|skin_2_2_p|7×8|[304,360)|fingertip-ellipsoid|
|skin_3_0_p|4×8|[360,392)|segment|
|skin_3_1_p|4×8|[392,424)|segment|
|skin_3_2_p|7×8|[424,480)|fingertip-ellipsoid|
|skin_4_0_p|4×8|[480,512)|segment|
|skin_4_1_p|4×8|[512,544)|segment|
|skin_4_2_p|7×8|[544,600)|fingertip-ellipsoid|
|skin_palm_p|7×16|[600,712)|mesh-uv|

## 2.5 原生读出与处理核验

调用链：builder→tactile.augment_spec→SimpleBox._add_taxel创建box site及mjSENS_TOUCH→bind按sensor名称验证dim=1并保存sensor_adr→read_raw从sensordata地址取float32副本→TactileSignalProcessor.process→processed712。sensordata地址不同于sensor编号。历史P40五个编译模型的记录均为730个sensor，其中712原生touch、18执行器传感器，插件实例0、flex0；这是历史报告L1，非本轮加载模型结果。

原生touch非负标量累加感知区内相关接触的法向力；SI模型约定下可解释为模拟力N，但不除面积、不含三轴切向力，不能说是认证真实力或Pa。processed经过死区、截断、幂响应、IIR、Patch邻域传播、可选高斯/噪声及归一化后，不能保留N单位。重叠site可能重复计同一接触力，712总和不自动等于总外力。

指定profile启用deadzone=.002、saturation=1、exponent=.7、alpha=.45、crosstalk=.12、normalize=true；gaussian_sigma=0、noise_std=0，因此不能写本profile启用了高斯扩散或噪声。处理顺序及公式见公式档案。crosstalk使用8邻域、按源邻居数量分配；Gaussian辅助函数虽名reflect，实际np.pad采用edge，不能照函数名写镜像边界。滤波reset初态也须按实际reset(size)路径说明，不能笼统保证无启动瞬态。

touch_grid插件在历史本机注册列表中存在不代表生产模型使用它。历史本机MuJoCo为3.10.0，服务器当前版本及插件实例UNKNOWN。不能用本机历史版本代替服务器证据。

## 2.6 历史功能证据核验

本轮只读以下五份NPZ的sensor_tactile、timestamp、字段shape，并核对result/manifest。五份均float32[T,712]、finite、非负、非全零，时间增量约0.05s。以下非零率是对源数组重新统计，不是引用旧summary。

|任务案例|T|非零比例|finite/非负|历史success|当前模型版本一致性|
|---|---|---|---|---|---|
|lift_sugar_rgb_00|447|0.084049720|是/是|True|UNKNOWN|
|pick_tomato_rgb_00|638|0.152964408|是/是|True|UNKNOWN|
|stack_ring_rgb_01|376|0.182632829|是/是|True|UNKNOWN|
|push_tomato_rgb_00|605|0.004954035|是/是|True|UNKNOWN|
|nut_retain_lower_rgb_00|599|0.127175911|是/是|True|UNKNOWN|

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
