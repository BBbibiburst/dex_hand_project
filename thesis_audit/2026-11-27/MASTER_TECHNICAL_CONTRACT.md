# 全篇技术定义总账

日期：2026-11-27。仓库根目录 `/home/zmy/dex_hand_project`。本轮为Stage A，只读静态审计及既有标量/登记复核，不产生L3，不修改论文或实验，不编译、不训练、不推理、不运行仿真，不commit/push。路径均相对仓库根目录；`文件:行`为可定位源码/CSV/JSON行，CSV行号含表头。PDF页码一律为文件物理页号，不能与印刷页码混用（本版本正文区相差8页）。

证据标记：VERIFIED=本轮直接核对所列代码/合同/既有结果；UNKNOWN=材料不足；UNVERIFIED=本轮未作所需深度认证；CONFLICT=来源之间实际冲突。直接读取历史结果文件仅证明其记录内容，不升级为本轮重新执行。L2只限所列范围。

## 命名基准

当前论文称“途见科技自研欠驱动灵巧手”；第一章:12给出“途见科技（北京）有限公司”。Dex Hand是用户确认的内部项目编号，保留在代码路径/资产/旧发行快照中，不再作为公开型号写回正文。RM75B为机械臂型号；厂商Realman/睿尔曼的品牌全称本轮未另作联网认证，本文技术核验以`source/robots/arms/rm75b.py`及当前XML为准。用户任务书中的Dex Hand为旧称索引，不构成要求恢复内部名称。

## 一、跨章定义与实现证据

下表“首→后”指实质定义与主要使用位置，不以绪论泛称作为完整定义；“历史/差异”明确哪些不能回填。

|统一名称|正确定义|首→后|直接代码/配置证据|跨章及历史判定|
|---|---|---|---|---|
|机械臂自由度|7个转动关节，7个position驱动通道|2.2→3/4/7|`assets/robots/rm75b/rm75b.xml:110–146`|VERIFIED；不是7维笛卡尔动作|
|欠驱动手|12转动关节，6固定腱、6位置通道；四指各1腱带2关节，拇指对掌1关节+屈曲3关节|2.2→3/4/7|`assets/grippers/dex_hand/dex_hand.xml:330–371`|VERIFIED；不含显式slide/equality闭链/flex；历史backend字符串不改变模型|
|安装坐标|手部附着末端body/site并应用给定旋转；对象目标转换须补偿安装旋转|2.2→3.3|`source/robots/arms/rm75b.py`；`source/grasping/executor.py`目标变换|本轮核对结构，实机几何误差UNKNOWN|
|物理积分|Builder最终0.001s、Newton、100次迭代、椭圆摩擦锥|2.3→3/4|`source/robots/builder.py:126–129`|XML原PGS不代表最终组合配置|
|控制/保存|指定profile0.05s控制，50物理子步即20Hz控制/1000Hz积分；RGB可独立20/5Hz|2.3→4.2|`ch02:174`、`production_recorder.py:75`、`production_capture.py`合同|不是传感器带宽；不宣称全部服务器视频参数一致|
|IK与position|IK解释末端位姿并求关节目标；底层MuJoCo位置伺服；物理推进是第三层|2.3→3.3/3.4|`source/control/arm.py:392–412`|无独立通用力矩PD；不将端点IK作为全轨迹认证|
|零空间与姿态权重|默认use_nullspace=True、gain=.1；posture_weight=0只关闭附加姿态残差；阻尼SVD投影I−J⁺J|2.3→3.4|`source/control/arm.py:48–51,386–406`|VERIFIED，fallback不叠加该项；各调用可覆盖|
|39D策略state|13 actuator_length+13 actuator_velocity+13上一位置命令|4.1→4.2/7|`source/imitation/observation.py:31–41`|不包含自由物体完整qpos，不是39关节；记录路径决定“上一命令”是否正确|
|13D position_action|臂7关节目标+手6腱目标|3.4→4.1/7|`source/imitation/observation.py:53–69`|臂rad与腱模型长度坐标不能一概作角度|
|13D ik_action|xyz(3)+四元数(4)+手命令(6)，是目标非实测位姿|3.4→4.1|`ch04:23–35`、`source/data/production_recorder.py`|与ctrl同维不等价；旧action字段须按来源解释|
|完整仿真状态|qpos/qvel及必要积分辅助状态，维数随模型变化|4.1/4.3|`production_recorder.py`、P38 comparison.json:18–54|不作为全部策略可观测输入|
|16 Patch/712D|五指每指3 Patch再加掌；每帧固定行主序通道，内部索引可追溯|2.4→4–7|`source/sensors/tactile/dex_hand.py:294–427`；`ch02:254–282`|不证明Real电气/物理通道配准|
|native touch raw|每site一个接触法向力标量读出；重叠site可重复覆盖|2.5→5.1|`dex_hand.py:373,400,420–427`|不是压力场，不直接除电极面积|
|simple_box与插件|实际712D来自mjSENS_TOUCH；site是有体积感知区域|2.4/2.5|`dex_hand.py:35–36,294–373`|不以touch_grid插件当712D直接来源|
|Sim processed|raw经死区、截断、幂、IIR、Patch内邻域扩散、可选高斯/噪声、归一化|2.5→4–6|`signal_processing.py:162–187,189–203`|d=.002,s=1,p=.7,alpha=.45,c=.12；gaussian/noise=0。第二章“串扰”与5.1“邻域扩散”是同一算法，非实机标定|
|Real processed|已有真实端处理数值；独立来源/尺度合同|4.5→5–7|P43 Real index及P44 data_contract.csv:2–3|原驱动/共同物理单位UNKNOWN，不把Real raw等同MuJoCo raw|
|Mask|M=1[X>0]；输出Decoder以logit>=0决定预测Mask|5.1→5.4–6|C3 `models.py:12,22,28`|微小数值正值也算激活；不是真实接触真值|
|4×712窗口|保留四帧顺序和712通道；Real帧间隔不均匀|5.1/5.2→6|P43 index/manifest；P44 data_contract.csv:2–3|帧索引差分非物理导数；与Euler步索引独立|
|GMM统计空间|各Patch二维网格的响应质心偏移分布；不是VAE潜空间|5.3→5.5|C3 `code/inverse.py:47–56`；GMM拟合源码见专项证据|拟合中心化与逆构造固定坐标原点不同，正文5.3:40/82已披露，启发式非平移一致逆映射|
|软责任度/Top5|责任度分解响应m；全部分量按m取最多5；已执行最多3候选中argmin再准入|5.3→5.5/5.6|C3 `inverse.py:47–56,67–87`及`expand.py`|不是物理归属概率；找到通过候选不保证最终登记；未执行候选不在搜索集|
|独立双VAE|两套相同结构独立参数，域内重建，各输出4×64|5.4→5.5/6.1|C3 `models.py:7–33`、`pipeline_cpu.py:27–56`|不共享Encoder/Decoder；同维不天然对齐|
|Encoder|[Mask,log1p/s]每帧1424→Linear128/GELU/LN→GRU128→均值/对数方差各64；lv截[-8,4]|5.4→6.1|C3 `models.py:7–12`|训练重参数化，推理posterior mean；不跨窗口传递GRU状态|
|Decoder|BOS32；前帧Mask嵌入32+潜64→GRUCell128；Mask712；当前Mask嵌入+状态→两分量Gamma|5.4→5.5/6|C3 `models.py:13–28`|Teacher Forcing用真实前帧Mask及当前幅值条件；当前Mask亦监督BCE；自由运行用预测Mask|
|幅值与损失|Gamma混合均值乘positive_scale；正值NLL有−log(scale)密度Jacobian；每窗正值数归一化|5.4→5.5|C3 `models.py:22,27–33`|NLL可负；C3总损失BCE+正值NLL+.001KL；条件NLL非自由迁移质量|
|Translator|2层GRU64→hidden128→Linear64，181824参数；4帧后验均值序列|5.5→6.1|C3 `models.py:34–36`、`pipeline_cpu.py:59–76`|仅T更新，固定Decoder仍向输入传梯度；Huber+.1(BCE+GammaNLL)，非直接分布对齐|
|共享向量场|T基项+c修正，c输入132→128/GELU/GRU128→64，末层零初始化|6.1→6.2–6.5|P44 `code/model.py:6–10`|306176可训练，完整G2494640；两个子网权重各自独立，跨Euler步共享整体参数|
|Euler|H0=0，Hk+1=Hk+v/K；K1/2/4/8为计算离散步|6.1→6.3–6.5|P44 `model.py:13–29`、`train.py:45–58`|tau非真实时间；每次field调用GRU重新沿四帧计算，不跨Euler传隐状态|
|全局D及Mask STE|5696→192/LN/LeakyReLU→64→1；硬Mask支路中断，连续支路可导|6.2→6.3|P44 `model.py:30–34`；P45 `code/`对应ablation实现|A连续2848、B硬Mask、C硬前向代理反向；单Seed/容量差异不作纯因果归因|
|Refiner|[log1p(Y0),zs]776→64/GELU/GRU64→712，输出层零初始；在log域修正再expm1|6.4→6.5|P50 `code/architecture.py`及6.4:6–15|120968可训练；非直接Y0+r；flow不直接约束Refiner残差|
|D50多视角|Global2848、Spatial192、Temporal2136，各128/LN/64/1；Joint取三支都接受，分数min logit|6.4→6.6|P50 `code/architecture.py`；parameter_inventory.csv:2–5|688643参数；50是实验标识非50轮；F50是生成器，D50是判别器|
|三类评价器|原11冻结C0；转换后重训Probe；逐轮D及D50|5.1/5.6→6|C0 training_registration及frozen；P49 registration；P51 registration|彼此不能互换；分类阈值、数据/组权重和角色需绑定|
|BA/AUC/D与FR|D=1−2|BA−.5|；BA与D门槛等价；AUC排序；FR是当前D接受生成比例|5.2→5.6/6/7|C3 c0.py；P49 round_metrics.csv；P51 classification_metrics.csv|C0点估计按窗口；FR Root宏平均、Real接受率Session宏平均；不能混用|

## 二、关键训练合同（实现与历史实际执行分开）

|版本|Seed及数据|实际训练/选择|冻结范围与区别|
|---|---|---|---|
|C3 Real VAE|4301，5153 Real Train|45 Epoch/3645步；30–45资格范围选30；batch64；AdamW lr.001 wd.0001|域内独立；complete.json:2–13；不是全45轮无约束择优|
|C3 Sim VAE|4302，4049 Natural+1200 Paired|各32窗/批等量混合，每轮127步，45轮5715步选30|小Paired组重复访问；Scaler以合并池拟合，不等于等权训练采样|
|C3 Translator|4303，Paired1200/240|19步/轮，45轮855；选30=570次更新；AdamW lr.0003 wd.0001，clip5|仅T更新，Natural不直接参加原配对训练；模型选择Huber+.1NLL|
|C13|Seed4313（registration.json:2），不回填C3 Seed|只修Real Decoder current与amplitude；训练细数见原training_summary|Encoder/Mask/循环骨干冻结，594512参数；跨域仍1/11|
|C23|既有C13自重建输出|只读诊断，不声称新训练|不是Natural迁移6/11|
|C24|Seed4313（registration.json:2）；18轮，best2，1458步|与C13同可训练范围另加时序幅值项|不替换C3 Sim VAE/Translator；2/11为Real读出诊断|
|P44/P45|4401及原A/C配置，5153/4049 Train|2轮D预热+18轮对抗；K调度每64批每K16次打乱；D1280/G1152|VAE固定；早期教师1→0，不把P49无教师倒填|
|P49|4901，K1，最多3轮|D0 20轮1320步；D1–3各10轮660步；G各20轮1280步，总D3300/G3840|从A终点F0继承G/优化器；D重新创建，CE门槛修订公开|
|P50|5001，F0重新开始|D10轮660，G20轮1280，固定终点|VAE固定；场+Refiner427144；非P49 F3续训|
|P51|无新增训练Seed|训练/优化0；已有冻结模型只读审计|不改D50、不反转AUC/阈值、不外推D3|

C3编码尺度Real=0.14118273556232452、Sim=0.04088745638728142；正值尺度Real=0.005002501420676708、Sim=1e−7（各complete.json:12–13）。输入只除尺度不减均值；与6.5的Real Train逐坐标中心化及std下限.05是不同用途的预处理，不得合并叫“同一Scaler”。

## 三、代码/正文一致性及未认证内容

已核对上述关键结构、公式方向、默认控制开关、参数数量和训练选模；未发现需要改写研究结论的P0事实冲突。尚未认证：历史Episode与当前完整模型一致性、真实驱动/物理通道标定、全服务器数据质量、跨域物理语义与实机任务收益。

未将“未认证”解释为“不可能实现”，也未将旧C3与C13/C24/C27/P50有利局部结果合成一个模型。P48与P49/P50的SWD合同可追溯复用；MMD数值必须绑定自己的采样/加权实现，P48 A_K1 MMD=0.3693778962与P49 F0=0.343677927不能当成同一原始统计值。当前6.5表使用P49/P50MMD，并未把P48值填入同一行，本轮不据此判定其数值错误。

C27 `config.json:4,15–19,39–49`：Seed4303，原pair Huber+.1固定Real Decoder NLL外，新增Real坐标矩约束lambda_m=.2985902260964023和源Patch份额lambda_s=.42934519902173574，lr=.0003，最大45轮；不与C13组合。P50 Refiner和多视角层次直接核对`code/architecture.py:6–30`，与当前图6-6一致。GMM曲面几何属于第2章模型采样而非第5章统计GMM，二者不混称。

补充直接源码定位：GMM拟合使用`outputs/tactile_gmm_dual_vae_2026-11-16/code/fit_gmm.py:8–38`，其中15行明确samples=grid−响应质心，23–25行仅从P37 Real Train取数据并按Patch正值P5筛选；C3 inverse.py:47–54责任度在固定grid计算，matching.py:6–8以pi混合均值与完整混合协方差构造锚点/误差。5.3披露的坐标不一致有直接代码依据，不是审计推测。原生touch与图6-1/6-6的读出结构亦按当前源核对。

第二章曲面采样：`source/sensors/tactile/fitting/segment.py`、`fingertip.py:1–45`、`projection.py:19–29`分别支持PCA/Bezier外壳、swept-shell指尖和mesh-uv投影；fingertip-ellipsoid为配置历史名称，当前函数明确非理想椭球。对应正文2.4:199–232已准确区分，不增加一个“名称即实现”的假问题。GraspQP官方工厂调用见`source/grasping/graspqp_adapter.py:72,101,144`；候选IK只预抓取与抓取两个目标见`source/grasping/executor.py:208–225`，与A05审查依据一致。
