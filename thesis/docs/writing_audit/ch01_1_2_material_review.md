# TH-CH01-1.2-V1 材料检索与核验报告

日期：2026-11-19。状态：阶段2完成，待用户与ChatGPT审核；未进入阶段3。

## 1. 核验结论与使用方法

已建立25篇候选文献的题录、全文来源、实际阅读范围及论点索引，覆盖四个研究方向。全文PDF均已取得；阅读的是与本节相关的摘要、方法与实验选段，**没有把取得PDF等同于通读全文**。S01是综述，核对分类与可达性讨论，不虚设其原创实验。每篇的PDF序号、文件Hash、版本及局限见`ch01_1_2_sources/literature_catalog.json`和`reading_log.json`。本报告是写作材料与比较提纲，不是1.2正式正文。

材料足以在审核后启动有范围限制的1.2写作。两个约束必须保留：

- 不能仅以“独立Dual-VAE＋潜变量Translator”的组合主张方法首创。S25已采用分别学习触觉仿真形变和真实电信号表示、再监督学习潜投影的路线。本文需要比较传感器、监督构造与评价条件，而不能忽略这一先例。
- 不能将判别器接受率、潜表示混淆与完整触觉重建质量视为同一问题。S18的理想判别器结论、S22的任务表示目标与S23/S24的分布评价分别有自己的假设；项目中的冻结C0、重训Probe、逐轮D必须分开。

本轮未修改`.tex`、`reference.bib`、规划、模板或历史产物；未训练、仿真、编译或执行Git提交。保护范围核对结果见`integrity.json`。原始数据只作已审核证据文件的Hash复核，没有读取Test/Holdout数组。

## 2. 与V2规划对应的建议结构

核对`THESIS_PLAN_V2.md:31`：1.2卡要求按研究问题分类、明确开源方法与本人适配，建议技术路线与监督条件比较表；**该卡没有给出固定的1.2.x标题**。以下为待审核建议，不是对既有目录的修改。

|建议小节|独立论证任务|与V2研究范围及后文章节的关系|建议字数/段落|
|---|---|---|---|
|1.2.1 灵巧手抓取生成与可执行操作|候选、稳定性、整机执行的分层及连接|第三章；平台约束仅作为接口，不展开实现|850–1000字，3段|
|1.2.2 机器人仿真与触觉感知建模|动力学接触、传感器形变/读出和标定的区别|第二章|700–850字，2段|
|1.2.3 专家示教与多模态操作数据|示教如何获得、如何组织与验证质量|第三章讲生成，第四章讲关联、存储与审计|750–900字，2–3段|
|1.2.4 触觉Sim-to-Real与跨域重建|物理/配对/非配对路线、评价条件及本文问题|第五、六章；不足放本小节末尾收束|1200–1500字，3段|

合计3500–4250字。末尾不足分析计入1.2.4，不另增“所有相关工作总结”等重复小节。不按作者逐条列摘要，不要求每篇占一个句子。若用户不采用下级标题，也可保留四个论证块，由最终任务书决定。

## 3. 四个方向的技术脉络与段落论证卡

### 3.1 抓取：从候选质量到具体机器人执行

**论证单元一——对象表示与抓取约束。** 用S01建立解析与数据驱动路线的基本分类，说明对象几何、手型与任务先验会影响候选。综述是2014年文献，只作分类基础，不用它断言2026年“最新趋势”。接S04：能量优化与仿真检验使多指候选可以规模化生成。分析对象是抓取姿态和接触配置，不能直接转换成整机任务成功率。

**论证单元二——优化与物理反馈各解决什么。** S02以受限接触扳手QP改进力闭合能量，S03将解析种子送入物理仿真进行无梯度进化优化。这两类方法不是简单的“新方法优于旧方法”：前者提供可优化的候选质量目标，后者利用仿真反馈处理解析近似。不同论文手型、扰动、去重和稳定性规则不同，不横向搬用其百分比。

**论证单元三——姿态目标到执行。** S05明确区分候选生成与目标条件执行策略。联系本文时，只能说具体RM75B机械臂和六驱动手仍需适配其控制参数化、运动学与执行验证；不能说上述研究“从不考虑可达性”。本项目GraspQP适配实际调用官方指标、用Adam优化腕位姿与六驱动参数，不能写成完整复现原MALA*。DexEvolve-style也不能等同原论文全部异步进化和蒸馏系统。

**可支持的研究切入点**：构造适合本平台控制约束的分级候选—可达筛选—轨迹/任务验证链。**不可证实**：该组合全领域首次、具有全局最优保证，或仅凭候选力闭合证明真实机器人可执行。

### 3.2 仿真触觉：物理求解不等于传感器读出

**论证单元一——平台与传感器模型分层。** S06说明机器人动力学、接触与约束求解基础；S07说明GPU仿真与训练数据通道的规模化。二者不构成本项目性能对比。Isaac Gym、Isaac Sim、Isaac Lab分别处理，不把Gym基准替换成Sim结果；Isaac Sim官方产品资料仅作平台说明，不计入25篇研究论文。

**论证单元二——接触如何转成观测。** 横向比较S08的视觉渲染、S09的经验光学与形变传播、S10的FEM/相机映射、S25的BioTac形变到电信号。用它们说明传感器几何、材料、标定和读出机制不同。本文现有712D链路是原生`mjSENS_TOUCH`的标量观测经自定义处理，不能称为上述光学模型的等价实现，也不能将刚体接触柔顺参数描述成硅胶真实变形。

**可支持的切入点**：面向具体曲面阵列定义布局、通道、读出和处理合同，并给出局限。**不可证实**：维度相同意味着物理Taxel一一对应；processed幅值已标定为压力Pa；所有历史Episode启用了相同滤波。

### 3.3 示教数据：数据生成与质量保障分别论证

**论证单元一——人工示教与自动扩展。** S11的系统性消融可支撑示教质量与观测设计的重要性；S16说明自动示教不是仅复制数组，而是将物体中心的分段轨迹变换后实际执行与筛选，同时依赖已知子任务顺序等假设。遥操作、脚本/规划与轨迹复用按实际采集方式比较，不把自动生成表述为普适替代人工。

**论证单元二——多样性与模态。** S12关注跨环境、多物体和语言/目标条件学习；S13提供多模态标定和实机技能数据；S14以分布式真实场景采集扩展环境覆盖。区别任务、场景、轨迹、视频和接触模态的统计单位。不将其数据规模与本项目窗口/索引时长直接相加或排序。

**论证单元三（可与第二单元合并）——统一格式与有效关联。** S15的粗粒度动作/观测整合说明异构数据利用需要适配，而不是改为同一字段名就自动获得相同控制语义。本项目同仿真Episode内关联、不同真实设备记录关联、统计伪配对、同步物理配对四者分开。该段连接第四章的lineage、时间关系及质量审核，第三章不重复介绍LeRobot字段。

**可支持的切入点**：报告可追溯的来源、采样/动作关联和质量检查。**不可证实**：所有公开数据都含触觉；本项目全部数据已通过完整同步和独立性验收；更多数据/模态必然更好。

### 3.4 跨域：监督条件决定可以学到什么

**论证单元一——物理建模、随机化与标定监督。** S17只用作视觉域随机化的经典依据，不能扩大为已经验证材料随机化能处理本项目压阻响应。触觉层面的实证主要取S10和S25：前者基于材料/FEM和相机标定进行力场估计，后者在物理模型和真实电信号之间建立带监督潜投影。两者传感器、监督与任务均需明确。S10真实分布力监督涉及仿真赋值，不能误称传感器提供了独立的高分辨率真实力场真值。

**论证单元二——非配对表示与映射。** S20说明VAE概率表示，S19用循环约束限制无配对映射，S21明确提出共享潜空间假设并施加权重共享等约束，S22保持源域任务判别性同时抑制域信息。它们分别回答生成、对应约束和任务适应问题，不能仅列模型名称。两个边际分布不能唯一确定联合对应，这是S21明确讨论的建模困难；将此联系本文“独立VAE不天然对齐”属于有依据的比较判断，不把图像方法当712D触觉成功证据。

**论证单元三——如何评价转换。** 结合S18理想条件、S23核两样本检验、S24质量/覆盖的分解，说明分数接受、分布相似和输入依赖是不同证据。结构保持要求还需结合本文接触与时序合同，不能归称某篇图像论文已证明其触觉有效性。此处只提出评价维度，不列项目P49/P50数值、不把P51对D50的NO-GO扩大到P49。结尾收束为：具体传感器读出差异、有限监督的可信度以及真实性与输入结构之间的平衡，是本文数据条件下需要研究的问题，而不是宣称全领域没有解决方案。

## 4. 可用于正文的技术路线比较表草案

这是一张候选表，待写作任务书确认；不制作虚构实验图，不跨论文排名。必要时压缩成8–10行。

|技术方向/代表|主要输入与监督|解决的问题|适用条件及局限|
|---|---|---|---|
|可微抓取优化：S02/S04|物体几何、手模型、接触质量能量|生成多样抓取配置|力闭合/接触近似及特定验证协议，非整机执行保证|
|物理进化：S03|初始候选、仿真反馈|细化稳定性与多样性|依赖物理模型和采样预算；本项目仅适配实现|
|生成与策略执行：S05|点云、抓取目标及策略学习|连接候选与动作执行|依赖机器人、观测和控制设置|
|平台：S06/S07|动力学、约束、控制与仿真管线|接触计算及规模化试验|不直接实现任意传感器读出|
|视觉触觉：S08/S09|接触几何、光学/形变配置和校准|渲染触觉图像或标记运动|不等价于712D标量阵列|
|物理触觉迁移：S10/S25|FEM/真实传感器及校准或配对监督|力场回归、电信号/形变转换|监督获取与传感器假设明确|
|自动示教：S16|源示教、物体坐标及已知子任务序列|适配并执行新场景轨迹|不是任意控制/任务的通用生成|
|数据体系：S11–S15|遥操作、机器人记录与标准化|学习、场景覆盖和跨机器人利用|模态、动作语义及版本不完全一致|
|非配对生成：S18–S21|单域样本、生成与结构假设|生成或非配对转换|边际相似不认证物理对应|
|特征域适应：S22|源任务标签、两域观测和域标签|任务表示的域适应|特征域不可分不等价完整触觉可重建|
|分布评价：S23/S24|真实/生成样本及核/特征空间|两样本差异及质量/覆盖诊断|依赖统计条件，不独自评价输入保持|

建议只用这张表。技术路线分类图不是必需；如后续需要，应为明确标注的文献归纳示意图，不使用“实验曲线”样式，不混入本文尚未验证的性能结论。

## 5. 项目内部证据与L0/L1/L2边界

等级沿用现有档案：L0仅定位；L1核对报告/配置声明；L2仅对列出的代码段/实际文件核验，不代表全量最终验收。本轮复用1.1已审核档案并逐文件Hash核对，另只读复查关键函数；没有重新运行数据审计实验。

|论点|原始证据与定位|等级/本轮范围|允许表述及不能扩大的部分|
|---|---|---|---|
|机器人组合/驱动|`source/robots/arms/rm75b.py`；`source/robots/hands/dex_hand.py`；R-01|既有L2，Hash未变|仿真模型7臂驱动＋6手驱动；不是13个独立关节或已实机集成|
|原生712D读出|`source/sensors/tactile/dex_hand.py:360–429`；R-02|本轮L2代码选段|site关联touch、sensordata地址读取；不是touch_grid源码生成|
|处理链|`source/sensors/tactile/signal_processing.py:162–213`|本轮L2代码选段|按配置处理；某处理项存在不等于全部产物均启用|
|抓取来源与适配|`source/grasping/graspqp_adapter.py:64–110`；`source/grasping/dexevolve.py:1`|本轮L2代码选段|官方指标＋本项目欠驱动适配，DexEvolve-style；非原创第三方算法|
|可达性层|`source/grasping/executor.py:161–223`；`source/rl/grasp_edit/templates.py`|前者本轮L2，后者复用1.1 L2且Hash相同|预抓取/抓取IK误差；不能代替完整连续碰撞证明|
|Strict与普通回放|`source/verification/strict_replay.py`；`source/grasp_pipeline/replay.py`|复用1.1 L2、Hash相同|两种接口标准不同，Lift验证不传播到全部任务|
|五任务开发案例|`outputs/five_task_p32_b_2026-11-03/`的五条`source_episode/episode.npz`；R-05|复用1.1指定文件L2，Hash相同|这些开发产物有finite非全零712D；不证明全量质量/独立Root或实机成功|
|LeRobot实际产物|`outputs/dp_2026-10-11/lerobot_5/`与`outputs/dp_p1_2026-10-11/lerobot_formal30/`；R-06|复用已审核文件检查，不新作全量扫描|指定导出存在，记录与声明一致；5/30的源独立性、视频PTS和全量因果关联仍未认证|
|采集与质量机制|`source/data/production_recorder.py`、`source/data/production_queue.py`；R-07/R-08|既有L2所列实现；本轮不启用采集|合同检查及去重机制存在；不宣称服务器当前最终规模或无损恢复|
|统计伪配对|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/matching.py`；E-C3|复用1.1 L2代码、Hash相同|统计描述指导构造，不是同步物理配对或认证Taxel对应|
|迁移评价边界|`thesis/EXPERIMENT_EVIDENCE.md`中E-P48至E-P51及其原始目录|本轮读取已审核索引；对应既有L2范围有效，不新增复算|F0距离改善、P49/P50质量问题和P51审计范围同时保留；1.2不提前展开项目结果|

完整路径及Hash见`internal_evidence_recheck.json`；本轮新增代码阅读边界见`internal_source_reading.json`。未追溯到实际模型/统计的文档总结不能提升为L2。该表中的历史等级与“本轮独立复算”不是同一件事。

## 6. 国内研究及来源核验

- S04 DexGraspNet：PDF首页明确北京大学、北京通用人工智能研究院和清华大学机构，支撑国内多指抓取生成方向。
- S05 UniDexGrasp：首页列北京大学、北京通用人工智能研究院、清华大学及斯坦福大学，为跨机构合作，不按作者姓名判定“国内”。
- S13 RH20T：首页明确上海交通大学，支撑国内多模态实机技能数据方向。
- S09 Taxim的论文机构为卡内基梅隆大学，不能因为作者中文姓名把它列为国内机构工作。

本轮没有为了“国内国外各一半”加入低相关论文。材料足以按技术问题比较国内外代表工作，但不足以得出国家/地区整体研究水平排名。若后续任务书要求独立的国内压阻阵列标定综述，还需定向补充，不能用光学触觉论文冒充。

## 7. 版本、元数据与事实冲突

|项目|核验结果|正文处置|
|---|---|---|
|GraspQP|已找到CoRL2025正式PMLR305记录和PDF|按正式论文引用，不写成未发表项目名|
|DexEvolve|RSS2026 paper059、DOI及正式PDF匹配|可引用正式研究；不把本地style实现等同完整系统|
|Isaac系列|Gym论文为NeurIPS Datasets and Benchmarks2021；Sim官方产品资料另存|不混用平台名称与速度基准|
|S10年份|在线2021、卷期2022，Soft Robotics9(5)|正式题录用2022，版本注中保留在线信息|
|BridgeData V2|PMLR作者顺序与PDF存在差异，摘要与PDF规模也不同|采用PMLR正式题录，不在本节以冲突数字排名|
|DROID|RSS网页摘要与正式PDF表的规模数字不同|论述协议/场景多样性；若未来引用规模必须绑定具体版本|
|OXE|正式ICRA2024题录，已读PDF是更新的arXiv v9（2025）|仅使用可明确定位的整合原则；不将更新实验数值称已核正式出版结果|
|DANN作者|JMLR网页元标签写Mario March，PDF署名Mario Marchand|按原文署名，目录保留说明|
|CycleGAN页码|IEEE题录2242–2251，CVF打印页2223–2232|正式题录沿用IEEE；阅读定位用PDF序号|
|DOI缺项|部分PMLR/JMLR/NeurIPS页面未列DOI|空值不是“证明不存在”；用可追溯URL，不生成猜测DOI|
|最接近的既有方法|S25已有独立VAE和监督潜投影|对照数据监督和物理读出条件，不以结构名称宣称创新|

局限栏包含原文设置直接支持的范围限定及本材料报告的比较判断，不把后者冒称原作者结论。题名、作者、年份和来源已整理入JSON；S14长作者列表完整保留，S15以团队署名展示并另存Crossref个人作者列表。正文引用阶段再按学校样式处理，不在本轮生成BibTeX条目。出版社HTML、DOI注册记录和访问失败日志均保留，未将ETH429、PMC验证页描述成成功阅读。

## 8. 建议引用位置与谨慎表述

- 1.2.1第1段S01/S04，第2段S02/S03，第3段S05。抓取质量与机器人执行的区别用原论文任务定义加项目控制/IK证据，不能仅凭规划推断。
- 1.2.2第1段S06/S07，第2段S08/S09，S10/S25的迁移细节留到1.2.4，避免重复介绍同一方法。
- 1.2.3第1段S11/S16，第2段S12/S13/S14，第3段S15。只比较模态/协议与适用条件，不列巨大的规模榜单。
- 1.2.4第1段S17/S10/S25，第2段S20/S19/S21/S22，第3段按需S18/S23/S24。经典论文只用来说明方法假设，重点篇幅留给触觉实证及监督条件。

25篇是候选上限，不要求全篇逐一展开。可将S01、S07、S20、S23等压缩为背景合并引用，实际正式使用约21–25篇；删减只能基于论证需要，不为了凑数量。

允许写“现有路线依赖各自的传感器、监督和结构假设”“本平台仍需验证欠驱动适配与执行”“本文数据缺少已认证同步物理配对”。不允许写“首次提出双VAE触觉迁移”“目前没有触觉仿真”“域分类分数达到目标即完成真实分布逼近”“所有公开数据已完全同步”“本文获得了独立盲测成功”。

## 9. 尚存缺口、审核项与阶段判断

**不阻碍本节起草、但必须限制表述的缺口**：真实Taxel物理映射尚未认证；本项目部分导出的源lineage、PTS和最终质量未全部认证；正式OXE全文与v9未逐行比对；本节没有对2026全部文献做系统综述覆盖率检验；未形成可支撑“方法原创性”结论的全面查新。完整参考文献并不自动填补这些项目证据缺口。

**需要用户/ChatGPT审核的事项**：

1. 是否采用本报告建议的四个1.2.x标题；不涉及七章结构调整。
2. 是否接受将S25作为第五章方法背景中的必引近邻工作，明确其监督配对与本文统计伪配对的差别。
3. 是否同意本节回避版本冲突的数据规模数字，改比较协议与监督条件。

**阶段判断：材料准备GO，正文写作仍待审核及最终任务书。** 已具备技术脉络、代表方法、横向比较和有边界的不足分析。GO不代表当前方法创新已经认定，不代表触觉迁移通过验收。不得据此自行写1.2或启动下一节。

## 10. 文献逐条核验清单

以下题录的页码是出版页码（如已有正式记录）；“实际阅读”中的页码是本地PDF序号。两者不混用。完整作者、元数据证据和文件Hash以机器可读目录为准。

### S01　Data-Driven Grasp Synthesis—A Survey

- **题录**：Jeannette Bohg; Antonio Morales; Tamim Asfour; Danica Kragic. Data-Driven Grasp Synthesis—A Survey. IEEE Transactions on Robotics, 30(2): 289–309, 2014.
- **来源与DOI**：[原始记录](https://arxiv.org/abs/1309.2660)；[DOI](https://doi.org/10.1109/TRO.2013.2289018)。
- **路线与原文支持**：解析规划与数据驱动抓取综述。从对象表示、先验和数据来源梳理抓取；讨论候选可达性筛选。
- **适用边界**：2014综述不代表截至2026的全部进展；不提供本项目六驱动适配结论。
- **实际阅读**：PDF第3,4页相关段落；分类框架及已知物体可达性段落；综述没有单独新实验。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：候选生成、稳定性与执行约束需要区分。

### S02　GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping

- **题录**：René Zurbrügg; Andrei Cramariuc; Marco Hutter. GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping. Conference on Robot Learning, PMLR 305:2583–2602, 2025.
- **来源与DOI**：[原始记录](https://proceedings.mlr.press/v305/zurbrugg25a.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：隐式QP力闭合能量与MALA*优化。将受限接触扳手组合写为可微QP，以改进解析抓取候选质量和多样性。
- **适用边界**：解析接触及扰动测试有特定假设；不证明RM75B完整轨迹可达。
- **实际阅读**：PDF第3,4,5,7页相关段落；定义、QP方法、跨手型实验表及评价设置的选段。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：优化路线和本项目GraspQP指标适配的来源。

### S03　DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis

- **题录**：René Zurbrügg; Andrei Cramariuc; Marco Hutter. DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis. Robotics: Science and Systems XXII, paper 059, 2026.
- **来源与DOI**：[原始记录](https://roboticsproceedings.org/rss22/p059.html)；[DOI](https://doi.org/10.15607/rss.2026.xxii.059)。
- **路线与原文支持**：仿真闭环进化优化与扩散蒸馏。利用Isaac Sim物理反馈选择、变异和交叉，改进解析种子抓取。
- **适用边界**：原论文包含异步搜索和蒸馏；本项目DexEvolve-style不自动等于完整复现。
- **实际阅读**：PDF第3,7页相关段落；正式RSS版方法概述、表示及跨手型/收敛结果选段；另读预印本3、6、8页选段。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：解析候选→物理优化→执行的不同层次。
- **版本核验**：RSS正式版11页与预印本17页不同，页号已分别标注。RSS网页编码异常以Crossref和PDF中的René Zurbrügg为准。

### S04　DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation

- **题录**：Ruicheng Wang; Jialiang Zhang; Jiayi Chen; Yinzhen Xu; Puhao Li; Tengyu Liu; He Wang. DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation. IEEE International Conference on Robotics and Automation (ICRA):11359–11366, 2023.
- **来源与DOI**：[原始记录](https://doi.org/10.1109/icra48891.2023.10160982)；[DOI](https://doi.org/10.1109/icra48891.2023.10160982)。
- **路线与原文支持**：可微力闭合优化与仿真验证的数据集。将抓取姿态、手关节和接触点纳入优化，生成并验证ShadowHand抓取。
- **适用边界**：数据生成验证与学习基线采用不同成功判定；不能直接搬为具体机械臂成功率。
- **实际阅读**：PDF第1,2,3,6页相关段落；摘要、候选/能量定义、学习基线及放宽验证条件。作者预印本；正式题录另核，不宣称逐行核对正式全文；不声称整篇已读。
- **引用位置**：数据生成优化与仿真质量评价。

### S05　UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy

- **题录**：Yinzhen Xu; Weikang Wan; Jialiang Zhang; Haoran Liu; Zikang Shan; Hao Shen; Ruicheng Wang; Haoran Geng; Yijia Weng; Jiayi Chen; Tengyu Liu; Li Yi; He Wang. UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR):4737–4746, 2023.
- **来源与DOI**：[原始记录](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_UniDexGrasp_Universal_Robotic_Dexterous_Grasping_via_Learning_Diverse_Proposal_Generation_CVPR_2023_paper.html)；[DOI](https://doi.org/10.1109/cvpr52729.2023.00459)。
- **路线与原文支持**：多样候选生成与目标条件执行策略。分离候选姿态生成和目标条件策略，结合课程学习及教师学生蒸馏。
- **适用边界**：抓取策略的机器人、动作和任务假设明确；不直接覆盖RM75B欠驱动链。
- **实际阅读**：PDF第1,3,6,7页相关段落；摘要、分布参数化动机、策略课程及生成/执行实验选段。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：从几何抓取目标到可执行策略。

### S06　MuJoCo: A physics engine for model-based control

- **题录**：Emanuel Todorov; Tom Erez; Yuval Tassa. MuJoCo: A physics engine for model-based control. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS):5026–5033, 2012.
- **来源与DOI**：[原始记录](https://roboti.us/lab/papers/TodorovIROS12.pdf)；[DOI](https://doi.org/10.1109/iros.2012.6386109)。
- **路线与原文支持**：广义坐标动力学与接触求解。讨论多关节、接触、等式约束和执行器建模，并进行计时测试。
- **适用边界**：2012版本论文不能证明当前引擎全部功能；接触求解不是压阻读出模型。
- **实际阅读**：PDF第1,3,7页相关段落；引擎摘要、求解流程、模型元素和计时实验开头。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：机器人仿真提供动力学基础但不自动提供特定触觉读出。
- **版本核验**：本轮找回作者托管PDF；旧链接曾重定向，未回改1.1历史档案。2012论文不用于描述当前全部版本功能。

### S07　Isaac Gym: High Performance GPU Based Physics Simulation For Robot Learning

- **题录**：Makoviychuk, Viktor; Wawrzyniak, Lukasz; Guo, Yunrong; Lu, Michelle; Storey, Kier; Macklin, Miles; Hoeller, David; Rudin, Nikita; Allshire, Arthur; Handa, Ankur; State, Gavriel. Isaac Gym: High Performance GPU Based Physics Simulation For Robot Learning. NeurIPS Datasets and Benchmarks, volume 1, Round 2, 2021.
- **来源与DOI**：[原始记录](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/28dd2c7955ce926456240b2ff0100bde-Abstract-round2.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：GPU物理仿真与张量训练管线。直接连接物理缓冲区与GPU策略学习，以并行环境验证多类机器人任务。
- **适用边界**：硬件、任务和训练管线限定加速结论；Isaac Gym不等于Isaac Sim。
- **实际阅读**：PDF第1,3,7页相关段落；正式版摘要、GPU管线、任务与手部/堆叠实验。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：仿真平台的规模化与物理读出分层。
- **版本核验**：正式NeurIPS Datasets and Benchmarks论文，不再标为仅arXiv；不将Gym测速套给Isaac Sim。

### S08　TACTO: A Fast, Flexible, and Open-Source Simulator for High-Resolution Vision-Based Tactile Sensors

- **题录**：Shaoxiong Wang; Mike Lambeta; Po-Wei Chou; Roberto Calandra. TACTO: A Fast, Flexible, and Open-Source Simulator for High-Resolution Vision-Based Tactile Sensors. IEEE Robotics and Automation Letters, 7(2): 3930–3937, 2022.
- **来源与DOI**：[原始记录](https://arxiv.org/abs/2012.08456)；[DOI](https://doi.org/10.1109/LRA.2022.3146945)。
- **路线与原文支持**：几何接触驱动的视觉触觉渲染。分离物理场景同步与光学渲染，支持相机、光照及传感器配置。
- **适用边界**：针对视觉触觉；背景图校正不是任意传感器物理标定。
- **实际阅读**：PDF第3,4,7页相关段落；复用1.1方法与位姿估计实验阅读，本轮复查流程/计时/背景校正。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：触觉模拟的传感器特定假设。

### S09　Taxim: An Example-Based Simulation Model for GelSight Tactile Sensors

- **题录**：Zilin Si; Wenzhen Yuan. Taxim: An Example-Based Simulation Model for GelSight Tactile Sensors. IEEE Robotics and Automation Letters, 7(2): 2361–2368, 2022.
- **来源与DOI**：[原始记录](https://arxiv.org/abs/2109.04027)；[DOI](https://doi.org/10.1109/LRA.2022.3142412)。
- **路线与原文支持**：实例光学校准与弹性标记位移近似。结合经验光学映射和FEM标定的形变传播，模拟GelSight图像与标记位移。
- **适用边界**：文中速度比较限定CPU；成像和材料近似不能直接移植到712维标量阵列。
- **实际阅读**：PDF第2,3,5,6页相关段落；复用方法阅读；本轮核对形变标定、受控压入、图像实验及CPU限制。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：几何、材料和读出模型共同决定触觉观测。

### S10　Sim-to-Real for High-Resolution Optical Tactile Sensing: From Images to Three-Dimensional Contact Force Distributions

- **题录**：Carmelo Sferrazza; Raffaello D'Andrea. Sim-to-Real for High-Resolution Optical Tactile Sensing: From Images to Three-Dimensional Contact Force Distributions. Soft Robotics 9(5):926–937, 2022.
- **来源与DOI**：[原始记录](https://journals.sagepub.com/doi/10.1089/soro.2020.0213)；[DOI](https://doi.org/10.1089/soro.2020.0213)。
- **路线与原文支持**：FEM合成视觉触觉与力分布回归。在FEM中生成图像特征和力场监督，经相机校准后迁移到真实传感器。
- **适用边界**：真实分布力真值部分借助仿真赋值，需区别独立力计验证；不是任意无配对触觉转换。
- **实际阅读**：PDF第1,3,4,7,9,11页相关段落；预印本v3摘要、FEM方法、相机映射、测试指标与可视化选段。作者预印本；正式题录另核，不宣称逐行核对正式全文；不声称整篇已读。
- **引用位置**：模型校准与物理监督假设。
- **版本核验**：正式卷期2022，在线发表2021；Crossref默认published是在线日期。卷期由出版社和PubMed交叉核对。全文读取arXiv v3，不称已读出版社排版全文。

### S11　What Matters in Learning from Offline Human Demonstrations for Robot Manipulation

- **题录**：Ajay Mandlekar; Danfei Xu; Josiah Wong; Soroush Nasiriany; Chen Wang; Rohun Kulkarni; Li Fei-Fei; Silvio Savarese; Yuke Zhu; Roberto Martín-Martín. What Matters in Learning from Offline Human Demonstrations for Robot Manipulation. Proceedings of the 5th Conference on Robot Learning, PMLR 164: 1678–1690, 2022.
- **来源与DOI**：[原始记录](https://proceedings.mlr.press/v164/mandlekar22a.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：离线示教系统性学习与消融。研究示教质量、算法、观测和历史信息对模仿学习的影响。
- **适用边界**：实验任务和演示分布限定结论；不说明更多模态必然更好。
- **实际阅读**：PDF第3,5,7页相关段落；复用1.1研究设置、方法和消融；本轮复读观测/相机消融。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：数据质量与观测设计影响学习。

### S12　BridgeData V2: A Dataset for Robot Learning at Scale

- **题录**：Homer Rich Walke; Kevin Black; Tony Z. Zhao; Quan Vuong; Chongyi Zheng; Philippe Hansen-Estruch; Andre Wang He; Vivek Myers; Moo Jin Kim; Max Du; Abraham Lee; Kuan Fang; Chelsea Finn; Sergey Levine. BridgeData V2: A Dataset for Robot Learning at Scale. Proceedings of the 7th Conference on Robot Learning, PMLR 229: 1723–1736, 2023.
- **来源与DOI**：[原始记录](https://proceedings.mlr.press/v229/walke23a.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：跨环境遥操作/脚本数据与多任务学习。提供多环境、多物体操作数据，并验证目标/语言条件及跨环境利用。
- **适用边界**：不同版本规模不一致；不应据本节构造触觉同步或物理配对结论。
- **实际阅读**：PDF第3,4,6页相关段落；复用数据采集协议；本轮读模型及实验问题/设置选段。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：数据多样性与迁移利用，不做跨论文数字排名。
- **版本核验**：PMLR正式作者元数据顺序与所读PDF有差异，按已核验PMLR记录列作者；网页与PDF的轨迹数不同，本节不引用规模作排名。

### S13　RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot

- **题录**：Hao-Shu Fang; Hongjie Fang; Zhenyu Tang; Jirong Liu; Chenxi Wang; Junbo Wang; Haoyi Zhu; Cewu Lu. RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot. 2024 IEEE International Conference on Robotics and Automation (ICRA): 653–660, 2024.
- **来源与DOI**：[原始记录](https://rh20t.github.io/)；[DOI](https://doi.org/10.1109/ICRA57147.2024.10611615)。
- **路线与原文支持**：多模态实机采集与技能预训练。组织视觉、力/触觉、音频和机器人状态并验证预训练迁移。
- **适用边界**：部分实验使用所选子集；其模态关联不能替本项目证明跨设备同步。
- **实际阅读**：PDF第1,3,4,5,6页相关段落；复用采集/对齐/设置；本轮核对ACT实验结果与组别。作者预印本；正式题录另核，不宣称逐行核对正式全文；不声称整篇已读。
- **引用位置**：国内多模态数据采集和标定代表。

### S14　DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset

- **题录**：Alexander Khazatsky; Karl Pertsch; Suraj Nair; et al.（完整作者见JSON）. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. Robotics: Science and Systems XX, paper 120, 2024.
- **来源与DOI**：[原始记录](https://roboticsproceedings.org/rss20/p120.html)；[DOI](https://doi.org/10.15607/rss.2024.xx.120)。
- **路线与原文支持**：分布式真实场景遥操作采集。用统一机器人采集装置及标定扩大场景覆盖，并检验联合训练效用。
- **适用边界**：网页与PDF规模口径不同；无712D触觉同合同保证。
- **实际阅读**：PDF第1,3,6,8页相关段落；正式RSS版摘要、数据定义、场景多样性和策略评估选段。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：场景多样性、采集协议与下游验证。
- **版本核验**：以RSS正式元数据列全作者；Crossref部分姓名缩写不同。RSS摘要65k与正式PDF表76k存在版本差异，不混写。

### S15　Open X-Embodiment: Robotic Learning Datasets and RT-X Models

- **题录**：Open X-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. IEEE International Conference on Robotics and Automation (ICRA):6892–6903, 2024.
- **来源与DOI**：[原始记录](https://doi.org/10.1109/icra57147.2024.10611477)；[DOI](https://doi.org/10.1109/icra57147.2024.10611477)。
- **路线与原文支持**：跨机器人数据整合与RT-X。以粗粒度观测动作统一支持跨机器人训练，包含迁移及消融。
- **适用边界**：统一格式不等于动力学相同或模态全部齐全；当前阅读全文片段来自2025预印本v9。
- **实际阅读**：PDF第1,2,4,6页相关段落；预印本v9摘要、数据整合和跨机器人/消融实验选段；正式ICRA元数据单列。作者预印本；正式题录另核，不宣称逐行核对正式全文；不声称整篇已读。
- **引用位置**：格式标准化与语义/控制合同需分开。
- **版本核验**：正式出版2024；本地arXiv v9更新于2025。团队作者与完整个人作者元数据分别保存，不把<sup>0</sup>排版残片放进题名；不将v9新增结果冒充已核正式版结果。

### S16　MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations

- **题录**：Ajay Mandlekar; Soroush Nasiriany; Bowen Wen; Iretiayo Akinola; Yashraj Narang; Linxi Fan; Yuke Zhu; Dieter Fox. MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations. Conference on Robot Learning, PMLR 229:1820–1864, 2023.
- **来源与DOI**：[原始记录](https://proceedings.mlr.press/v229/mandlekar23a.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：物体中心分段轨迹变换与实际执行。通过分段、坐标变换及插值重用少量示教，在执行成功后形成新示教。
- **适用边界**：依赖已知物体中心子任务序列、位姿及末端动作接口；不是任意灵巧操作通用生成。
- **实际阅读**：PDF第1,3,7页相关段落；摘要、三项任务/动作假设、生成管线和数据/性能实验。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：自动示教必须适配控制和任务验证。

### S17　Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World

- **题录**：Josh Tobin; Rachel Fong; Alex Ray; Jonas Schneider; Wojciech Zaremba; Pieter Abbeel. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS): 23–30, 2017.
- **来源与DOI**：[原始记录](https://arxiv.org/abs/1703.06907)；[DOI](https://doi.org/10.1109/IROS.2017.8202133)。
- **路线与原文支持**：视觉域随机化。随机纹理、光照、相机和干扰物训练视觉定位并做真实评估与消融。
- **适用边界**：原实验主要是视觉随机化；不能直接支持压阻材料随机化已经有效。
- **实际阅读**：PDF第2,3,4,5页相关段落；复用方法阅读；本轮读随机化范围、真实定位和消融。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：以覆盖变化降低仿真依赖，限定到原验证模态。

### S18　Generative Adversarial Nets

- **题录**：Goodfellow, Ian J.; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua. Generative Adversarial Nets. Advances in Neural Information Processing Systems 27, 2014.
- **来源与DOI**：[原始记录](https://papers.nips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：生成器与判别器极小极大优化。给出对抗生成目标及理想最优判别器条件下的分布结论。
- **适用边界**：有限/冻结判别器被接受不等价理论最优分布相等；原实验为图像生成。
- **实际阅读**：PDF第1,2,3,5页相关段落；摘要、优化目标/交替过程、理论前提及实验设置选段。作者预印本；正式题录另核，不宣称逐行核对正式全文；不声称整篇已读。
- **引用位置**：对抗优化依据及判别分数的解释边界。
- **版本核验**：出版页未列DOI/连续页码，留空而非猜测；正文读取作者预印本。

### S19　Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks

- **题录**：Jun-Yan Zhu; Taesung Park; Phillip Isola; Alexei A. Efros. Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. 2017 IEEE International Conference on Computer Vision (ICCV): 2242–2251, 2017.
- **来源与DOI**：[原始记录](https://openaccess.thecvf.com/content_ICCV_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)；[DOI](https://doi.org/10.1109/ICCV.2017.244)。
- **路线与原文支持**：非配对对抗转换与循环一致性。用双向映射和循环损失约束无配对图像转换，并做消融。
- **适用边界**：不认证触觉物理语义及唯一对应；图像可视质量不能替代触觉结构指标。
- **实际阅读**：PDF第2,3,5页相关段落；复用方法阅读；本轮读有真值评价与无配对训练区分、实验/消融。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：非配对学习需要额外结构约束。
- **版本核验**：IEEE题录2242–2251；CVF版打印页2223–2232不同，保留差异，阅读范围均指PDF序号。

### S20　Auto-Encoding Variational Bayes

- **题录**：Kingma, Diederik P.; Welling, Max. Auto-Encoding Variational Bayes. International Conference on Learning Representations (ICLR), 2014.
- **来源与DOI**：[原始记录](https://dare.uva.nl/id/cf65ba0f-d88f-4a49-8ebd-3a7fce86edd7)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：变分下界与重参数化。通过概率Encoder/Decoder与KL正则学习潜变量生成模型。
- **适用边界**：两个独立VAE的坐标和潜语义不由同维度或高斯先验保证。
- **实际阅读**：PDF第1,3,6,7页相关段落；摘要、ELBO、训练设置及MNIST/Frey比较选段。作者预印本；正式题录另核，不宣称逐行核对正式全文；不声称整篇已读。
- **引用位置**：双域表示的基础与不能自动对齐的边界。
- **版本核验**：正式会议年份2014，预印本首发2013；会议录用状态由作者所属阿姆斯特丹大学机构库核实（accepted paper），全文为作者arXiv录用稿，不将预印本标识当会议DOI。

### S21　Unsupervised Image-to-Image Translation Networks

- **题录**：Liu, Ming-Yu; Breuel, Thomas; Kautz, Jan. Unsupervised Image-to-Image Translation Networks. Advances in Neural Information Processing Systems 30, 2017.
- **来源与DOI**：[原始记录](https://papers.nips.cc/paper_files/paper/2017/hash/dc6a6489640ca02b0d42dabeb8e46bb7-Abstract.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：共享潜空间假设下的非配对图像翻译。通过显式权重共享、VAE/GAN及循环约束实现双域转换。
- **适用边界**：共享潜空间是建模假设及网络约束，不是独立VAE天然性质；实验为图像。
- **实际阅读**：PDF第1,3,5页相关段落；摘要、共享高层函数、VAE建模、消融和实验设置。作者预印本；正式题录另核，不宣称逐行核对正式全文；不声称整篇已读。
- **引用位置**：非配对对应关系的可识别性依赖约束。
- **版本核验**：正式NeurIPS2017元数据；全文读取作者预印本。出版页未列DOI/页码，不猜测。

### S22　Domain-Adversarial Training of Neural Networks

- **题录**：Yaroslav Ganin; Evgeniya Ustinova; Hana Ajakan; Pascal Germain; Hugo Larochelle; François Laviolette; Mario Marchand; Victor Lempitsky. Domain-Adversarial Training of Neural Networks. Journal of Machine Learning Research 17(59):1–35, 2016.
- **来源与DOI**：[原始记录](https://www.jmlr.org/papers/v17/15-239.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：任务分类与GRL域对抗表示。联合源域任务监督和域判别学习域不变表示，并评估分类/检索。
- **适用边界**：主要输出是任务表示；不能把潜空间域混淆直接视作完整触觉重建。
- **实际阅读**：PDF第1,10,24页相关段落；摘要、浅层更新算法及图像域适应实验表。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：域适应与触觉输出重建的区别。
- **版本核验**：JMLR HTML citation_author误列Mario March，PDF署名为Mario Marchand，采用PDF署名；重音字符已核对。

### S23　A Kernel Two-Sample Test

- **题录**：Arthur Gretton; Karsten M. Borgwardt; Malte J. Rasch; Bernhard Schölkopf; Alexander Smola. A Kernel Two-Sample Test. Journal of Machine Learning Research 13(25):723–773, 2012.
- **来源与DOI**：[原始记录](https://www.jmlr.org/papers/v13/gretton12a.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：RKHS最大均值差异与两样本检验。将分布比较写为核均值嵌入差异，研究检验与核选择。
- **适用边界**：有限样本、核/表示和独立性影响估计；不检查输入与输出接触语义。
- **实际阅读**：PDF第1,5,25页相关段落；摘要、均值嵌入/定理条件及数据库/生物/神经实验设计开头。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：分布指标须写清实现和统计条件。

### S24　Reliable Fidelity and Diversity Metrics for Generative Models

- **题录**：Muhammad Ferjad Naeem; Seong Joon Oh; Youngjung Uh; Yunjey Choi; Jaejun Yoo. Reliable Fidelity and Diversity Metrics for Generative Models. International Conference on Machine Learning, PMLR 119:7176–7185, 2020.
- **来源与DOI**：[原始记录](https://proceedings.mlr.press/v119/naeem20a.html)；DOI未核实/出版页未列，暂不填写。
- **路线与原文支持**：生成保真度、多样性及密度覆盖。用density/coverage补充precision/recall，并检验离群点和模式丢失。
- **适用边界**：依赖特征及邻域；原图像特征不能无验证充当触觉物理距离。
- **实际阅读**：PDF第1,4,7页相关段落；摘要、指标定义、Gaussian和图像实验选段。正式开放版本或已审核1.1存档；以local_pdf首页与来源记录为准；不声称整篇已读。
- **引用位置**：不能仅用单一分数概括生成质量。

### S25　Sim-to-Real for Robotic Tactile Sensing via Physics-Based Simulation and Learned Latent Projections

- **题录**：Yashraj Narang; Balakumar Sundaralingam; Miles Macklin; Arsalan Mousavian; Dieter Fox. Sim-to-Real for Robotic Tactile Sensing via Physics-Based Simulation and Learned Latent Projections. IEEE International Conference on Robotics and Automation (ICRA):6444–6451, 2021.
- **来源与DOI**：[原始记录](https://research.nvidia.com/publication/2021-06_sim-real-robotic-tactile-sensing-physics-based-simulation-and-learned-latent)；[DOI](https://doi.org/10.1109/icra48506.2021.9561969)。
- **路线与原文支持**：FEM触觉模型与独立VAE监督潜投影。分别学习BioTac形变与电信号表示，冻结后用有限配对数据训练潜空间投影。
- **适用边界**：依赖监督对应和特定传感器；未见接触模式也可能重建失败，不是纯无配对成功。
- **实际阅读**：PDF第1,3,4,5,6页相关段落；摘要、两VAE结构、监督投影、FEM及信号/接触重建实验。作者预印本；正式题录另核，不宣称逐行核对正式全文；不声称整篇已读。
- **引用位置**：与本课题最近的方法先例，必须正面比较。
