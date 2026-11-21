# 第一章1.1研究背景与意义：材料核验报告

任务编号：TH-CH01-1.1-V1。日期：2026-11-19。状态：阶段2材料核验完成，待用户与ChatGPT审核；本报告不是任务书修订，也不是正式正文。

## 1. 核验结论及使用范围

本节可以按“具身操作需求→灵巧手触觉→仿真与可执行示教→多模态数据质量→跨域差异→本研究意义”的主线准备写作。建议保留六个自然段，不分成六个正文小标题。通用背景有文献支撑；项目实现须以代码和指定产物限定，不把研究目标写成已完成成效。

已取得11篇候选文献全文并阅读与本节有关的正文位置，**并非11篇逐字全文通读**。来源为作者预印本、机构接受稿或会议正式PDF；逐篇版本及页码见第4节，PDF页码不冒称正式出版页码。文献题录与可访问状态见[机器可读清单](ch01_1_1_sources/literature_catalog.json)。未修改reference.bib。

内部核验读取实际代码、五条开发Episode的NPZ、两个历史LeRobot导出的全部Parquet数值记录及相关元数据。未加载模型、运行仿真或读取触觉研究Test/Holdout数组。没有对服务器全量数据库、视频PTS和所有源lineage重新验收。[内部证据及Hash](ch01_1_1_sources/internal_evidence.json)限定每项核查范围。

## 2. 建议的六段论证结构（仅提纲）

|段落|论证职责与衔接|建议字数|引用位置|
|---|---|---:|---|
|1|从接触丰富、多指协调的操作需求，指出感知—决策—控制闭环；不展开宏观产业口号|250–300|操作特征句后W01；不据此宣称触觉唯一性|
|2|视觉有场景与几何优势，但接触后遮挡及局部状态需要补充观测；触觉与视触觉实例|350–450|遮挡/触觉信息后W03；闭环重抓取实例后W04|
|3|真实试验成本与可重复条件需求引出仿真；候选抓取还需可达轨迹和任务验证|350–450|仿真动机W01/W05；可达性区别W02；项目实现接I01/I03/I04|
|4|多模态示教不仅是存储字段，还涉及来源、观测—动作关联、时间戳与质量|300–400|示教质量W06，数据覆盖W07，标定/同步机制W08；项目能力仅按I05/I06|
|5|仿真并非真实传感输出；结构、接触响应、处理及非配对监督限制引出跨域重建|350–450|仿真差距W05；触觉专属模型W09/W10；非配对歧义可选W11；本项目缺配对用I07|
|6|落到平台、分级示教、数据质量及统计伪配对重建探索四项意义；工程和方法价值分开|250–350|以内部证据归纳，不堆外部引用；具体方法、结果留到1.4及后章|

合计约1850–2400字，处于任务书1800–2500字范围。篇幅按论证需要调整，不求等长。1.1无需写BA、FR、SWD等成绩表，也不提前评述全部算法；“为何值得研究”与“本文已经解决什么”分开。上述为任务书要求的材料组织建议，不另行制定或变更写作任务。

## 3. 论点—外部文献—内部证据对应表

|论点|外部依据|内部依据|允许的限定表述|本节不能声称|
|---|---|---|---|---|
|复杂操作需处理多指协调、接触变化和反馈|W01、W02|I01、I03|多指操作将感知及可执行控制联系起来|所有具身智能都等于灵巧手操作；视觉无法完成任何灵巧操作|
|触觉补充接触后局部状态信息|W03、W04|I02；本项目未实测滑移算法|触觉可为接触、滑移和精细调整提供信息|本项目已经完成滑移检测/闭环触觉策略；触觉是唯一途径|
|仿真支持受控重复试验，真实试验有成本与安全约束|W01、W05、W08的采集讨论|I01、I05|仿真降低对反复实机试验的依赖，提供可配置条件|仿真天然准确、完全安全或直接替代实机；本项目量化降本若干倍|
|候选抓取与机器人可执行性不同|W02|I03、I04|需结合机器人可达性与实际任务轨迹验证|高抓取评分即一定可达/成功；本流程优于全部算法|
|示教数据的来源、观测、历史与质量影响学习|W06、W07|I05、I06|记录多模态还需核查数据角色与关联合同|字段齐全就保证训练收益；本项目验证了所有模态同步|
|异频多源采集需要时间与坐标管理|W08|I05、I06|时间戳与源记录关联是质量控制内容|所有本项目视觉、动作、状态、实机触觉同步且物理配对|
|触觉仿真还受传感器读出及校准影响|W03、W09、W10|I02、I07|两域processed观测不能只凭同维度视为同物理量|光学模型直接适配本712D阵列；真实驱动与标定已认证|
|无配对监督下，分布逼近与输入保持需同时关注|W11（跨领域概念）|I07、E-P49/P50/P51仅作边界|跨域重建需处理对应关系及结构保持|GMM/双VAE/Translator已成功消除域差异；FR提高证明真实分布逼近|
|工程及方法意义|上述背景，避免引文代替本人成果|I01至I07及三方向档案|集成平台与示教链；组织和审计数据；探索统计先验辅助重建|全部工作均属原创算法；已获迁移后机器人性能验证|

## 4. 外部文献：题录、正文位置、适用性

下列W编号仅供材料核验，不是最终BibTeX键或论文引用顺序。DOI有核验依据才列；PMLR页面未列DOI的两项不编填，也不据此断言绝对不存在DOI。全文下载成功不等于每个出版社页面都可访问。

### W01 Learning dexterous in-hand manipulation

**题录**：OpenAI: Marcin Andrychowicz; Bowen Baker; Maciek Chociej; Rafal Józefowicz; Bob McGrew; Jakub Pachocki; Arthur Petron; Matthias Plappert; Glenn Powell; Alex Ray; Jonas Schneider; Szymon Sidor; Josh Tobin; Peter Welinder; Lilian Weng; Wojciech Zaremba. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1): 3–20, 2020.

**原文入口**：[来源页](https://arxiv.org/abs/1808.00177)；[本地全文](ch01_1_1_sources/L02_dexterity.pdf)。核验DOI：[10.1177/0278364919887447](https://doi.org/10.1177/0278364919887447)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页2,3的相关正文。任务与系统概述、硬件和仿真迁移条件。正文展示多指协调与动态接触，并说明真实试验慢且昂贵。

**能支持及边界**：支持复杂灵巧操作涉及感知—控制闭环，以及仿真用于训练与受控试验的动机。该实验明确未使用手上触觉传感器，不能用它证明触觉必不可少；不是本项目实机验证。

**引用作用**：第1段操作背景、第3段仿真动机；不引用其成功率或计算规模。

### W02 Data-Driven Grasp Synthesis—A Survey

**题录**：Jeannette Bohg; Antonio Morales; Tamim Asfour; Danica Kragic. Data-Driven Grasp Synthesis—A Survey. IEEE Transactions on Robotics, 30(2): 289–309, 2014.

**原文入口**：[来源页](https://arxiv.org/abs/1309.2660)；[本地全文](ch01_1_1_sources/L12_grasp_survey.pdf)。核验DOI：[10.1109/TRO.2013.2289018](https://doi.org/10.1109/TRO.2013.2289018)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页3,4的相关正文。I-C至I-E及II开头：候选生成、数据来源、任务/手模型因素及可达性过滤。

**能支持及边界**：支持候选抓取质量与可执行抓取的区别；正文对已知物体明确讨论位姿估计后的可达性筛选。不能把旧综述对当时研究的观察写成2026年全领域现状。

**引用作用**：第3段，支撑候选生成与机器人约束需要连接；不展开综述分类。

### W03 Robotic tactile perception of object properties: A review

**题录**：Shan Luo; Joao Bimbo; Ravinder Dahiya; Hongbin Liu. Robotic tactile perception of object properties: A review. Mechatronics, 48: 54–67, 2017.

**原文入口**：[来源页](https://kclpure.kcl.ac.uk/portal/en/publications/robotic-tactile-perception-of-object-properties-a-review/)；[本地全文](ch01_1_1_sources/L03_tactile_review.pdf)。核验DOI：[10.1016/j.mechatronics.2017.11.002](https://doi.org/10.1016/j.mechatronics.2017.11.002)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页2,3,9的相关正文。I引言、II感知模态与解释层次、V接触位姿估计。PDF含机构封面，PDF第2页才是手稿第1页。

**能支持及边界**：支持触觉可提供接触、滑移及物体属性线索；V节讨论机械臂接近物体时的视觉遮挡。不同触觉硬件与任务影响解释方式；不将综述对所有传感器能力的汇总套在项目712D上。

**引用作用**：第2段触觉价值；第5段传感与处理差异的背景。

### W04 More Than a Feeling: Learning to Grasp and Regrasp Using Vision and Touch

**题录**：Roberto Calandra; Andrew Owens; Dinesh Jayaraman; Justin Lin; Wenzhen Yuan; Jitendra Malik; Edward H. Adelson; Sergey Levine. More Than a Feeling: Learning to Grasp and Regrasp Using Vision and Touch. IEEE Robotics and Automation Letters, 3(4): 3300–3307, 2018.

**原文入口**：[来源页](https://arxiv.org/abs/1805.11085)；[本地全文](ch01_1_1_sources/L04_visuotactile.pdf)。核验DOI：[10.1109/LRA.2018.2852779](https://doi.org/10.1109/LRA.2018.2852779)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页3,4,5的相关正文。III硬件、IV动作条件视触觉预测、V数据收集、VI-A观测消融。

**能支持及边界**：支持视觉、触觉和动作共同用于接触后重抓取；同文消融只证明其任务条件下的互补价值。还观察到GelSight胶体更换带来的输入变化，不能借此诊断本项目Session漂移的确切原因。

**引用作用**：第2段，作为具体视触觉闭环实例；不详细介绍网络或搬运外部性能数字。

### W05 Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World

**题录**：Josh Tobin; Rachel Fong; Alex Ray; Jonas Schneider; Wojciech Zaremba; Pieter Abbeel. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS): 23–30, 2017.

**原文入口**：[来源页](https://arxiv.org/abs/1703.06907)；[本地全文](ch01_1_1_sources/L06_randomization.pdf)。核验DOI：[10.1109/IROS.2017.8202133](https://doi.org/10.1109/IROS.2017.8202133)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页2,3的相关正文。II-B/C域适应与现实差距、III-A随机化设计。

**能支持及边界**：支持模拟与实测观测有现实差距，以及仿真数据利用依赖训练/实际条件差异的处理。主要实验是视觉定位，不是触觉；不据其背景段转引其他未读论文，也不称本项目采用了其随机化方法。

**引用作用**：第3段仿真动机和第5段Sim-to-Real问题。

### W06 What Matters in Learning from Offline Human Demonstrations for Robot Manipulation

**题录**：Ajay Mandlekar; Danfei Xu; Josiah Wong; Soroush Nasiriany; Chen Wang; Rohun Kulkarni; Li Fei-Fei; Silvio Savarese; Yuke Zhu; Roberto Martín-Martín. What Matters in Learning from Offline Human Demonstrations for Robot Manipulation. Proceedings of the 5th Conference on Robot Learning, PMLR 164: 1678–1690, 2022.

**原文入口**：[来源页](https://proceedings.mlr.press/v164/mandlekar22a.html)；[本地全文](ch01_1_1_sources/L07_robomimic.pdf)。所核对PMLR题录未列DOI，留空。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页3,5,7的相关正文。挑战C2/C3/C4，4.1观测历史和示教质量、4.3观测、4.5选择、4.6规模。

**能支持及边界**：支持示教质量、观测历史和观测选择影响学习，训练损失并非任务成功率的替代。不是触觉同步误差的受控实验，不能用其结果给本项目时间对齐收益定量背书。

**引用作用**：第4段数据质量，必要时第3段示教动机。

### W07 BridgeData V2: A Dataset for Robot Learning at Scale

**题录**：Homer Rich Walke; Kevin Black; Tony Z. Zhao; Quan Vuong; Chongyi Zheng; Philippe Hansen-Estruch; Andre Wang He; Vivek Myers; Moo Jin Kim; Max Du; Abraham Lee; Kuan Fang; Chelsea Finn; Sergey Levine. BridgeData V2: A Dataset for Robot Learning at Scale. Proceedings of the 7th Conference on Robot Learning, PMLR 229: 1723–1736, 2023.

**原文入口**：[来源页](https://proceedings.mlr.press/v229/walke23a.html)；[本地全文](ch01_1_1_sources/L08_bridgedata.pdf)。所核对PMLR题录未列DOI，留空。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页3,4的相关正文。3.1系统、3.2采集、3.3技能/任务和数据组成。题录作者按PMLR正式元数据保留。

**能支持及边界**：支持多任务、多环境与示教/自动采集来源的组织需求。该工作不提供本项目的712D触觉，不能证明所有机器人数据集均有触觉或所有失败数据都适合任意算法。

**引用作用**：第4段，作为多源与覆盖设计的实例，不作本项目规模对标。

### W08 RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot

**题录**：Hao-Shu Fang; Hongjie Fang; Zhenyu Tang; Jirong Liu; Chenxi Wang; Junbo Wang; Haoyi Zhu; Cewu Lu. RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot. 2024 IEEE International Conference on Robotics and Automation (ICRA): 653–660, 2024.

**原文入口**：[来源页](https://rh20t.github.io/)；[本地全文](ch01_1_1_sources/L13_rh20t.pdf)。核验DOI：[10.1109/ICRA57147.2024.10611615](https://doi.org/10.1109/ICRA57147.2024.10611615)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页3,4,5的相关正文。III-A多模态/层级、III-B采集处理及图6同步说明；所读全文为arXiv:2307.00595v2（2023），正式题录为ICRA2024。

**能支持及边界**：支持多传感器数据需要时间戳、坐标标定与处理；指尖触觉仅在部分配置提供。不能把腕部六维力矩等同于全量指尖阵列，也不把作者同步声明当成本项目同步证据。

**引用作用**：第4段数据关联与时间对齐；不承诺本项目达到其采集条件。

### W09 TACTO: A Fast, Flexible, and Open-Source Simulator for High-Resolution Vision-Based Tactile Sensors

**题录**：Shaoxiong Wang; Mike Lambeta; Po-Wei Chou; Roberto Calandra. TACTO: A Fast, Flexible, and Open-Source Simulator for High-Resolution Vision-Based Tactile Sensors. IEEE Robotics and Automation Letters, 7(2): 3930–3937, 2022.

**原文入口**：[来源页](https://arxiv.org/abs/2012.08456)；[本地全文](ch01_1_1_sources/L09_tacto.pdf)。核验DOI：[10.1109/LRA.2022.3146945](https://doi.org/10.1109/LRA.2022.3146945)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页3,4,7的相关正文。III-B架构权衡、III-D配置及力—形变近似、V Sim2Real位姿估计。

**能支持及边界**：支持触觉仿真除接触求解外还涉及传感器读出与配置近似；文中Sim2Real实验存在差距。对象是视觉触觉，不能证明本项目原生touch等价于真实阵列，亦不能推广为普适准确形变模型。

**引用作用**：第5段，以传感器专属模型说明跨域差异，细节留给1.2及第二章。

### W10 Taxim: An Example-Based Simulation Model for GelSight Tactile Sensors

**题录**：Zilin Si; Wenzhen Yuan. Taxim: An Example-Based Simulation Model for GelSight Tactile Sensors. IEEE Robotics and Automation Letters, 7(2): 2361–2368, 2022.

**原文入口**：[来源页](https://arxiv.org/abs/2109.04027)；[本地全文](ch01_1_1_sources/L10_taxim.pdf)。核验DOI：[10.1109/LRA.2022.3142412](https://doi.org/10.1109/LRA.2022.3142412)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页2,3的相关正文。III-A/B模型与逐传感器校准，III-B球形压头及形状—光响应关系。

**能支持及边界**：支持传感器结构、读出机制和校准影响仿真响应。模型为GelSight光学响应与标记位移，不是本项目压阻/标量数组模型；球形压头相似不代表本项目实现了Taxim或完成了实机标定。

**引用作用**：第5段，与TACTO择一或并列引用，避免重复堆文献。

### W11 Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks

**题录**：Jun-Yan Zhu; Taesung Park; Phillip Isola; Alexei A. Efros. Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. 2017 IEEE International Conference on Computer Vision (ICCV): 2242–2251, 2017.

**原文入口**：[来源页](https://openaccess.thecvf.com/content_ICCV_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)；[本地全文](ch01_1_1_sources/L11_unpaired.pdf)。核验DOI：[10.1109/ICCV.2017.244](https://doi.org/10.1109/ICCV.2017.244)。

**取得与阅读**：已取得PDF并提取文字；实际阅读PDF页2,3的相关正文。引言图2之后的非配对映射歧义与第3节定义。所读为CVF正式会议PDF。

**能支持及边界**：支持缺少逐样本对应时，单纯分布匹配不足以确定有意义映射的概念背景。这是图像领域的论证，不是触觉实验证明，不说明本文使用CycleGAN或已解决重建。

**引用作用**：第5段末可选引用一处，只有在明确跨领域概念边界时使用。

## 5. 内部材料核验及证据等级

L0=定位；L1=核对报告/配置/历史清单；L2=核对指定实现或实际产物。L2只对所列范围有效，既不表示重新执行实验，也不表示全量最终验收。下列路径默认相对项目根目录，详细Hash见internal_evidence.json。

|ID|实际来源与核验位置|级别与事实|允许用在1.1的内容|边界|
|---|---|---|---|---|
|I01 平台组合|source/robots/arms/rm75b.py:10–27；source/robots/hands/dex_hand.py:16–34；source/envs/rl_env.py:41–42|L2代码：默认RM75B/Dex Hand，7臂位置执行器+6手驱动，注册触觉factory|本研究以该机械臂—灵巧手组合建立仿真平台|不是13个独立手指关节；XML精度/硬件一致性未本轮认证|
|I02 触觉来源|source/sensors/tactile/dex_hand.py:360–427；source/sensors/tactile/signal_processing.py:162–203|L2代码：site原生mjSENS_TOUCH→sensordata→死区/饱和/幂/低通/空间处理/噪声/可选归一化；具体开关取决配置|平台集成712维触觉观测与处理|不是touch_grid插件；不宣称Pa/N标定压力图或已认证真实Taxel物理映射|
|I03 候选与可达|source/grasping/graspqp_adapter.py:64起；source/grasping/dexevolve.py；source/grasping/executor.py:161–238；source/rl/grasp_edit/templates.py:370–385；tools/grasping/batch_grasp_edit.py:702起|L2指定函数：优化候选姿态/驱动，实际RM75B IK对pregrasp/grasp计算误差，按阈值过滤及执行模板|结合候选生成、机器人可达性与执行验证的分级流程|IK预检不等于整条轨迹碰撞自由；GraspQP/DexEvolve来源须引用，1.1不展开算法优越性|
|I04 两种回放|source/grasp_pipeline/replay.py:23–69；source/verification/strict_replay.py:32–44；outputs/five_task_p32_b_2026-11-03/lift_sugar_strict.json|L2代码；L1历史结果：普通Replay尾部任务成功与FINAL的抬升、速度、穿透、接触等条件不同|示教需要任务与轨迹质量验证|不称五任务均已通过同一FINAL Strict；不把历史JSON说成本轮重新Strict通过|
|I05 五任务指定记录|outputs/five_task_p32_b_2026-11-03/下五个case/source_episode/episode.npz，见下表；summary.json及各历史manifest|L2实际数组shape/finite/时间，L1任务成功记录|五类任务有指定仿真示教案例及传感记录|不是全部对象、全部初态、独立Root或所有视频已复验；不据此给实机成功率|
|I06 LeRobot及关联|outputs/dp_2026-10-11/lerobot_5/；outputs/dp_p1_2026-10-11/lerobot_formal30/；source/data/production_recorder.py:56–96、146–205；source/grasping/executor.py::_Recorder|L2实际Parquet选定数值字段；L2当前代码的pre-action链与旧记录区别|多模态示教存在，关联/因果/版本需要质量审计|同一Episode索引、单调时间戳不是全部跨设备同步；当前代码合同不回填旧产物|
|I07 跨域关联|outputs/tactile_p34_feasibility_audit/common_support_matrix.csv；outputs/tactile_p41_real_informed_gmm_forward_2026-11-15/source_audit.md、real_sim_taxel_mapping.csv；outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/paired_manifest.csv及code/matching.py|L1历史元数据与源码缺口；L2指定匹配代码：有Real来源索引/GMM统计/探针配置和SimEpisode字段，非同步物理观测|当前数据缺乏经过认证的Real–Sim同步物理配对；已有统计伪配对探索|0个“已确认物理配对”不代表未来绝对不能配对，也不证明潜空间天然共同坐标|

### 5.1 五任务数组实读结果（开发案例，不是新增实验）

统一目录：`outputs/five_task_p32_b_2026-11-03/`。每行文件为`case/source_episode/episode.npz`。

|case|sensor_tactile|state|position_action|数值与时间检查|
|---|---|---|---|---|
|lift_sugar_rgb_00|447×712|447×39|447×13|触觉float32、finite、非全零；timestamp间隔0.05s|
|pick_tomato_rgb_00|638×712|638×39|638×13|同上|
|stack_ring_rgb_01|376×712|376×39|376×13|同上|
|push_tomato_rgb_00|605×712|605×39|605×13|同上|
|nut_retain_lower_rgb_00|599×712|599×39|599×13|同上|

这些文件来自历史实际执行；本轮读取不能重新证明其独立性、任务成功或视频同步。PickPlace目标site与第二动态对象不同；Stack/Nut对象身份以各manifest为准。1.1建议不列上述帧数或非零率，留作后章事实依据。

### 5.2 LeRobot实际规模及未解决关联

|目录|metadata与实际读取一致的结果|本轮额外检查|未检查|
|---|---|---|---|
|outputs/dp_2026-10-11/lerobot_5/|5个episode_index、1550行|全部所列Parquet的state39/action13/tactile712均finite；各Episode timestamp严格递增|源Root独立性、内容重复、RGB/视频PTS与动作的时序对应、全源manifest|
|outputs/dp_p1_2026-10-11/lerobot_formal30/|30个episode_index、14310行|同上|同上；目录名formal30不是成功/Strict证明|

这两项是不同历史导出，不能去重前相加为35个独立试验。单调timestamp只说明选定数值记录顺序，不等同统一硬件时钟。当前OnlineEnvironment在动作前捕获状态和触觉，动作后保存实际ctrl；旧Recorder记录env.step之后的状态/动作，监督配对需要版本化处理。不得把修复后的pre-action合同说成所有历史样本的既成事实。

### 5.3 Real–Sim关联的可核查结论

- P34共同条件表将同712D标为COMMON_NUMERIC_CONTRACT_ONLY，同名阶段标为COMMON_STAGE_NAME_ONLY；传感器/物体/控制对应NOT_ESTABLISHED。confirmed_joint_physical_pairs登记0，含义是现有证据未认证，不是物理不可能。
- P41映射表有712行，但每维物理对应标为UNKNOWN；可恢复数组顺序不等于恢复实机安装方向、面积或标定。
- Real同帧raw/processed配对只能说明Real处理链内部观测关系，不是Real与Sim的同步接触。
- C3 paired_manifest以real_source_window_id、sphere_configuration、sim_root_fingerprint、actual_sensor_episode等追踪构造过程；matching.py依据固定索引统计衡量响应。这是统计监督来源，不是相同真实接触的实测ground truth。

## 6. 四项研究意义的可写范围

|方向|工程意义|方法研究意义及限定|
|---|---|---|
|平台与感知|将机器人、任务、控制和阵列观测集成于可检索的仿真链路|研究曲面阵列近似及感知合同；不声称已标定真实软皮肤或完成实机部署|
|分级示教|连接候选、机械臂约束、实际任务执行和回放|研究几何候选与可执行轨迹之间的筛选；不把使用开源算法本身称原创|
|数据质量|记录来源、关联、存储转换与审计，便于后续复核|研究哪些质量条件支持训练数据使用；不声称本论文已验证VLA性能增益|
|跨域重建|在已有数据与模型产物上组织完整触觉转换探索|研究统计伪配对与输入保持/分布逼近之间的关系；不宣称GMM、独立双VAE或Translator已经解决迁移|

1.1可概括为研究动机与价值，不需要重复1.4的方法贡献清单。P49/P50/P51仅约束不能声称的成果，背景段无需按编号逐项汇报。保留F0特定距离改善、P49接受率与质量分离、P50退化和P51仅针对D50的NO-GO；任何跨实验数值比较仍需原协议可比性检查。

## 7. 允许与禁止表述清单

允许的表达单元（不是已批准正文）：

- “多指接触操作需要将感知与可执行控制联系起来。”
- “视觉和触觉可提供具有互补性的操作信息，作用依任务与传感器条件而异。”
- “抓取候选还需结合机械臂可达性和任务执行进行验证。”
- “本研究围绕仿真平台、示教生成、数据质量与跨域重建开展工作。”
- “当前既有Real和Sim记录缺乏已认证的同步物理配对。”
- “仿真响应与真实processed触觉之间的对应关系仍需研究。”

禁止或须补证后使用：

- “触觉是实现具身智能的唯一途径”“视觉不能进行精细操作”。
- “712维均为真实压力Pa”“已恢复真实Taxel几何”“MuJoCo刚体柔顺就是硅胶变形”。
- “已构建完全同步的真实视觉—动作—状态—触觉五任务数据集”。
- “五任务百物体统一Strict全部通过”“30条LeRobot记录就是30条独立成功专家试验”。
- “统计伪配对就是同步接触真值”“独立Dual-VAE天然对齐潜空间”。
- “FR提高意味着接近真实分布”“本模型已完整达到0.90”“本论文改善实机任务成功率”。
- “所有平台实现和开源算法组合都是原创算法创新”。

## 8. 事实冲突、材料缺口与处理建议

|事项|已发现情况|对1.1的处理|
|---|---|---|
|OpenAI论文年份|在线发表/版权2019，正式IJRR卷39(1)为2020；arXiv2018|正式题录用2020，读过的预印本版本明确标注，不把它叫2020新数据|
|robomimic年份|CoRL会议届次对应2021，PMLR卷164正式题录2022|用出版题录2022，避免同文拆为两篇|
|BridgeData作者顺序|PMLR页面元数据与其PDF首页中间作者顺序、部分姓名展开不同|报告保存两份来源；题录暂按PMLR元数据，录入bib前确定规范化版本，不自行拼接姓名或猜排序|
|RH20T版本|所读是arXiv2023v2，正式ICRA2024 DOI已核验|只用所读III节数据处理背景，披露版本；最终正式引用前可核对终稿，不能称本轮通读终稿|
|Real总Session数|旧档案有77有效组与P34元数据79组等不同口径|1.1不引用总数；后章需按可用性和角色分别核对，不能取大数合并|
|真实映射/标定|原驱动、Calibration和安装方向不完整|足以说明研究限制，不虚构补齐；不能在背景承诺两域同物理量|
|时间关联|五条NPZ及两个导出数值时间检查通过；视频/源lineage尚未全量验收|只写质量控制需求，不宣布全量同步|
|服务器规模|本机不可读取最新数据库与盘|1.1不写最新小时数或把旧快照称实时；不影响非定量背景撰写|
|安全与成本|代表性论文有实机试验成本和接触碰撞风险讨论|作一般动机，项目无量化降本/安全试验，不能承诺零风险|
|范围与题名|七章结构通过，但封面题名仍较聚焦触觉|需用户/导师后续确认正式题名与课题来源；本轮不修改封面或任务书|

访问失败的背景候选未计入11篇：Billard/Kragic的Science综述只核对到题录及页面片段，EPFL下载429，未取得可确认全文；MuJoCo2012原论文尝试的作者地址返回非PDF或404。两者不作为“已阅读全文”证据，也不靠二手摘要补充技术论点。现有W01/W02/W03已覆盖对应基础背景，无需为数量继续扩展文献。初步误命中的无关arXiv条目已排除，未纳入清单。

目前没有需要改任务书的实现冲突：任务书要求谨慎区分的配对、通道与时序边界均应保留。用户需审核六段组织、引用取舍及四项意义的用语；正式题名/课题来源不凭路径用户名推断。不要求本轮补采数据或启动新实验。

## 9. 引用、公式和图表建议

- 11篇是候选材料池，不要求1.1全部引用。建议实际引用约8–10篇，W09/W10可择一，W11作为可选的非配对概念出处；详细算法比较留1.2。
- 引用紧随其支持的论点，不在段末用一个文献覆盖整段项目事实；项目实现直接追踪后章及代码，不以外部文献替代本人成果。
- 1.1不必加入推导公式。若必须给输入输出记号，仅用统一术语，不引入未经实现的共同latent坐标或新损失。
- 1.1可不放实验图；若任务书后续指定总览图，仅表现平台—示教—数据—重建关系。未认证物理配对不能画成逐Taxel同步监督，不能放示意截图冒充实验。

## 10. 本轮交付与停止状态

主报告：本文件。支撑文件：ch01_1_1_sources/中的11篇PDF及文本/逐页索引、literature_catalog.json、access_manifest.json、元数据原件、internal_evidence.json、只读核验脚本和完整性记录。原始PDF/全文仅作本地阅读，不复制到论文正文或自动发布。

没有修改.tex、reference.bib、模板、封面、原实验报告、数据或模型；没有训练、仿真、实机操作、P52/P53、编译或Git提交。本报告不把材料核验通过视为已获写作许可。完成后停止，等待用户与ChatGPT审核，不进入阶段3或下一节。
