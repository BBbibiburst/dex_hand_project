# 实验依据档案

建立日期：2026-11-19。仅核验和整理，不改原报告、模型、数据或划分，不启动训练。本文件不是论文正文。

## 一、使用规则与覆盖范围

1. 数字必须绑定路线/日期/目录、数据版本、评价协议和平均方式。P编号并非全仓库唯一：DP模仿学习、五任务生产与触觉迁移有重复编号，不能凭P44三个字引用。
2. 已索引170份报告/档案，清单见[来源目录](docs/writing_audit/evidence_catalog.csv)，并非170份全部完成数组独立复算。L0=仅定位；L1=核对报告/配置摘要；L2=本轮核对相关实现或实际CSV。历史已经验证的结果与本轮独立复核深度分别记录。
3. 当前20个关键数值直接从CSV读取，字段选择器、原值和文件Hash见[verified_claims.csv](docs/writing_audit/verified_claims.csv)。其他历史数字标为报告记载，正式写入该节前仍须读取其表/配置。
4. 状态使用：已执行开发结果、机制诊断、负对照、失败、未执行、历史修正、UNKNOWN。未找到不等于未执行；不把NOT_EVALUATED写0分。
5. Test/Holdout只引用历史报告的角色与既有结论；本轮不加载这些数值数组。开发Validation、历史测试角色的重新命名不生成新盲测。
6. “可用于正式论文”指按范围报告真实研究证据，不等于足以宣布正式项目验收通过。

## 二、主线数据合同：E-DATA（L2）

主要来源：`outputs/tactile_p37_full_data_revalidation_2026-11-15/p37_report.md`；`outputs/tactile_p43_latent_translator_integrated_2026-11-16/P43_FINAL_REPORT.md`；`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/p43_c3_report.md`；P49/P50 data_inventory及common.py。

|数据版本|来源与规模|禁止混用的语义|
|---|---|---|
|P23|Real370窗、Sim792窗，报告中的Session/Root分组|并非全库、并非当前Natural|
|P24全保存池|Real11110窗、77有效Session、1645完整资产Trial；Sim6359窗、119注册初态组|全库量不等于单模型Train量|
|P33/P35小池|每域Train128/Val48窗；Real16/6 Trial；P35 Sim实际35/11 Root|允许Root清单38/13不等于实际贡献数；旧转换器暴露验证Trial|
|P37|Real5153/1408窗、32/7 Session；Sim1824/406窗、36/8 Root|旧来源，不是服务器持续生产的全五任务样本|
|P43 Natural及P44–P51|Real5153/780窗、32/4Session；Train755Trial、Val120Trial。Natural4049/905窗、50/10Root、五任务各10/2Root|另有历史测试角色，本轮不读；Real外层4Session有开发暴露|
|P49/P50 D内部|Real4217/936窗、24fit/8inner Session；Sim3249/800窗、40/10Root|inner没有D梯度，但已开发诊断；G可用全部NaturalTrain|

Real/Sim没有已证明同步物理配对。Paired Sim是训练统计监督，Natural Sim才是主要部署输入。P43构造数据与服务器累计小时数不可相加；重复读取索引不算新物理时长。当前服务器实时状态UNKNOWN，本轮不查询生产进程。

## 三、早期P0–P18与生产旁线的定位

|路线/编号|目标及已找到材料|数据、模型、指标核验状态|论文用途与边界|
|---|---|---|---|
|P0|DP P1文件夹中存在latency_vs_p0.json等引用，但未找到单独P0正式报告|L0，配置/最终结果UNKNOWN|只能保留线索，不能补造实验|
|IMITATION P1/P2|`outputs/dp_p1_2026-10-11/`、`dp_p2_2026-10-11/`|L0，示教/学习策略的具体冻结成绩未在本轮逐项复核|非本论文主线对比，不冒充触觉跨域实验|
|IMITATION P3|`docs/progress/2026-10-11_dp_p3_report.md`；同物体固定reset比较BC/ACT/DP|L1，39D状态/13D动作；BC/ACT未完成Lift为报告记载|可作仿真背景，不证明触觉迁移改善机器人成功率|
|IMITATION P4–P9|对应`docs/progress/*_dp_p[4-9]_report.md`及outputs同号目录|L0，目标/配置/数值须逐报告再核验|保留在索引；不按编号塞入触觉方法章|
|IMITATION P10|`outputs/dp_p10_vision_2026-10-24/`及vision_resolution_audit文档|L0，非正式触觉迁移结果|不是未执行，也不将其图像实验当本论文证据|
|IMITATION P11–P17|`docs/progress/2026-10-24_dp_p11_report.md`至p17；对应outputs目录|L0，模型/指标未逐项核对|本阶段仅定位；不引用未经复核数字|
|TACTILE P18及子阶段|`docs/progress/2026-10-25_tactile_domain_baseline.md`、raw_domain_audit.md、mapping_provenance.md；`outputs/tactile_sim2real_audit/`|L1，存在127D聚合统计诊断与raw/processed审计；不是完整712D迁移通过|适合来源/处理差异背景，不能混入后期712D主表|
|DATASET P29/P30|`outputs/five_task_storage_runtime_smoke_2026-10-27/`与`five_task_runtime_p30_2026-10-27/`，及对应apps/报告|L0，存储/编码/吞吐，不是Real-Real上界或触觉模型|不作为迁移质量对比|
|DATASET P31（E-P31）|`outputs/five_task_p31/p31_final_report.md`|L1，45attempt→21唯一Episode、443.65秒为报告记载|证明统一链路试点；不是1000小时已采完|
|DATASET P32及P33–P51|`docs/progress/2026-11-03_p32_*`、后续生产文档、five_task_p*目录|L0–L1，录制、恢复、调度等工程证据；与触觉同号实验不同|可作数据组织背景；不据此宣称实机验证/当前全量完成|

P0–P51不是从同一个模型连续迭代的51项同口径实验。未逐项核验的早期条目在正式写作前按小节需要补读，当前不做新实验。

## 四、触觉早期方法：127D到完整712D

|证据ID/阶段|目标；数据和模型|主要指标/结果及状态|来源/配置/产物|可支持／不能支持；论文使用|
|---|---|---|---|---|
|E-P19|127D聚合统计残差GAN；P19非配对统计池|失败；独立Probe仍易分，存在非法统计输出；L1|`docs/progress/2026-10-25_tactile_p19_gan_report.md`；`outputs/tactile_p19_gan_2026-10-25/`相关manifest，确切目录以报告为准|支持聚合统计代理的局限；不是完整712D实验，可作背景|
|E-P20|冻结P19三checkpoint失效诊断|确认生成器有改变，幅值/时间仍可分；L1|`docs/progress/2026-10-25_tactile_p20_gan_diagnosis.md`及所列summary/feature_diagnostics|支持目标/选择代理不足，不证明所有GAN无效|
|E-P21|数据可辨识性及采集软件就绪；P21.0/P21.1|空间映射与原采集栈缺失，阻塞；L1|对应两份2026-10-25报告；非训练|可写数据限制，不虚构已有硬件采集能力|
|E-P22|127D约束映射及经验传输；Real184、Sim504窗|开发验证D达标但既有测试Logistic/MLP未全部达标；失败，L1|`docs/progress/2026-10-25_tactile_p22_domain_confusion.md`及冻结evaluation|验证选择与最终泛化不同；不是712D成功，不跨协议混排|
|E-P23|4×712 Patch-aware GAN；Real370/Sim792窗|六个有界验证候选Logistic仍BA/AUC=1；失败，L1|`docs/progress/2026-10-26_tactile_p23_712d_gan.md`、对应outputs config/评估|支持当前目标下完整响应仍可分；历史holdout已公开，不称新盲测|
|E-P24|扩大完整712D池、GAN与组量学习曲线|扩容候选验证仍可分；报告记载测试D=0，L1|`docs/progress/2026-10-26_tactile_p24_data_expansion_gan.md`；全库数见E-DATA|支持该方案扩容未消除差异，不证明任何方法数据量无用|
|E-P25|P23/P24梯度、过拟合与特征族诊断|梯度实际有效，模型可优化代理却恶化Probe，L1|2026-10-27_tactile_p25_gan_diagnosis.md；对应梯度/消融JSON|排除未训练这一解释，不自动归因于模型容量|
|E-P26|固定1472928参数模型、替换目标|没有稳定超过epoch-zero；最差D仍很低；失败，L1|2026-10-27_tactile_p26_objective_redesign.md；对应比较CSV|支持目标设计不足；不以一个最好seed通过替代全Probe|
|E-P27|显式支持Mask及条件幅值|总体零比例接近Real，Mask-only BA/AUC仍1；最佳最差D约0.002，失败，L1|2026-10-27_tactile_p27_support_modeling.md；support/Probe结果|支持联合支持未解决，不把零比例匹配当完整迁移|
|E-P28|P24 Train内4折Real donor→held Session诊断|47Real Session；Real-balanced Logistic D约0.590–0.595，MLP0.550–0.565；非参数传输按规则未执行，L1|2026-10-27_tactile_p28_transport_feasibility.md|提示Real Session漂移；不是冻结C0的理论上界，不能证明迁移不可达|

### 无阶段号条件诊断，不强行改名为P29/P30

- `docs/progress/2026-11-08_tactile_conditional_real_upper_bound.md`：13组条件/阶段、4折Real-only诊断，最好最差D=0.7375，仍未联合通过。L1，适合共同条件/Session差异说明，不是Sim转换成绩。
- `docs/progress/2026-11-08_tactile_calibration_transformer.md`及11月10–11日后续档案：同校准Session内Trial隔离Transformer开发。不同版本训练/预训练与验证暴露要分别核对，不能引用旧讲稿数值充当当前最终结果。
- TACTILE/P29–P32没有在本轮找到与DATASET编号一一对应的独立正式命名报告。保留无编号来源和UNKNOWN关系，不虚构阶段映射。

## 五、条件模型、数据与物理响应审计

|证据ID|目标、数据/模型|结果/性质|来源与论文边界|
|---|---|---|---|
|E-P33|小开发集校准、DANN及独立Probe；每域128/48窗|有限候选未通过；物理输出与潜特征分开；L1|`outputs/tactile_p33_final_2026-11-14/independent_acceptance_report.md`、all_probe_results.csv；开发对照可用，不冒充跨Session验收|
|E-P34|共同条件/未暴露数据审计；无训练|没有足够证据认证共同物理条件及全新独立验收组，L1|`outputs/tactile_p34_feasibility_audit/final_recommendation.md`及common_support/holdout_inventory；支持边界，不说现实中绝无共同条件|
|E-P35|冻结候选下Logistic/kNN/MLP/GRU/Transformer|所有候选未过；旧Transformer用过验证Trial；L1|`outputs/tactile_p35_deep_probe_2026-11-15/p35_report.md`；成绩只能作历史开发诊断|
|E-P35.2|规模、实际贡献Root和新增15条仿真采集|实际14成功1失败、13初态组；主要学习曲线未执行，L1|同名目录p35_2_report.md/condition_compatibility/new_collection；不能以15Episode称15独立Root或将其强混Real匹配训练|
|E-P36|数据/训练全链路审计，不是原提议Raw桥接训练|说明小样本、分组及不同模型使用量；L1|`outputs/tactile_p36_full_audit_2026-11-15/p36_full_audit_report.md`；明确本轮审计与早期同号任务提案不同|
|E-P37|完整开发池条件Transformer单次1600步；参数1014193|全部训练窗口参与，未通过；L1|同名目录p37_report.md/training_history/config及Probe表；数据见E-DATA。Val用于选模型，不称盲测|
|E-P38|冻结P37校准前后消融，重新训练Probe|Mask逐值相同；校准后部分Probe改善，整体未通过，L1|同名目录p38_report.md/Probe表；不把尾部放大直接解释为总体可分性放大|
|E-P39|Mask/Amplitude/Full分解；不训新转换器|最差D分别0.0272/0.1494/0.0573；Amplitude有效4×5而非Patch位置特征；L2表示定义|同名目录p39_report.md及代码；各表示分数不能相减当信息分解|
|E-P40-C|共享特征Encoder/GRL、阶段/结构代理|最差独立D0.2427，仍未过；C空间不是完整712D输出，L1|`outputs/tactile_p40_shared_features_2026-11-15/p40_report.md`；只能支持潜特征探索|
|E-P40|实际MuJoCo传感语义，只读模型/源码|当前712D来自native touch及后处理，不是touch_grid插件或Flex，L2|`outputs/tactile_p40_sensor_semantics_2026-11-15/sensor_semantics_report.md`及dimension_map；物理语义要限定当前模型，不能倒推旧数据单位|
|E-P41|原驱动/mapping/Calibration前置审计|未通过前置条件，未拟合GMM；L1|同名目录p41_report.md/execution_status；不得把后续独立代理GMM成果回写为本轮完成|
|E-GMM-VAE|后续授权固定索引代理GMM、Dual-VAE与幅值修正|已有GMM、C/D及amplitude_repair产物；本轮仅定位，不引用未经逐表核验的成绩|`outputs/tactile_gmm_dual_vae_2026-11-16/README.md`、STATE.md、registration/vae_config/probe_metrics；L0–L1，可作方法谱系，非认证物理坐标|
|E-P42|Dual-VAE训练充分性、Mask翻转和学习曲线设计|独立Mask采样导致过快翻转的机制证据；学习曲线尚未执行，L1|同名目录p42_report.md及转移/训练CSV；不据训练时间认定欠拟合或数据充足|
|E-P43-MASK|分层时空随机Mask/索引GRU|修正后的汇总记载Seed4301实际45轮3645步，最差D0.02051；其余Seed未运行，L1|`outputs/tactile_p43_autoregressive_mask_2026-11-16/p43_report.md`及hierarchical_v2；早期“未训练”被后续日志更正，不能引用旧状态|

## 六、GMM物理伪配对及潜变量路线

|证据ID/真实命名|目标及配置/来源|结果与状态|支持／不支持的结论；正式论文用途|
|---|---|---|---|
|E-NATURAL / P43 integrated|70次新MuJoCo Natural，五任务；完整Episode及初态指纹|50/10/10 Root，去重4049/905/765窗；早期六次GMM粗控制全部未通过，模型当轮未训练，L1|Natural来源可用；不能据初始停止报告说后续C3也没训练；来源P43_FINAL_REPORT.md|
|E-C2 / P43-C2|单球、3Patch、多球及时序响应验证；独立球+weld|单球装置产生有效可控响应；逐条件结果保留，L1|仿真探针链可行，不代表压阻皮肤标定或全Real配对；p43_c2_report.md|
|E-C3 / P43-C3|16Patch×5位置×3深度×2初态字典；独立VAE+GRU Translator；C0|480条件、160实际初态指纹；有效Paired1200/240；三模型及11C0完成；原联合仅1/11，D专项11/11，L2代码/报告|统计伪配对及模型实现证据可用；不是同步物理配对/全联合通过。目录`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`，pair_manifest、c0/training_audit与metrics、config/日志|
|E-C4 / P43-C4|冻结C3，检查概率、Paired独立性及latent支持|约99.5%加载饱和，监督歧义、Natural覆盖缺口；10个非kNN实例将765Sim全判Real，L1|支持受控根因假设，不把全部原因归Translator容量；p43_c4_report.md和距离/配对表|
|E-C5 / P43-C5|固定目标Train-only单调加载标定，最多受限物理候选|B降饱和但Train19/48、Val8/12，完整门槛未过，NO-GO；0训练，L1|支持加载与字典支持缺口；不扩大Paired；p43_c5_report.md|
|E-C6 / P43-C6|固定加载空间构型修正|只有1个主要S1可执行多球修正，整体无改善；NO-GO，L1|证明支持不足，不证明所有五球构型不可能；p43_c6_report.md|
|E-C7 / P43-C7|Hertz启发响应代理、旧记录与80新条件|守恒分配局部成立，但空间分离未优于原基线，NO-GO；L1|代理局部性质可讨论，不声称硅胶真值或迁移成功；p43_c7_report.md|
|E-C8 / P43-C8|几何表面采样与B4软皮肤模型；重放821条|NO-GO，72个计划新条件未执行，0训练；L1|几何积分诊断可用，不能把821次重放当独立Root或新采集；c8_report.md|
|E-C9 / P43-C9|29多球执行/330Patch帧的积分与连通归因|多接触相邻Taxel/处理造成部分合并；高分辨率场是冻结方程重建，不是C8已存原场；L1|限这批构型，不是任意712D硬件理论上限；c9_report.md|
|E-C10 / P43-C10及R1|既有双VAE、条件latent Diffusion及v预测单Seed修复|R1按Paired Validation规则停止扩散，不追加Natural Test，L1|可作选择依据，不能称扩散模型普遍无效；两目录final_report.md|
|E-C11 / P43-C11|冻结Real Decoder后验/先验/Sim潜变量采样|总体UNKNOWN，未证明新监督可用；L1|可用作读出前提诊断，不把Real随机生成称Sim转换；REPORT.md|
|E-C12…C22|读出反馈、幅值修复、随机/Token Mask、条件投影及有限适配|已定位各report和研究路线审计；状态不能由目录名推定；L0–L1|只在具体机制小节补读后引用，不用“连续二十次实验”证明必要性|
|E-C23 / 独立C23|C13修复模型对Real自重建；780窗/4Session，仅冻结C0读出|6/11联合；报告MSE0.02632、Mask IoU0.51172；诊断，L1|不能写成Natural迁移6/11；`outputs/tactile_c23_real_reconstruction_c0_2026-11-18/REPORT.md`和metrics|
|E-C24 / 独立C24|只加相邻log幅值差损失、保留原参数范围|时序误差改善但6/11降至2/11，尾部/多样性退化；失败，L1|局部指标改善不等价读出改善；同名目录REPORT.md/对照CSV|
|E-C25 / 独立C25|独立Encoder latent的GMM-OT前提审计|共同坐标不成立，未做跨域W2/OT，NO-GO；L1|不能写成OT算过且失败；同名目录REPORT.md|
|E-C26 / 独立C26|同一冻结C13 Encoder编码两域/零样本读出|共同数值坐标成立但输出结构未过；原C0未运行，不写0/11；L1|共享函数不证明语义对齐；同名目录REPORT.md|
|E-C27 / 独立C27|GRU Natural覆盖适配，原11C0|开发D专项11/11，原联合2/11；L2封存scope|`releases/tactile_c27_D090_v1/acceptance_scope.json`、models与results；可如实给限定专项成绩，不可说完整质量/原联合目标完成|

P43-C12…C22每个已定位目录与报告均在evidence_catalog，未把未逐表复核成绩写成统一数字。后续章节若使用某一项，必须先补其数据、配置和结果字段。

## 七、当前最终主线：E-P44–E-P51

统一数据见E-DATA新Natural版本；输出均为完整4×712。各条目的“可用”均限开发验证范围。

### E-P44（L2代码，L1报告）

- 目标：在独立Dual-VAE和原Translator上构建共享条件向量场，考察K1/2/4/8迁移。
- 配置：冻结VAE、训练306176参数field；同一场Euler积分，h0=0，D为5696维全局Mask+log响应。Real/Natural Train见E-DATA。
- 指标/结果：旧11C0读出、输出结构、路径；没有完整联合通过。具体每K数值须读取C0_metrics.csv，不凭目录名填写。
- 来源：`outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/REPORT.md`、config.json、code/model.py、vae.py、training_history.csv、parameter_sharing_report.json。
- 支持：共享参数与完整输出实现、受控目标与路径研究。不能支持：积分步数增加必然改善、连续物理过程已重建。正式论文可用作方法及受限实验。

### E-P45（L2结构/配置，L1完整成绩）

- 目标：比较Continuous-only A、复用P44 B、D侧Mask STE C。
- 数据同P44；同初始G，A的D参数559809，B/C为1106625。A/C新训练，B复用，不算三次独立训练。
- 结果：没有任一组/K通过11/11原联合；A_K1成为后续F0距离基线。A/B容量不相同，不能将差异全归为一个纯Mask开关。
- 来源：同名目录REPORT、config、gradient_audit.csv、initialization_audit.json、A/C训练日志和C0_metrics。
- 可支持梯度路径和退化对照，不证明显式Mask永远无用或A已完成迁移。可用于方法消融。

### E-P46（L1）

- 目标：Real–Real自然变异包络能否作为迁移等价参照；冻结P44/P45候选。
- 结果：RVC会接受Original、Constant或无依赖donor，且受参考组/距离影响，NO-GO。
- 来源：同名目录REPORT.md、RealTrain_preprocessing.npz的历史定义及负对照表；本轮不重新拟合。
- 支持自然变异参照存在局限，不支持把RVC95≥0.90与历史D≥0.90互换。可作评价方法讨论。

### E-P47（L2指标实现，L1完整表）

- 目标：PRDC、Root内HSIC及结构负对照的互补性。
- 数据为既有12份A/B/C×K输出；无生成器训练。PRDC k3/5/10/20、100采样；HSIC带宽0.5/1/2。
- 结果：部分指标能拒绝退化，但没有新的统一90%口径；PairShuffle不改变边际集合，不能由边际指标识别。
- 来源：同名目录REPORT.md、prdc/hsic/negative_control等CSV、code；详细字段见源目录，不猜文件名。
- 可支持评价分工，不证明任务语义。可用于第五章定义与反例。

### E-P48（L2距离定义，L1历史分类器全表）

- 目标：统一冻结四路线评价，独立审计分类器与积分路径。
- 配置：RealTrain固定标准化，组等权；K为同一区间Euler步数。审计JSD代理族不完整有效，不能当全部分类器通过。
- 结果：A_K1的SWD相对Original约49.77%缩减，为该距离改善，不是“域混淆49.77%”或90%完成。
- 来源：同名目录REPORT.md、distribution/path/equivalence相关CSV、projection_directions与group_bootstrap定义。
- 可支持F0分布改善及更多K不必更好；不能与P46窗口权重距离直接混比。正式图必须同一估计目标。

### E-P49（L2）

- 目标：同一D逐轮适应上一轮F，再冻结D优化同一F；Seed4901、K1、最多3轮。
- 配置：D0原域20Epoch；D1–D3各10Epoch；G各20Epoch；原preserve/flow。原CE差0.2额外硬停止被用户明确改为诊断，旧registration/停止记录/授权修订均保留。
- 结果：最终D3下FRpre0%、post100%，Real接受100%。F0→F3：SWD0.206609→0.411330，MMD0.343678→0.676806；完整质量失败。早期F1样本差异严重收缩，不能隐藏。
- 来源：同名目录REPORT、round_metrics、distribution_metrics、quality_gates、training_log、protocol_amendment_2026-11-19；模型F0–F3/D0–D3保留。
- 支持当前冻结边界可被改变；不支持所有域差异消失。B类部分成功，可正式报告失败/机制结果，不能当新的完整验收模型。

### E-P50（L2）

- 目标：在F0后增加零初始化Refiner及Global/Spatial/Temporal连续D，单Seed5001、单轮。
- 配置：field306176+Refiner120968可训练；VAE/RealDecoder冻结；D688643参数。D660步，G1280步，固定Epoch20终点。
- 结果：Joint FR100%，Real联合接受68.083%；SWD4.247227、MMD0.882140，均显著差于F0。关闭终点Refiner仅诊断时SWD0.340365、MMD0.365503。C类失败。
- 机制与限制：每支有有限非零梯度不等于质量改进；preserve对整体幅值缩放不敏感，flow不约束新增R。实际用全窗组加权Loss而非严格每batch组均衡采样；重载近零Mask数值敏感性保留。不能说同容量公平证明三视角必然更好。
- 来源：同名目录REPORT、architecture.py、D50_metrics、acceptance_metrics、distribution_metrics、quality_gates、branch_gradients、loss_invariance_audit、inference_audit；F50/D50权重及原输出。
- 论文可用作架构受控探索及失败归因，不采用终点去Refiner替换主模型制造成功。

### E-P51（L2；当前停止依据）

- 目标：冻结D50、只读分数及分组泛化审计；训练步数0。
- 校准：8个历史inner Real Session，按Session等权logit2.5/97.5分位；评价4个开发Real Session/10 Sim Root。固定分类阈值0.5。
- 核验数值：Real Joint宏接受0.6808333333；F50 Joint FR1.0、AUC0.02726444845；F50 Joint高于包络上界0.8499348583；Original Joint FR0.78154920454、AUC0.56431896481。字段逐项见verified_claims.csv。
- 每Real Session联合接受87.5%、87.5%、50.6667%、46.6667%。原Bootstrap仅4组，不能夸大总体证据。
- 来源：`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/REPORT.md`、decision.json、classification_metrics.csv、score_summary.csv、session_root_scores.csv、envelope_LOSO.csv、replay.json。
- 结论：NO-GO；D50不满足当前最终评价器使用前提。分数包络不能替代0.5分类验收，也不能独自拒绝RealDonor/PairShuffle。
- 可正式用于评价可靠性和停止分析；不证明全领域不可达。P52、P53未执行，不自动启动。

## 八、正式数字引用卡及后续整理

每个拟入正文数字使用：证据ID→原文件→行/字段选择器→数据角色→宏/微与组数→预处理版本→Hash→可支持结论。可复用verified_claims结构，不把聊天四舍五入值当源数据。

主论文候选材料：E-DATA、E-P40、E-C3/C4、E-P44/P45、E-P48/P49/P50/P51；历史GAN/条件模型与物理代理按论证需要择取，不为凑工作量全堆入正文。失败、未执行和历史修正与正向结果一同保留。

正式图表尚需逐图登记源CSV和绘图命令，这属于材料整理，不是新的训练。不得将本档案自动转写成全部章节；每次只按用户审核的小节核验并写作。

## V2协作同步（2026-11-19）

当前研究范围与七章候选目录见THESIS_PLAN_V2.md，机器人平台、抓取、示教及数据证据见ROBOTICS_EVIDENCE.md；THESIS_PLAN.md为历史六章V1。本文既有技术证据及范本阅读记录保持原范围，不据目录变更回填实验。后续遵守ChatGPT任务书→Codex材料核验并获通过→仅写指定tex→用户审核→用户确认后编译并冻结，Codex不编写或修改任务书、不自动下一节。当前目录尚待审核，无正文冻结。

### V2补核：E-P48距离缩减的CSV定位

`outputs/tactile_p48_unified_four_route_audit_2026-11-19/metrics_swd.csv`，model=A_K1：SWD=0.20660925217182965，Original_SWD=0.41134301198750495，R_W=0.4977202817338595（约49.77%）。该行采用4 Session/10 Root组等权及1024投影W1；其point90=False。与后续P50表中F0值有浮点级差异，应按所属版本原值引用，不抹平成同一精确数。特定距离下降不证明完整域迁移、独立盲测或全部质量指标通过。机器人及示教档案R-01至R-08只对各自列明产物有效，不能用其规模替换本档案冻结训练子集。
