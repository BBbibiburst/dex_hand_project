# 全篇核心结论证据总表
日期：2026-11-27。Stage A只读审计。论文位置以当前主文件及93页PDF为准，PDF页为物理页（正文印刷页=PDF页−8）。文件路径相对仓库根目录，行号含CSV表头。VERIFIED只代表直接核对该代码/登记/既有结果，不是重新运行模型；UNKNOWN/UNVERIFIED不补成结论；本轮没有L3。

## 一、证据定位规则
目录简写仅为审计索引，不是论文引用；下文每条来源均解析到实际路径。原论文只能证明“写了什么”，核心值优先回查原CSV/JSON。历史summary的执行含义仍受当时配置限制；不以本轮读取升级为新实验。
## 二、第三、四章核心证据

|编号/章|结果及精确分母|数据角色/Seed|直接来源|论文位置|可支持与不可支持|
|---|---|---|---|---|---|
|E01/3|100条候选路由：98 LATTICE_SUCCESS、2 RL_SUCCESS|历史筛选清单；Seed逐对象，不存在统一独立Seed|`outputs/dex_hand_top100_v2/summary.json:3298`|ch03:252–271；PDF41–42|分级路由已有产物；不是100次FINAL通过或算法排名|
|E02/3|PickPlace100 SUCCESS，其中cached100；Push100 SUCCESS，其中cached98|历史记录；非本轮新执行|`outputs/pick_place_top100_final/summary.json:4`|ch03:252–271；PDF41–42|缓存不计新物理验证；Push另见outputs/push_top100_final/summary.json|
|E03/3|Stack task-valid48条：43成功、5失败|历史v24筛选域；Seed UNKNOWN|`outputs/stack_task_valid_v24/summary.json:1064`|ch03:264；PDF42|不能拿旧stack_top100_final的47/41/6替换本组48/43/5|
|E04/3|Nut100 SUCCESS登记，单位variant|历史旧几何；Root UNKNOWN|`outputs/nut_assembly_top100_final/summary.json:4`|ch03:266；PDF42|不是100独立物体或当前模型泛化|
|E05/3|Sugar10条历史Strict FINAL_VERIFIED；Seed18011–18020|同物体源模板局部扰动|`outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p00/candidate_00/strict_replay_final.json:20`|ch03:273–297；PDF42–43|每条独立保存结果但同源关联；不称未见场景10/10泛化|
|E06/3|90尝试：46接受、18 Strict不稳定、26 Lift失败|历史汇总，Seed与逐例版本须原档案|`outputs/dp_p1_2026-10-11/generation_formal_summary.json:2`|ch03:307–314；PDF43|互斥汇总可加；重叠诊断不可加；不与Sugar10叠加独立分母|
|E07/3|Strict：抬升≥.055m、线/角速度≤.1、接触/抓持≥.8、尾桌触0、穿透≤.003m|final profile；非所有任务通用同一验收|`source/verification/strict_replay.py:34`|ch03:181–196；PDF36–38|当前实现约束与历史profile需分别绑定，非物理实机认证|
|E08/4|LeRobot5/1550、30/14310、1/340|meta声明Train，历史版本非独立Root|`outputs/dp_2026-10-11/lerobot_5/meta/info.json:4`|ch04:220–229；PDF52|另外两原件见DATA_LINEAGE；不能相加36Root|
|E09/4|修复340行：action/命令段/触觉指定映射最大绝对差0；源477行|指定Sugar source，非全部字段/RGB认证|`docs/writing_audit/ch04_material_audit_v1_2026-11-20/dataset_checks.json:548`|ch04:140,229；PDF49/52|本轮复核历史检查原件和映射代码；未重新读全Parquet/视频，不冒充本轮重算|
|E10/4|旧PickPlace373个相邻转移超1e−6，最大约.100000024；旧Stack缺pre_qpos/pre_qvel|旧在线开发例|`docs/writing_audit/ch04_material_audit_v1_2026-11-20/dataset_checks.json:691`|ch04:169–171,239–240；PDF50–53|已保存检查事实；不指定唯一原因/不称已修好|
|E11/4|v4五包2727行，5lineage/4asset_hash/5content，均development_verified_not_formal_production|开发包；独立Root UNKNOWN|`outputs/five_task_p44_integration_2026-11-05/dataset/episodes/lift_sugar_ac9b12bc48cbf5775e50/episode.json:25`|ch04:241–249；PDF52–53|本轮直接重计5个episode元数据，行数复核既有数组审计；非服务器全库|
|E12/4|同380帧旧qpos偏差1.6288973093032837，新0，回放False→True|Sink资产的Stack历史对照；Seed0|`outputs/five_task_p38_exact_replay_2026-11-03/comparison.json:27`|ch04:251–277；PDF53–54|完整积分状态/float64同时改变，无单因素归因；非新实验|

第三章静态可达性只预检预抓取/抓取两个目标：`source/grasping/executor.py:225`排序20e_p+.6e_R+.08travel；正文3.3:132–134明确非全轨迹碰撞规划。原14物体、127物体批次未有本轮可认证共同原件，保持UNVERIFIED，不拿Top100数量替换。

## 三、第五章合同、训练与版本结果

|编号|精确结果/分母|数据角色及Seed|直接原件|论文位置与结论边界|
|---|---|---|---|---|
|E13|旧P37 Real5153/1408、Sim1824/406；与P43不同|Train/Validation，旧开发|`outputs/tactile_p39_domain_decomposition_2026-11-15/mask_statistics.csv:2`|5.1:7–30；PDF56–57。差异不全归因传感器物理，不外推P43|
|E14|Real5153/780、Natural4049/905；32/4Session与50/10Root|当前Train/Validation|`outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/data_contract.csv:2`|4.5:279、5.2、6.3；PDF54/57/74。与P43索引及windows清单一致|
|E15|字典480行，160放置后指纹，32帧/条|历史探针，无本轮生成|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/dictionary_complete.json:2`|5.3:50–60；PDF59–61。非480或160独立Root|
|E16|Paired1200/240，pilot32/10+expansion1168/230|Train/Validation|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/paired_manifest.csv:1`|5.3:84；PDF61。1440登记关联非同步物理标签；本轮逐role计数|
|E17|Real45轮3645步best30，scale.14118273556232452/.005002501420676708|Real Train，Seed4301|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/real_vae/complete.json:2`|5.4:训练段；PDF62–63。不是域迁移指标|
|E18|Sim45轮5715步best30，scale.04088745638728142/1e−7|Natural+Paired Train，Seed4302|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/sim_vae/complete.json:2`|5.4:训练段；PDF63。32+32等量混合，重复采样非新数据|
|E19|Translator181824参数；45轮855步；选30=570步|Paired1200/240，Seed4303|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/translator/complete.json:2`|5.5:训练段；PDF64–65。Huber+.1条件NLL，不替代自由输出质量|
|E20|Paired Val Huber .32591983675956726 vs均值模板.4942297637462616|240窗；全窗口×4×64；局部Session例外|`outputs/tactile_p43_c4_root_cause_audit_2026-11-16/translator_paired_error.csv:36`|5.5末段；PDF65。精确对照取C4原CSV的validation/ALL/ALL行；history批次精度末位与该全窗汇总不同，不称Natural成功|
|E21|C3 Validation联合1/11、D11/11，仅kNN|780Real+905Natural输出；原C3三Seed|`outputs/tactile_p43_c11_latent_support_audit_2026-11-17/probe_metrics.csv:299`|5.6表5-3/5-4；PDF66–67。本轮只选group F/role validation的299–309行|
|E22|C13 Validation联合1/11、D11/11，仅kNN|同780/905；Real局部幅值修复|`outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv:1`|5.6:针对目标域读出段；PDF66–67。非fresh Probe|
|E23|C23 Real自重建读出6/11|780重建Real vs780原Real；无新VAE训练|`outputs/tactile_c23_real_reconstruction_c0_2026-11-18/metrics.csv:1`|5.6:C23段；PDF67。不是Natural迁移6/11|
|E24|C24 Real自重建读出2/11|同780/780；18轮best2，Seed以registration为准|`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/metrics.csv:1`|5.6:C24段；PDF67。不能与跨域表当同一成功率|
|E25|L_temporal .00281496740036965→.0016136413835846951；MSE .026319601646693502→.02715744920401311|同780 Real验证窗，C13 vs C24|`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/Real_C13_C24_structure.csv:1`|5.6:C24段；PDF67。log1p相邻保留帧误差，非物理导数|
|E26|C27联合2/11、D11/11；kNN、GRU12通过|780Real+905Natural；保留原C3双VAE|`outputs/tactile_c27_c3_route_improvement_2026-11-18/metrics.csv:1`|5.6与7.2；PDF67/86。不是11/11完整迁移，也不叠加C13|

### 原冻结Validation逐实例复核

判定以原始精度BA∈[.45,.55]、AUC∈[.475,.525]、D≥.9计算；不是重算预测或AUC。以下展示六位小数。

|实例|C3 BA/AUC/D|C13 BA/AUC/D|C27 BA/AUC/D|联合通过版本|
|---|---|---|---|---|
|logistic|0.498718/0.682604/0.997436|0.498718/0.658306/0.997436|0.501480/0.673062/0.997039|无|
|knn|0.501944/0.500536/0.996111|0.498629/0.497752/0.997259|0.506364/0.509993/0.987272|C3V,C13V,C27|
|mlp_11|0.497436/0.591571/0.994872|0.497436/0.571536/0.994872|0.498541/0.600164/0.997082|无|
|mlp_12|0.498718/0.598765/0.997436|0.498718/0.589597/0.997436|0.498718/0.630157/0.997436|无|
|mlp_13|0.498077/0.610894/0.996154|0.498077/0.601097/0.996154|0.499734/0.617238/0.999469|无|
|gru_11|0.498718/0.239559/0.997436|0.498718/0.187208/0.997436|0.498718/0.571915/0.997436|无|
|gru_12|0.498718/0.237257/0.997436|0.498718/0.191583/0.997436|0.511978/0.487958/0.976045|C27|
|gru_13|0.498718/0.383622/0.997436|0.498718/0.308620/0.997436|0.511425/0.548114/0.977150|无|
|transformer_11|0.497436/0.282288/0.994872|0.497436/0.266441/0.994872|0.497988/0.357940/0.995977|无|
|transformer_12|0.496154/0.217189/0.992308|0.496154/0.236963/0.992308|0.499469/0.307241/0.998938|无|
|transformer_13|0.498077/0.698591/0.996154|0.498077/0.675462/0.996154|0.503602/0.643598/0.992796|无|

C23读出诊断通过：mlp_11, mlp_12, mlp_13, gru_13, logistic, knn；总6/11。

C24读出诊断通过：transformer_13, knn；总2/11。

### 不可替代来源

- C3 `final_evaluation/metrics.csv`属于历史开发Test分母765Sim/628Real；本轮只作文件角色辨识，不用于当前Validation通过数或表5-4。C27 baseline重读也存在读出后端末位差异；当前表5-4的C3数值来源是上述C11已保存Validation链路，而非任取一个叫baseline/final的文件。
- C14 freshly trained Probe不是原冻结C13结果；C13应使用`outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv`。
- C25未执行跨域OT、C26未运行C0；均不记0/11。未完成的实验不产生失败率。
- C4 `paired_cross_split.json:24–34`的19指纹/97验证窗支持登记复用风险，不证明物理全状态相同。

## 四、第六章核心结果

|编号|指标/精确值|分母、角色、Seed|原始文件及行|论文位置/可得结论与局限|
|---|---|---|---|---|
|E27|完整G2494640，可训练field306176；同一参数跨K共享|模型参数清单；Seed4401|`outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/parameter_inventory.csv:1`|6.1:48–71；PDF70–71。不是只训练末层，VAE冻结|
|E28|A/B/C联合：A1/0/0/0，B1/1/1/1，C0/1/1/1；分母均11|K1/2/4/8；A/C新训练、B历史复用|`outputs/tactile_p45_mask_ablation_2026-11-18/C0_summary.csv:2–13`|6.2表6-2；PDF72–73。本轮直接复核C0_summary.csv逐K联合数及训练对照|
|E29|P48 OriginalSWD .41134301198750495；A_K1 .20660925217182965；R_W .4977202817338595，区间[.40851387597073363,.5667432326739397]|780/905，4Session/10Root；固定1024投影；非新训练|`outputs/tactile_p48_unified_four_route_audit_2026-11-19/metrics_swd.csv:7`|6.3/6.5/7.2；PDF74/80/86。R_W区间非SWD原值区间|
|E30|P49 FR：D1下.0007751937984496124→1；D2/3下0→1；D3 AUC .9977284537663516|905Sim按10Root宏平均；Real780按4Session；4901|`outputs/tactile_p49_round_frozen_adversarial_2026-11-19/round_metrics.csv:3`|6.3:29–40；PDF75–77。判别边界适应非域不可辨识|
|E31|F0 .20660925215170187/.343677926989022；F1 .530662295753659/.8335668176956493；F2 .2864444503944135/.5179216453173372；F3 .4113299756912987/.6768059340116337|SWD/MMD，780/905；固定合同；4901|`outputs/tactile_p49_round_frozen_adversarial_2026-11-19/distribution_metrics.csv:2`|6.3图6-5；PDF76。非单调改善，也非简单单调恶化|
|E32|Original .4113430117241668/.39255359721971717；F3 SWD近Original但MMD更高|同P49合同，未转换Natural|`outputs/tactile_p49_round_frozen_adversarial_2026-11-19/distribution_metrics.csv:11`|6.3:38，6.5；PDF76/80。非零潜变量解码Original|
|E33|F50 SWD4.247227259017046、MMD.8821400323571661；关闭Refiner诊断.3403651960851213/.36550345430385656|905生成/780Real；单Seed5001；终点20轮|`outputs/tactile_p50_residual_multiview_2026-11-19/distribution_metrics.csv:5`|6.4表6-3；PDF78–79。诊断不替代主结果|
|E34|F50 Joint FR1；Real Joint接受.6808333333333334|10Root/4Session宏；Seed5001|`outputs/tactile_p50_residual_multiview_2026-11-19/acceptance_metrics.csv:17`|6.4:52，6.6；PDF79/82。两个比例统计对象不同|
|E35|F0 P@5 .108859/R@5 .039936；F3/F50 P=R=0；关闭Refiner P .243167/R .000321|100共用子样本，k=3/5/10/20；开发Validation|`outputs/tactile_p50_residual_multiview_2026-11-19/PRDC_metrics.csv:1`|6.5:25–35；PDF80。邻域定义非通用物理成功率|
|E36|HSIC Original1，F0约.507147，F3约.492566，F50约.514820；PairShuffle约.312631|Root内中心化核再宏平均；开发905窗|`outputs/tactile_p50_residual_multiview_2026-11-19/HSIC_metrics.csv:1`|6.5表6-4/6-5；PDF80–81。统计依赖≠输入物理语义|
|E37|D50 Real Joint .6808333333333334；CI[.4866666666666667,.875]；F50 AUC .02726444844879738|4Real Session/780，10Root/905；1000组Bootstrap|`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/classification_metrics.csv:17`|6.6:8–26；PDF82。固定Real=1，不反向AUC|
|E38|Real窗口566接受/214拒绝；合并566/780=.7256410256410256；CM[[0,905],[214,566]]|矩阵行真实Sim/Real，列预测Sim/Real；窗口计数|`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/classification_metrics.csv:17`|6.6:8/26；PDF82。合并比例不替代Session宏|
|E39|LOSO均值G .7708333333333334/S .811235119047619/T .9471279761904762/J .8557291666666667；最低.0416667/.2041667/.8/.1291667|8inner Session，每次7校准1检验，非独立外层盲测|`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/envelope_LOSO.csv:1`|6.6表6-7；PDF83。均值不能掩盖最低值|
|E40|NO-GO；原Real .90要求未满足，训练步0|P51冻结D50历史审计；本轮不推理|`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/decision.json:2`|6.6:67及7.2；PDF84/86。仅D50，不外推D3/整个方向|

### 历史数值及协议边界

P49 `protocol_amendment_2026-11-19.json:2–22`明确：CE fit/inner差0.2从硬停止改为诊断；保留原登记和停止记录；从D1/F0及优化器恢复，不重训D1，其他阈值/Seed/预算不变。论文6.3:20已披露，属于修订后的开发实验，不是始终未变预注册执行。7.2未将FR写成整体成功，因此不构成跨章结论冲突。

P50 `inference_audit.json`及已有6.4审计报告：初始CUDA/CPU最大差1.5591305494308472、Mask169差异；终点重载8.58306884765625e−6在allclose内，但近零Mask5695条不同，响应上限4.76837158203125e−7。不同零比例约.491242与.500657是未加权与组等权口径，不互相覆盖。本轮不重算输出。

D50实际文件`outputs/tactile_p50_residual_multiview_2026-11-19/checkpoints/D50.pt`字节SHA256为`f436a4a3b4ad93f9c5853da89873dcb51b3be2a2564a1965e36b17e20883533a`，与P50 checkpoint_hashes.json及P51 replay.json:2一致。只计算字节Hash，不加载模型。P51 replay.json最大logit差0是历史重放证据，不是本轮执行。

PRDC、HSIC、SWD和MMD公式逐行核对P49 `code/evaluate.py:15–58`、`code/support47.py:20–36`：加权CDF积分、混合RBF V统计平方根、严格小于近邻球、Root内中心化核对齐均与正文一致。P48 MMD采用自己的抽样/距离代码，不能将P48的.3693778962覆盖P49/P50的F0 .343677927。SWD约2e−11的来源差异已在6.5:14保留。

### 状态基准

- 已有实验/历史结果：C3、C13、C27跨域，P44/45/49/50优化。
- 派生诊断：C23/C24 Real读出、关闭Refiner、Constant/RealDonor/PairShuffle、P51包络和LOSO。
- 未执行：C25跨域OT、C26 C0、P52/P53；不记0/11或伪造失败率。
- 推测/待验证：更多软配对、物理监督和实机收益；不得当成果。

## 五、本轮直接定位原件Hash索引

下列Hash绑定本轮所读原件；不代表对其历史执行做L3复验。代码/个人论文源的保护另见逻辑总账及问题清单执行记录。

|原件|SHA256|
|---|---|
|`docs/writing_audit/ch04_material_audit_v1_2026-11-20/dataset_checks.json`|`e3e50328e349b84ac165b7c90f8d67e7092e586f5d5e8a76bec6952f83065315`|
|`outputs/dex_hand_top100_v2/summary.json`|`e9a23b52f43d434d9634b25ae92ebc38fc96e2e81b84db5a130c0574bb9c77c3`|
|`outputs/dp_2026-10-11/lerobot_5/meta/info.json`|`b2444c6b35d06aa82a505607decdaa7ae6e6294208cb05c43c0e85cb96307f4b`|
|`outputs/dp_p1_2026-10-11/generation_formal_summary.json`|`6ba85fb341b30cfc690a616c24189b878f0e6a3e50ca51772d61b089e7364540`|
|`outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p00/candidate_00/strict_replay_final.json`|`65ab7496265b0c8871d4774072f80c0ad0f38e16a40bfab59ed7f3ad6302e8aa`|
|`outputs/five_task_p38_exact_replay_2026-11-03/comparison.json`|`d0e520fd82db0c9d59889c6572fbf14900e0d9a9d11308b830ea5faae775b9da`|
|`outputs/five_task_p44_integration_2026-11-05/dataset/episodes/lift_sugar_ac9b12bc48cbf5775e50/episode.json`|`6c9904d7d88cd9815777e2be7135975d5962bff2dd56a0778a4e4f8bd7ae65c2`|
|`outputs/nut_assembly_top100_final/summary.json`|`42ca95049066483a19a58d77d116311478f082aa02f35c03bbdfc854ad5c4342`|
|`outputs/pick_place_top100_final/summary.json`|`a70295cbaee6cc3a132d08d1fabf0c1080beb5c3914b660bf72f6e19a11a64d5`|
|`outputs/stack_task_valid_v24/summary.json`|`18ec12b71cb095e7dc40984364060ffb5e5d290a6fa6b67bceec964727f8bc40`|
|`outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv`|`7ab17c8bfc8485801042906411cbe7589ef25943dee9cf484fcdb67974347954`|
|`outputs/tactile_c23_real_reconstruction_c0_2026-11-18/metrics.csv`|`4334f38eb7bd7e402a8911e2eaa2b4c0c755130d0a70455bc11cf06cff06fcb0`|
|`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/Real_C13_C24_structure.csv`|`78c759cc46137f929dccca7bf39905c0516df673fd90153c6eef89fc35267f42`|
|`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/metrics.csv`|`b085ad746b3c0d0f1d782ddc4bae8c5a4c2f18d2cfd1f5969d7d547380273f1f`|
|`outputs/tactile_c27_c3_route_improvement_2026-11-18/metrics.csv`|`68d935242c7eb542354951a77b3e99353c7b0103c1fff234d7d393989559dde9`|
|`outputs/tactile_p39_domain_decomposition_2026-11-15/mask_statistics.csv`|`6e6a805a9384180825229e2e1bb20b75d1339e1ee33a428451391abf5a9bb19f`|
|`outputs/tactile_p43_c11_latent_support_audit_2026-11-17/probe_metrics.csv`|`cda446cf2f42032a0eef262fe998305d70eae99cb666bce0819924d12068aec5`|
|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/dictionary_complete.json`|`330080d70f43291bd9129a6df965dc04be6dcb6618adf198157bd3d142a56ad2`|
|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/paired_manifest.csv`|`1c305bca621a241a6fbe1a18318223f1a778928e244ac14aea30939a0f17adb3`|
|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/real_vae/complete.json`|`54877440d6ce49732ba44e8f8aad108c202554b3adfc5731c13e249a79cba99c`|
|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/sim_vae/complete.json`|`d79dc74ea6c4ae9a22c96cd78e67c6ef03cb815e08fec64a1104026f2c27432f`|
|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/translator/complete.json`|`c99d59618d916488c9c51d5b96bf478b46b260e969d22040329dd1c0f0cdfb93`|
|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/translator/history.csv`|`33a91a8d9dfb8f9b21760a9ab81e2d602a79ee02b5220418d266bf72b2921e38`|
|`outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/data_contract.csv`|`5408a12416fa814d3b54b5b3f4f77207fe3cb491093b7c538632462d487aaf77`|
|`outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/parameter_inventory.csv`|`9056e31bf32acbdeacca7ddf47d786631cab9cb1ee89fcf4bf480d0780649ed2`|
|`outputs/tactile_p45_mask_ablation_2026-11-18/training_comparison.csv`|`1e7afa78799192fab8d5bb198fec3ea4679cb1002d3a0ba0bfdbaf4a14259185`|
|`outputs/tactile_p48_unified_four_route_audit_2026-11-19/metrics_swd.csv`|`15d5106776442af3d2907713a539a4a2a5adc6ddec4210d844fc355386e2e233`|
|`outputs/tactile_p49_round_frozen_adversarial_2026-11-19/distribution_metrics.csv`|`e786b2d2c184d996999a763e92fc50297d02a26ef449c56a7b96183c26ebf5f5`|
|`outputs/tactile_p49_round_frozen_adversarial_2026-11-19/round_metrics.csv`|`bb4486fc4a908db27c19315406572b61264c6f3f79d5a04d61607aee5dbd00c3`|
|`outputs/tactile_p50_residual_multiview_2026-11-19/HSIC_metrics.csv`|`d3345403ed1d78adaf21e8f210ef40c640c8a54687133e3295f0d38aa2b5d01a`|
|`outputs/tactile_p50_residual_multiview_2026-11-19/PRDC_metrics.csv`|`95c8ba78fca93287a09d5ae93f0472d75763f2066e8e22574a946644d93b21d8`|
|`outputs/tactile_p50_residual_multiview_2026-11-19/acceptance_metrics.csv`|`7fb15fd6456706e1e064e09de1f5d7b21682e40f3fe3abd179ad612968512dc5`|
|`outputs/tactile_p50_residual_multiview_2026-11-19/distribution_metrics.csv`|`ccb44291e0cf1e3b7dbccaac7bac881c4def5fdaa1d4eb47c7a18a37c93c2b1c`|
|`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/classification_metrics.csv`|`dc58386b742b1988f2bc56e7f157413caffa936ca4a459ab2d94bbb9866347dc`|
|`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/decision.json`|`36490d8df81ee0fd26c8c0e631d787fee7d6024b96d88fdcccdbf085a9c42b73`|
|`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/envelope_LOSO.csv`|`050e4872991822b5f9bbfe79bf15c8369708f76733c81d4c2963b1fa44bae904`|
|`source/verification/strict_replay.py`|`25edfdfbd2c10a5fae26b722fb06c6f105e16569deae12fe7352fa7e43fcb830`|

### 补充精确来源Hash

|原件|SHA256|
|---|---|
|`outputs/tactile_p43_c4_root_cause_audit_2026-11-16/translator_paired_error.csv`|`17740386c51d677422a4ec99ec402037a803b616f96438dbb737c48d82f24a9c`|
|`outputs/tactile_p45_mask_ablation_2026-11-18/C0_summary.csv`|`8344b909b20f13655921c41fbe2d297fc36f467cf256be53c51286b98713ff57`|
|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/paired_manifest.csv`|`1c305bca621a241a6fbe1a18318223f1a778928e244ac14aea30939a0f17adb3`|
|`outputs/tactile_p43_c4_root_cause_audit_2026-11-16/paired_cross_split.json`|`052e7e82b4a6affdc8d473c34381add6c7d9cf1a74d7f2c7bd6371db404c4997`|
|`outputs/tactile_p49_round_frozen_adversarial_2026-11-19/protocol_amendment_2026-11-19.json`|`10d4398a072cf42b5abdf2b1405fa8915de54fae7af27d004c54176e8185f7ba`|

本轮补充源码Hash：

|文件|SHA256|
|---|---|
|`outputs/tactile_gmm_dual_vae_2026-11-16/code/fit_gmm.py`|`3cd1f8a1cc64ac5ccd05f0ff912d0c872bb4b05929e02732b719cf7f620d33d5`|
|`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/matching.py`|`a23870860ade9568ce887d5d259a07771da70ea1f1f8b4b3b070905388d5c096`|
|`outputs/tactile_p50_residual_multiview_2026-11-19/code/architecture.py`|`d51217cc808d6d9fc650baefb19e1ea2aa5f01e113d00b5a830278e0b569c6b5`|
