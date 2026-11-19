# P46：Real–Real 自然变异包络与 RVC 审计

日期：2026-11-18。结果性质：开发数据上的评价方法审计，不是新的迁移训练或独立盲测。

## 1. 研究决策

**NO-GO：当前 RVC 不适合替代冻结 C0 主协议，也不能据此宣布新的 0.90 验收通过。**

Real 自身存在明显采集组差异，使用自然变异作为参照有意义。但当前包络会同时接纳未迁移 Sim、退化常量输出和无输入依赖的 Real donor；同一个方案的 RVC 还随参考组和距离变化。因此它目前只适合辅助描述，不能证明迁移等价性。

没有修改旧的 D-only 验收包，也没有改变原 11C0 的 BA/AUC/D 联合标准。RVC95≥0.90 与历史 D≥0.90 是不同定义，不能互换。

## 2. 数据与保护范围

- Real Train：5153 窗口、32 Session、755 Trial；Real Validation：780 窗口、4 Session、120 Trial。
- Natural Train：4049 窗口、50 Root；Natural Validation：905 窗口、10 Root，五任务各 2 Root。
- P44 Hard-Mask（B）、P45 Continuous-only（A）、STE-Mask（C），各 K=1/2/4/8，共 12 份已保存 Validation 输出。P45 B 复用 P44，不重复算成新方案。
- 每份输出逐项核验历史文件 SHA、窗口索引、Root、阶段、数值 shape/finite/非负；均对应同一批 905 窗口。group_inventory.csv 保留逐组数量及数值 Hash。
- 仅读取注册 Train/Validation 数值行。历史 Test/Holdout 未读取；历史文件整体 SHA 校验不是读取其中受保护角色的数组。
- 未进行转换器推理、神经网络训练、生产程序访问、commit/push。仅训练 8 个 Train 内 Logistic C2ST 诊断模型，与原 C0 完全分开。

本轮 Real Validation 为 C3/C0 注册的 4 Session/780 窗口，不冒用其他阶段的 7 Session/1408 窗口。

## 3. 预注册测量方法

主要输入是完整 4×712，展平 2848 维，没有删 Taxel、时序或更换成潜空间。

`u=(log1p(x)-mean_RealTrain)/max(std_RealTrain,0.05)`；所有拟合仅用 Real Train。固定 512 个 Train 窗口的成对距离中位数确定 σ=30.251723；RBF 带宽为 0.5σ、σ、2σ，多尺度核平均为主指标。保存 V-statistic MMD²、其平方根，以及**带符号的 U-statistic MMD²**；U 不裁零、不强行开根号。Energy 同时保存 `E_V=2E||X-Y||-E||X-X'||-E||Y-Y'||` 和 sqrt(E_V)。全文 Energy 列指平方根距离。

Train 标准化尺度有 86.62% 坐标触及预定 0.05 下限，因此本结果依赖这一明确的测量几何，不能声称单位无关。Train 配对距离 CV=0.319，σ 核非对角值标准差 0.165，未出现所有核值近似相同的完全退化；这不等于排除了高维统计功效问题。

32 个 Session 固定分成 4 个 8-Session 参考组。主分析预先指定 reference_block_0，另外三组只作敏感性，未按成绩挑选。每组余下 24 Session 校准。

最小 Real Session 只有 7 窗口，不能删除它。统一查询 7 个原窗口，参考侧从 8 个独立 Session 选 7 个，各取一窗；每次两侧样本量相同，校准 Session 与参考 Session 不重叠。固定 32 次抽样，先平均 V 统计再开根号，得到每个真实 Session/Root 的一个距离。32 次不是独立采集单位。

q95 是 24 个 Session 距离的经验线性分位数。rho_NV 是平均 Root 距离/q95。RVC95 是 10 个原始 Root 中不超过 q95 的比例。全文 RVC90 指使用 RR 的 q90 包络，**不是 90% Root 目标**；候选目标始终是 RVC95≥0.90。

300 次 Bootstrap 同时重抽校准 Session 和按任务分层重抽 Root，条件于固定参考组、预处理与已生成输出。区间未包含全部参考组/Scaler/模型训练不确定性。四参考组另行报告。没有增设“区间下界必须≥0.90”的通过规则。

## 4. Real 自身差异有多大

主参考组 MMD 的 q50/q90/q95/q99 分别为 0.4808/0.7077/0.7235/0.7337；q95 条件 Bootstrap 区间 [0.6464,0.7366]。Energy 对应 3.9081/5.5880/5.7619/5.8225，q95 区间 [5.1232,5.8370]。

24 个校准 Session 的 MMD 范围 0.3676–0.7366，Energy 3.0207–5.8370。4 个 Real Validation Session 在主参考组两种距离下均落在 q95 内，但只有 4 个，不能认证总体 95% 覆盖率。

更换四个预注册参考组，MMD q95=0.7235/0.6503/0.7061/0.6587；Energy q95=5.7619/5.1759/5.6316/5.2582。上尾仅由 24×0.05≈1.2 个 Session 支撑，重复 32 次或 300 次不会增加独立 Session 数。

64 次 4-vs-4 Session-disjoint 宏观划分另见 RR_macro_grouping_sensitivity.csv，每侧最多 128 个窗口。它们仍复用 32 Session，也具有不同的分组构成和估计噪声，不能替代 7-vs-7 Root 包络分母。

**归因边界：**能确认采集组异质性，不能仅凭本轮把它分解为纯 Session、Trial、物体或硬件原因。Session 与配置、采集批次、物体和时间耦合；RR_session_configuration_diagnostics.csv 保存这些元数据关联，没有做因果归因。原始驱动和 calibration 缺失，配置开关不等于实际执行证据。

## 5. 原始和迁移结果

A=Continuous-only；B=P44 Hard-Mask；C=STE-Mask。前五列采用主参考组匹配采样；最后两列为全部原始 Validation 窗口的未经重加权边际距离。**不能把最后两列除以前面的 q95。**

| 方案 | Root均距 MMD | rho MMD | RVC95 MMD | Root均距 Energy | RVC95 Energy | 全量 MMD | 全量 Energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.5501 | 0.760 | 1.0 | 4.9296 | 0.6 | 0.4336 | 3.6041 |
| A_K1 | 0.4180 | 0.578 | 1.0 | 3.3723 | 1.0 | 0.4043 | 3.0648 |
| A_K2 | 0.4187 | 0.579 | 1.0 | 3.3826 | 1.0 | 0.4045 | 3.0712 |
| A_K4 | 0.4192 | 0.579 | 1.0 | 3.3874 | 1.0 | 0.4050 | 3.0753 |
| A_K8 | 0.4195 | 0.580 | 1.0 | 3.3892 | 1.0 | 0.4052 | 3.0767 |
| B_K1 | 0.7679 | 1.061 | 0.2 | 6.1242 | 0.2 | 0.5514 | 4.2242 |
| B_K2 | 0.7705 | 1.065 | 0.2 | 6.1538 | 0.2 | 0.5543 | 4.2574 |
| B_K4 | 0.7713 | 1.066 | 0.2 | 6.1640 | 0.2 | 0.5553 | 4.2708 |
| B_K8 | 0.7718 | 1.067 | 0.2 | 6.1700 | 0.2 | 0.5557 | 4.2768 |
| C_K1 | 0.6367 | 0.880 | 1.0 | 4.9763 | 1.0 | 0.6807 | 5.2216 |
| C_K2 | 0.6397 | 0.884 | 1.0 | 5.0028 | 1.0 | 0.6823 | 5.2308 |
| C_K4 | 0.6382 | 0.882 | 1.0 | 4.9913 | 1.0 | 0.6810 | 5.2203 |
| C_K8 | 0.6380 | 0.882 | 1.0 | 4.9883 | 1.0 | 0.6804 | 5.2153 |

未迁移 Original 的主分析 MMD RVC95 已为 1.0，而 Energy 只有 0.6。这不证明原始域已对齐，只表明宽泛且混杂的 RR 包络可以覆盖某些未经迁移的统计量。

**A 确有距离下降。**A_K1 的主分析 Root 距离相对 Original：MMD 差 -0.1321，配对 Root Bootstrap 区间 [-0.1480,-0.1121]；Energy 差 -1.5573，区间 [-1.7568,-1.3342]。同一批 10 Root、固定参考、固定采样下两者一致。A_K2/4/8 同方向，K1 仅为完整列举中的最低观察距离，不据此选择新模型或重训。12 方案×多指标属于开发比较，未做多重比较校正，不能包装为最终确认性显著性。

B 的两种主距离均变差。C 的 MMD 变差；Root Energy 差的区间含 0。C 的全量 Energy 却明显升高，说明“整体混合分布距离”与“单组查询对参考的平均距离”是不同估计目标，不能只留好看的一列。

A 距离较小并不消除 P45 已观察到的幅值尾部异常。本轮结构表显示 A_K1 的相邻帧绝对变化 0.005106，而 Real Validation 为 0.000631；C_K1 跨窗口标准差均值仅 0.00519，Real 为 0.04644。边际距离和输入/结构保持必须分开评价。

## 6. 参考组、Kernel 与退化负对照

以 K1 为例，四参考组下 MMD RVC95：Original=1.0/0.6/1.0/1.0；A=1.0/0.9/1.0/1.0；B=0.2/0.0/0.1/0.1；C=1.0/0.4/1.0/0.4。不能选第一组把 C 说成稳定通过。其他 K、单带宽核、Energy 全部见 rvc_metrics.csv、figures/reference_sensitivity.png。

**关键证伪：**将每个输入都替换为同一个 Real Train 均值，主参考下 MMD 和 Energy 均有 10/10 人工诊断块落入 q95；无条件抽取 Real Train donor 也为 10/10。它们不保留输入响应。这里的块不是实际 Root，未计入迁移 RVC，但足以表明该边际包络不能识别某些退化或无输入依赖生成。

这是检验指标局限的负对照，不是伪造一个新迁移成绩。RVC 可以作为分布差异描述；若作为完整触觉迁移验收，则当前构造存在明确反例。

## 7. Train 内 C2ST

仅使用 Train：Real 32 Session 固定分成两个人工伪域，每域 16 Session；4 折各留出 4+4 Session。原始 Sim 50 Root 按任务分成 4 个互斥折，与相同 Real held-Session 折配对。模型是独立 Logistic，固定 L2/C=1 等效、类别平衡损失、最多 100 LBFGS 迭代，使用全部符合折身份的窗口。每折标准化仅拟合 Real fit-fold，不使用 held-fold。没有拟合或修改旧 C0。

| 对照 | 四折平均 BA | 四折平均 AUC | 四折平均 D | 平均AUC条件组Bootstrap区间 |
|---|---:|---:|---:|---|
| Real_vs_Real | 0.2836 | 0.3435 | 0.5671 | [0.1746, 0.5386] |
| OriginalSim_vs_Real | 0.9424 | 0.9742 | 0.1152 | [0.9483, 0.9912] |

这些是四个固定训练模型的折均值，不能当作一个部署分类器的混淆矩阵。逐折 Train/held 指标、优化器迭代、收敛警告、分数及 300 次组 Bootstrap 完整保留。达到 100 次上限的模型未认证收敛，不以它们不能区分作为混淆证据。

Real–Real 的伪域标签没有跨 Session 的固定物理含义；某些 held-fold AUC 显著小于 0.5，表示排序反转而不是“更混淆”。c2st_direction_diagnostic.csv 额外给出 1-AUC/max(AUC,1-AUC)，仅作诊断，不翻转正式标签、不替换正式 AUC。因此不能机械地用 RR 折平均 AUC 给 Sim 指标放宽门槛。

P44/P45 没有保存迁移后的 Train 输出。本轮不额外运行生成器、不拿 Validation 拟合分类器，所以这 12 组迁移后 C2ST 为 NOT_AVAILABLE。**不是已经通过，也不是以冻结原始分类器代替新 C2ST。**

## 8. 任务、阶段、时间和独立性

- Real 是 hardness/slip/stable 等采集协议，Sim 是五个操作任务；没有认证的共同物理任务条件。
- CLOSE/HOLD/LIFT 仅作字面大小写匹配。每方案用相同阶段采样索引；其余阶段不从全量主结果删除。task_stage_marginal_distances.csv 同时保存各任务对完整 Real 参考的描述，不能称为纯传感器域差。
- Real 712D 与 Sim 的物理 Taxel 对应、单位和校准关系未认证；完整索引上的数值比较不等于真实空间位置相等。
- Real 四帧时间顺序由已有 C24 合同核验，严格递增但存在不均匀采样；Sim 为约 20Hz。因此相邻帧统计是现有窗口结构诊断，不能不控制采样间隔就归因为材料时间常数。
- 一个 Real Session 可包含多个 Trial，Sim Root 多为一个 Episode。统一窗口数消除了部分样本量差异，**没有消除采集单位和条件构成的不可交换性**。
- 10 个开发 Root 的 RVC 以 0.1 跳变，且都已被历史开发使用。即便观察 10/10，假设固定阈值、iid Root 时的描述性双侧精确区间仍约 [0.692,1]；它不是当前异质 Root 总体的认证区间，也不是新加的通过门槛。
- A 的条件 Bootstrap 可出现 [1,1]，仅表示在这批经验组和固定参考下重采样都覆盖，不表示总体不确定性为零。

## 9. 最终九问

1. **Real 自身差异多大？**匹配设计下主 MMD q95=0.7235，Energy q95=5.7619；均依赖已声明的预处理与单位构成，没有通用物理单位。
2. **来源是什么？**采集组/配置/任务/物体耦合的异质性已确认；纯 Trial、硬件、处理链各自贡献 UNKNOWN。
3. **原始 Sim 比 Real 内部大多少？**主 Root 均距比为 MMD 0.760、Energy 0.856，未超包络均值比；但 MMD/Root 与 Energy 覆盖、C2ST 不一致，不能推出原始 Sim 等价。
4. **P44/P45 是否缩小差异？**A 的两种主距离及全量距离均缩小；B 变差，C 没有稳定一致的改善。
5. **哪个最接近？**在这组固定距离下 A 各 K 最接近，K1 最低观察值；这只是开发描述，不代表联合域验收或结构保持通过。
6. **90% RVC 目标可信吗？**作为清楚定义的经验比例可以计算；当前不够支持“迁移等价性”的正式解释，负对照、参考敏感性和采集单位差异均未解决。
7. **能证明总体至少 90% Root 进入自然范围吗？**不能。可报告这批 10 Root 的点估计及条件区间；尚无可信的总体覆盖认证。
8. **可以替代 C0 吗？**不可以。当前 NO-GO 针对“替代正式验收”，不是否认自然变异参照的研究价值。
9. **替代方案？**预先明确共同操作/接触条件及目标总体，以 Trial/Episode 为查询单位、Session/Root 为外层聚类，固定独立 Real 参考与校准/验收组，匹配样本量和条件构成；先验证负对照、参考敏感性及物理结构/输入响应门槛，再预注册非劣/等价容限与覆盖判据。若现有条件不足，只保留开发距离对照和冻结 C0 联合结果，不宣布新 90%。这只是后续设计建议，本轮不实施。

## 10. 交付与复现

主要文件：registration.json、group_inventory.csv、condition_audit.csv、real_real_distances.csv、q_RR_quantiles.csv、root_distances.csv、sim_real_distances.csv、translated_sim_real_distances.csv、rvc_metrics.csv、bootstrap_results.csv、negative_control_checks.csv、paired_method_differences.csv、c2st_metrics.csv、c2st_direction_diagnostic.csv、structure_diagnostic.csv、task_stage_marginal_distances.csv、figures/。

复现需原项目注册数据与 Python numpy/scipy/torch/matplotlib/PyYAML；不需要 sklearn。代码按 prepare.py → compute_distances.py → c2st.py → report.py 执行，输出目录应全新，prepare/完成标记阻止误覆盖。C2ST 用 float64 Torch LBFGS 实现平衡 Logistic。run_commands.json 保存命令；输入输出 Hash、模型保护与数值检查见 integrity.json/verification.json。

执行中修正了一个本轮脚本命名问题：初名 statistics.py 遮蔽标准库，导致 Torch 导入意外触发重复的确定性距离计算；中止后更名为 compute_distances.py。没有新模型训练或历史修改。日志保留原异常。当前完整统计取已完成的确定性结果；重入部分使用相同种子、数据和公式，不作为新增独立实验。

**P46 到此停止；未开始 P47。**
