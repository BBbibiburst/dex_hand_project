# P47：条件触觉迁移评价体系可行性实验

日期：2026-11-18

项目根目录：`/home/zmy/dex_hand_project`

实验性质：评价方法研究与负对照验证。

核心目标：建立能够同时评价触觉迁移的分布质量、输入条件保持和任务信息保持的评价体系，为后续重新定义迁移质量的 0.90 指标提供依据。

**本轮不训练任何生成模型，不修改 P44/P45，不运行原始 C0，不读取 Test/Holdout，不宣布迁移验收通过。**

---

# 一、研究背景与需要解决的问题

请先完整阅读 P44、P45、P46 的实验报告和相关代码。

目前已经发现：

1. P44/P45 能够生成完整的 4×712D 迁移触觉输出。
2. P46 发现 Continuous-only（A）能够明显缩小 Real–Sim 的部分统计距离。
3. 但是 A 存在明显的幅值尾部和时序结构退化。
4. P46 的 RVC 可以接受 Constant 和 Real donor 等不保留 Sim 输入信息的负对照。
5. 因此，仅使用 MMD、Energy、RVC 或冻结分类器的域混淆程度，不能完整描述迁移质量。

本轮研究的核心假设是：

**合格的触觉迁移必须同时满足三个独立条件：**

A. Distribution Fidelity：生成结果具有合理的 Real-like 分布质量。

B. Conditional Consistency：生成结果仍然与原始 Sim 输入存在正确的对应关系，而不是无条件生成一份 Real-like 数据。

C. Task Information Preservation：迁移结果保留原始 Sim 中可以验证的任务、阶段和时序信息。

注意：这三个条件目前只是研究假设，不能提前认定它们都能通过现有数据验证。

本轮需要判断的是：能否构造出有效、可复现、能够识别退化负对照的评价方法。

---

# 二、实验保护范围与数据

## 2.1 固定数据

以 P46 的数据注册和文件清单为准，核对以下数据：

| 数据 | 已知规模 |
|---|---:|
| Real Train | 5153 窗口、32 Session |
| Real Validation | 780 窗口、4 Session |
| Natural Sim Train | 4049 窗口、50 Root |
| Natural Sim Validation | 905 窗口、10 Root |

以上数字必须通过实际文件重新核验，不得直接相信报告中的数字。

## 2.2 迁移候选

使用 P44/P45 已经保存的 Validation 输出：

- A：Continuous-only
- B：Hard-Mask
- C：STE-Mask
- K=1、2、4、8

总计12个配置。

其中 P45 B 对 P44 B 的复用必须保留原始身份，不能当成独立实验。

同时使用：

- 原始 Natural Sim Validation。
- 原始 Real Validation。
- Real Train。

不得重新运行生成器来获得更好的结果。

## 2.3 严格禁止

- 不训练 Dual-VAE。
- 不训练 Translator。
- 不训练 GAN、Diffusion、Flow。
- 不训练新的神经网络判别器。
- 不重新训练或修改11个冻结 C0。
- 不重新拟合原有 Scaler。
- 不修改原始数据。
- 不修改 P44/P45/P46 产物。
- 不读取 Test/Holdout。
- 不运行实机。
- 不 commit/push。

允许进行确定性的统计计算、核方法、距离计算、分组 Bootstrap 和必要的负对照构造。

所有新增脚本和产物必须放在全新的 P47 输出目录。

---

# 三、P47-A：Density / Coverage 生成质量评价

参考论文：

Naeem et al., ICML 2020, *Reliable Fidelity and Diversity Metrics for Generative Models*。

Kynkäänniemi et al., NeurIPS 2019, *Improved Precision and Recall Metric for Assessing Generative Models*。

目标：判断迁移数据是否真正进入 Real 数据的合理分布区域，以及是否覆盖 Real 数据的多种模式。

## 3.1 计算四项指标

使用 PRDC 方法：

1. Precision：生成样本是否位于 Real 局部支持区域。
2. Recall：Real 样本是否能够被生成数据覆盖。
3. Density：生成样本落入多少个 Real 局部邻域。
4. Coverage：有多少 Real 样本能被生成样本覆盖。

必须使用原论文的定义实现，不能自行改变公式。

固定 k：

`k = {3, 5, 10, 20}`

所有 k 均报告，不根据哪个配置成绩最好来选择 k。

## 3.2 特征空间

主要分析必须复用 P46 的处理合同：

`u = (log1p(x) - mean_RealTrain) / max(std_RealTrain, 0.05)`

使用完整 4×712D 窗口，展平为2848维。

不得删除 Taxel、帧或通道。

注意：P46 中86.62%的标准化坐标触及0.05下限，因此必须明确说明该距离几何的局限。

另外允许构造一个固定的全局统计描述子作为敏感性对照，包括：

- 每帧均值和标准差。
- 每帧正值比例。
- 每帧中位数及高分位数。
- 相邻帧绝对变化量的统计。
- 窗口整体幅值分布。

该描述子只能作为辅助评价，不能取代完整2848D结果。

不得使用新训练的Encoder构造特征。

## 3.3 样本数量和独立性

主比较：

- Real Validation：780窗口。
- 迁移 Sim Validation：905窗口。

为了减少样本数量对邻域估计的影响，使用固定随机种子进行Root分层等量抽样，匹配为780窗口。

重复100次抽样，报告均值、中位数和离散程度。

100次抽样不能当作100次独立实验。

同时计算10个Root的分组结果，保留Root身份，不将窗口视为独立样本。

Real只有4个Validation Session，Bootstrap必须以Session为单位，明确报告不确定性限制。

## 3.4 Real–Real 自然基线

利用 P46 已经固定的32个Real Train Session及其四组划分，构造Session-disjoint Real–Real比较。

使用等量样本，保持每个Session的身份。

计算Real–Real的Precision、Recall、Density、Coverage，形成自然变异参考。

不得把Real–Real的点估计直接当成普适阈值。

不得使用Validation结果反向选择k或特征表示。

## 3.5 必须回答

1. A/B/C各自的Precision和Recall如何？
2. 是否存在Precision高但Recall低的模式坍塌？
3. Density和Coverage是否提供了额外信息？
4. Continuous-only的分布改善是否在多种k下稳定？
5. 原始Sim距离Real有多远？
6. 不同Root之间是否存在明显差异？
7. 结论是否依赖P46的标准化方式？

本阶段不允许根据PRDC单项结果宣布迁移成功。

---

# 四、P47-B：负对照与评价体系证伪

这是本轮最重要的部分。

不能只证明新指标对P44/P45有区分能力，还必须证明它能够识别明显不合格的生成方式。

构造以下负对照。

## 4.1 Constant Control

将所有Sim输入替换为同一个Real Train均值窗口。

输出形状保持：

`[N, 4, 712]`

该负对照不保留输入响应和个体差异。

预期检查：

- Precision是否可能偏高？
- Recall和Coverage是否下降？
- Density是否能够识别集中化？
- 新指标是否仍然误判其为合格？

## 4.2 Real Donor Control

从Real Train中随机抽取真实窗口，组成与Sim Validation等量的数据。

每个Sim输入对应一个独立随机Real donor。

它应该具有较好的Real边际分布，但不应该保留原始Sim输入的特定对应关系。

重点检验：

边际分布评价是否能够识别它与真正条件生成结果的区别。

## 4.3 Pair Shuffle Control

保留迁移输出的全部数值，只随机打乱Sim输入与迁移输出之间的配对关系。

优先在相同Root、Trial、Stage内部打乱；如果分组样本不足，明确记录实际使用的置换范围。

这是一个重要的数学对照：

打乱配对不改变生成输出的边际分布，因此PRDC、MMD、Energy等纯边际指标原则上不应发生变化。

如果出现明显变化，必须检查实现是否存在问题。

## 4.4 Temporal Shuffle Control

保持每个窗口的四帧数值不变，仅改变帧顺序。

用于检测评价体系是否能够识别时间顺序被破坏。

同时重新计算：

- lag-1 correlation。
- mean absolute frame delta。
- 每窗变化幅度。
- 各阶段的时间结构。

注意：Real采样间隔并非均匀20Hz，不得将相邻帧差分直接解释为每秒变化率。

## 4.5 负对照验收要求

逐项形成：

| 负对照 | 边际分布评价 | 配对依赖评价 | 时序评价 |
|---|---|---|---|
| Constant | 待验证 | 待验证 | 待验证 |
| Real Donor | 待验证 | 待验证 | 待验证 |
| Pair Shuffle | 应保持不变 | 应明显下降 | 应保持不变 |
| Temporal Shuffle | 实测 | 视具体指标而定 | 应能识别顺序变化 |

这里的预期是实验假设，不允许为了满足预期修改指标或筛选结果。

如果某项指标无法识别对应负对照，应明确判定该指标不具备预期的评价能力。

---

# 五、P47-C：输入与迁移输出的条件依赖评价

参考论文：

DeVries et al., 2019, *On the Evaluation of Conditional GANs*。

这篇论文提出FJD，用联合分布评价条件生成结果。

但是我们的Real与Sim目前没有经过认证的共同潜空间，也没有充分证实共同的物理条件标签。

因此：

**本阶段不能直接把两个独立Encoder的Latent拼起来计算FJD。**

也不能仅仅因为两个数据都有4×712维，就认为它们具有完全一致的物理坐标语义。

本阶段采用不要求两个域共享特征坐标系的配对依赖分析。

## 5.1 配对依赖

定义：

- X：原始Sim窗口。
- Y：对应的迁移Sim窗口。

即：

`X_i -> G(X_i) = Y_i`

使用HSIC（Hilbert-Schmidt Independence Criterion）分析X和Y之间的统计依赖。

要求：

1. X和Y分别构造自己的RBF核。
2. 两个域分别使用自己的距离带宽。
3. 使用固定的中位数带宽规则。
4. 报告0.5、1、2倍带宽敏感性。
5. 计算整体HSIC及Root内的条件依赖诊断。
6. 使用Root/Trial分组置换，不将905个窗口视为905个独立样本。

这里的HSIC只用于判断输入与输出之间是否保留统计依赖。

它不能证明输出具有正确的真实触觉语义。

## 5.2 配对对照

至少比较：

- 原始Sim与A输出。
- 原始Sim与B输出。
- 原始Sim与C输出。
- 原始Sim与Real Donor。
- 原始Sim与Constant。
- 原始Sim与Pair Shuffle。

重点回答：

1. 正常迁移结果是否保留输入依赖？
2. Constant是否被识别为缺乏输入依赖？
3. Real Donor是否被识别为缺乏输入依赖？
4. Pair Shuffle是否显著破坏输入输出关联？
5. 不同K是否改变输入输出依赖程度？
6. HSIC是否被Root之间的差异主导？

如果HSIC只反映Root身份而不能反映Root内部输入变化，不能将它作为条件一致性指标。

## 5.3 结论限制

不要将统计依赖直接等同于语义正确。

例如：

生成器可能保留了Sim输入的变化关系，但输出的接触位置仍然错误。

因此HSIC只能成为Conditional Consistency的一部分，不能单独作为最终验收依据。

---

# 六、P47-D：任务与阶段信息保持

目标：检验迁移是否抹掉Sim原本具有的任务、阶段和时序区别。

首先审计原始Sim Validation的元数据：

- Task。
- Stage。
- Root。
- Trial。
- 时间戳。
- 窗口对应关系。

在同一套固定数据上，对比原始Sim与迁移输出。

## 6.1 Task Information

以原始Sim的任务标签为条件，检查不同任务在迁移输出中是否仍然具有可区分的统计结构。

可以采用：

- 基于类别核的HSIC。
- 基于固定描述子的任务间距离。
- 各Task的均值、分位数和分布结构。
- Root-balanced任务间差异。

不得仅凭窗口数量获得虚假的统计显著性。

每个Task目前只有2个Validation Root，因此必须强调独立样本量限制。

## 6.2 Stage Information

如果Stage标签具有可靠的原始定义：

检查迁移前后不同Stage之间的统计差异是否保留。

特别关注：

- CLOSE。
- HOLD。
- LIFT。

但不能仅凭字符串相同就认定Real与Sim的Stage具有相同物理含义。

如果Stage语义未被认证，只能进行Sim内部的信息保持评价。

## 6.3 重要限制

Task/Stage信息保留，不等于已经完成Sim-to-Real语义对齐。

本阶段不能声称生成结果符合Real侧对应的操作任务，除非能够提供真实的共同标签和对应关系。

---

# 七、P47-E：FJD适用性与共同条件审计

这是一个单独的前提审计。

请建立Real和Sim条件变量对照表：

| 维度 | Real | Sim | 是否具有共同语义 |
|---|---|---|---|
| 任务 | 实际采集协议 | 五种操作任务 | 待审计 |
| 接触阶段 | 实际采集阶段 | CLOSE/HOLD/LIFT | 待审计 |
| 接触强度 | 实际触觉响应 | 仿真响应 | 待审计 |
| 时间 | 实际时间戳 | 仿真时间戳 | 待审计 |
| 传感器位置 | Real Taxel | Sim Taxel | 待审计 |

每一项必须提供数据来源、定义和证据。

只有同时满足以下要求，才能进入FJD实验：

1. 两域条件变量具有相同物理含义。
2. 条件标签或数值具有可靠的对应关系。
3. 输出特征具有经过验证的公共表示。
4. 不需要通过测试数据重新学习映射。

如果以上前提不成立：

- FJD标记为NOT_APPLICABLE。
- 不得强行拼接Latent。
- 不得拟合GMM。
- 不得求解OT。
- 不得人为构造Real/Sim标签对应关系。

报告必须说明缺少什么条件，以及未来如何补充。

---

# 八、综合结果与研究决策

最终需要形成一张总表：

| 方法 | Fidelity | Coverage | Input Dependence | Task/Stage Retention | 负对照识别 |
|---|---|---|---|---|---|
| Original Sim | | | | | |
| A_K1 | | | | | |
| A_K2 | | | | | |
| A_K4 | | | | | |
| A_K8 | | | | | |
| B_K1 | | | | | |
| C_K1 | | | | | |
| Constant | | | | | |
| Real Donor | | | | | |
| Pair Shuffle | | | | | |
| Temporal Shuffle | | | | | |

其他K配置必须完整报告，不能只展示表现最好的配置。

对每个指标说明：

- 定义。
- 数值范围。
- 统计单位。
- 不确定性。
- 是否通过负对照。
- 是否存在退化反例。
- 能够支持的结论。
- 不能支持的结论。

## 8.1 评价方法GO/NO-GO

P47不是生成模型验收，而是评价方法验收。

分别给出：

- PRDC是否有效。
- HSIC配对依赖是否有效。
- Task/Stage保持评价是否有效。
- FJD是否具有实施前提。
- 三类评价能否互相补充。
- 是否仍然存在能够同时骗过全部评价指标的负对照。

不能因为某个迁移配置的指标更好，就认定它是最终模型。

## 8.2 0.90目标的处理

特别注意：

本轮不能擅自修改原来的0.90定义。

历史C0的BA、AUC、D仍然作为独立历史基准保存。

P47探索的新评价体系与旧C0不能混为一谈。

本轮只回答：是否已经具备建立新0.90验收指标的统计基础？

如果具备，提出下一轮独立验证方案；如果不具备，说明缺少哪些数据或条件。

不得用当前10个开发Validation Root直接宣称总体90%通过率。

---

# 九、交付物

新建独立目录：

`outputs/tactile_p47_conditional_transfer_evaluation_2026-11-18/`

至少交付：

1. `REPORT.md`：完整实验报告。
2. `experiment_config.json`：所有固定参数和配置。
3. `input_inventory.csv`：输入文件、Hash、shape及来源。
4. `alignment_audit.csv`：输入输出配对审计。
5. `prdc_metrics.csv`：Precision/Recall/Density/Coverage。
6. `real_real_prdc.csv`：Real–Real自然基线。
7. `root_metrics.csv`：逐Root评价。
8. `negative_control_metrics.csv`：全部负对照。
9. `hsic_metrics.csv`：输入输出依赖。
10. `task_stage_retention.csv`：任务及阶段信息保持。
11. `condition_semantics_audit.csv`：FJD共同条件审计。
12. `bootstrap_results.csv`：分组不确定性。
13. `figures/`：必要的可视化。
14. `integrity.json`：数据保护、Hash及完整性检查。
15. `run_commands.json`：复现命令。

所有实验必须保存随机种子、代码Hash、输入Hash和参数。

---

# 十、最终需要回答的研究问题

报告最后必须直接回答：

1. 为什么P46的RVC会让Constant和Real Donor通过？
2. PRDC能否有效识别Constant和模式坍塌？
3. PRDC是否能区分Original Sim与迁移Sim？
4. Continuous-only的分布改善是否伴随着覆盖率变化？
5. HSIC能否区分正确配对和随机配对？
6. 迁移结果是否保留了Sim的Task/Stage信息？
7. 这些信息保持能否证明真实触觉语义正确？
8. 我们目前是否具备计算FJD的条件？
9. 新评价体系是否仍然存在明显的负对照漏洞？
10. 是否有足够依据重新定义0.90迁移验收标准？

**最终研究决策必须明确：**

A. 可以继续建立新的条件触觉迁移评价体系。

B. 部分指标有价值，但需要补充数据或改进统计设计。

C. 当前评价方法仍然无法区分合格迁移和退化生成，需要重新设计。

不要为了得到A而放宽指标。

**P47完成后停止，不自动开始P48，不进行新的生成器训练。等待人工审查实验结果后再决定下一步。**