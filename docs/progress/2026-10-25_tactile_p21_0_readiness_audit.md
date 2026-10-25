# P21-0：触觉 Sim-to-Real 实验可辨识性与数据就绪审计

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
Git HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`  
执行性质：只读决策审计；未训练模型、未运行实机、未修改真实数据、checkpoint、冻结划分、物理或Strict

## 1. 决策摘要

当前数据足以继续做**探索性的127D聚合统计域诊断**，但不足以支持可信的“域混淆达到0.90并可泛化”结论；真实任务信息保持和712D逐taxel空间迁移仍然阻塞。

三条路线的就绪状态：

| 路线 | 状态 | 当前允许的结论 |
|---|---|---|
| 127D聚合统计域混淆 | **CONDITIONALLY READY** | 可做一次有界、训练集拟合的诊断基线；不能作为真实触觉迁移完成证据 |
| 真实任务信息保持 | **BLOCKED** | stage可用于分层，不能替代接触、多指、对向或滑移真值 |
| 712D逐taxel空间迁移 | **BLOCKED** | 真实patch内部方向、指节顺序、电气通道和processed合同未确定 |

唯一推荐的下一项实验是：**逐patch、已知位置与接触条件的Real/Sim受控配对采集**。它同时解决空间映射、数值标定、任务真值和配置混杂，是当前最能提高实验可辨识性的操作。

如果短期无法取得硬件数据，可以运行一次有严格输出合同的127D统计映射基线，但必须标为探索性替代实验；无论结果如何，都不能据此声称712D迁移或真实任务保持成立。

## 2. 实际文件复核

本轮交叉读取了以下实际产物，而非仅引用历史报告：

- P18.2-B：`diagnostic_baseline/summary.json`、`results.json`、`split_manifest.json`、`dataset_inventory.json`、`task_preservation.json`；
- P18.2-C：`raw_domain_audit/summary.json`、`analysis/dataset_inventory.json`、`domain_results.json`、`signal_statistics.json`、`task_preservation.json`；
- P19：`dataset.npz`、`training_manifest.json`、`evaluation/domain_results.json`、`task_results.json`、`collapse_checks.json`；
- P20：`summary.json`、`feature_diagnostics.json`；
- 原始库清单：`outputs/tactile_sim2real_audit/realdata_inventory.json`。

关键hash保持为：

| 产物 | SHA-256 |
|---|---|
| Frozen split | `2e3d38eb055e9029f22fa75c0db2f0a81a95c413493b9f3c7dab9b996f70630d` |
| P19 dataset | `b6e1628da08a16d6a89906bddfa430c64122e066dc3e118f5f95f2217bd6877f` |
| P19 seed 0 | `f20daf401ec8a8879c0a3e0ef209cf8084f20c2fdd72d6c7bffafff9c3d002ee` |
| P19 seed 1 | `05bcbd67846d8bff0c13b49c8711010f1e43c4db5d6296ed32ceccfb67c3588c` |
| P19 seed 2 | `9c6aff3b559e29ea7fd91953fc1b26c306f016ed643c9a3baf94ff1d7d4f33f3` |

没有发现报告与实际JSON之间影响决策的冲突。

## 3. 从79个session到P19子集的数据漏斗

### 3.1 原始库

| 层级 | 数量 |
|---|---:|
| Session目录 | 79 |
| Trial目录 | 1,649 |
| 同时具备`samples.jsonl`、raw NPZ、processed NPZ的session | 77 |
| 同时具备三项资产的trial | 1,645 |

完整trial的stage覆盖：BASELINE 1,645、CLOSE 1,640、HOLD 1,062、LIFT 1,636、SLIP_INDUCE 581。数量看似充足，但trial不是独立采集主体；同一session内trial、帧和窗口高度相关，不能把1,645当成1,645个独立样本。

全库配置版本：

| 配置hash | Sessions | 完整资产sessions |
|---|---:|---:|
| `2c295d254fdd` | 52 | 52 |
| `d89a69f0904e` | 19 | 17 |
| `a585fe7c2bd8` | 8 | 8 |

### 3.2 P18冻结选择

P18.2-B冻结选择24个真实sessions，三个配置版本各8个；split为16 train、4 validation、4 test。仿真为8个独立roots，split为5/1/2。

该24-session清单是诊断抽样，不是全库正式benchmark。真实侧每session最多选2个trial，避免单session贡献过多。

### 3.3 P19实际子集

P19继续只取每个冻结session排序后的前2个trial，并要求存在processed NPZ及可形成8帧窗口的CLOSE/HOLD/LIFT数据。第三个配置版本`a585fe7c2bd8`的8个冻结sessions没有形成P19有效窗口，因此最终只剩：

| Domain | 独立groups | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Real session | 16 | 11 | 3 | **2** |
| Sim root | 8 | 5 | 1 | **2** |

窗口数：Real 184（124/34/26），Sim 504（315/63/126）。Real测试窗口按stage为CLOSE 16、HOLD 6、LIFT 4；Sim为30、30、66。

这解释了“79→24→16”：不是79个session普遍缺文件，而是冻结抽样、每session trial上限、stage/window筛选及第三配置版本未进入P19共同造成。

## 4. 采样时序与配置差异

真实数据降采样后的间隔仍不是严格20 Hz：

| 配置/stage | Median | P05–P95 |
|---|---:|---:|
| `2c295d254fdd` CLOSE | 57.1 ms | 55.8–75.4 ms |
| `2c295d254fdd` HOLD | 62.5 ms | 55.6–75.4 ms |
| `2c295d254fdd` LIFT | 65.5 ms | 55.9–75.5 ms |
| `d89a69f0904e` CLOSE | 57.0 ms | 55.6–75.0 ms |
| `d89a69f0904e` HOLD | 57.0 ms | 55.6–69.7 ms |
| `d89a69f0904e` LIFT | 57.0 ms | 54.7–67.1 ms |
| Sim各stage | 50.0 ms | 约50.0–50.0 ms |

因此abs-diff、自相关等时间特征包含采样协议差异。仅把帧数整理成相同窗口长度，并不能使时间统计物理可比。

## 5. 当前共享任务变量审计

| 变量 | Real | Sim | 可比性 |
|---|---|---|---|
| Stage | 人工/采集程序标签 | 轨迹阶段代码 | 可用于粗分层，不是接触真值 |
| 时间戳 | tactile/sample timestamp | simulation time | 已记录，但采样间隔不同 |
| 窗口内相对位置 | 可由时间戳/stage内索引恢复 | 可由frame/stage恢复 | 可构造，但动作持续时间未严格配对 |
| 动作类型 | arm/hand状态及command字段部分存在 | control、qpos/qvel/ctrl完整 | 语义与单位尚未统一 |
| 任一指接触 | 无逐帧真值 | MuJoCo contact真值 | 不可跨域验证 |
| 多指/拇指对向 | 无可靠真值 | 有 | 不可跨域验证 |
| 物体条件 | session/trial名称及配置中有线索 | Sugar Box明确 | 物体、材料、几何未配对 |
| SLIP | 人工stage，非逐帧滑移真值 | 当前P19无匹配stage | 不可比较 |
| Patch解剖名 | 图片表支持 | 代码确定 | 指级聚合可用 |
| 逐taxel方向 | 未知 | 确定 | 不可比较 |

P18.2-B中Real自身“BASELINE vs操作stage”可被probe识别，只证明人工stage携带信号变化，不等价于真实接触、多指或滑移标签。

## 6. 已有域可分性证据

无需重复训练即可得到一致结论：

- P18.2-B：Real processed vs Sim processed在robust/quantile后仍完全可分；
- P18.2-C：processed/processed及raw/raw均BA/AUC=1.000；
- P19：GAN后独立Logistic/MLP仍BA/AUC=1.000；
- P20：非零率族从BA 0.952降到0.610，说明GAN改变了部分统计；但幅值族BA=1.000、时间族BA=0.950；多个单region仍约0.88–0.98；
- P19生成输出还产生26%–30%负统计量和4%–7%比例越界，使probe获得新的非法域标志。

现有证据可以稳定回答“当前数据分布差异很强”，不能回答这些差异中有多少来自传感器域、processed算法、物体/动作、配置或采样协议。

## 7. 三类任务的就绪状态

### 7.1 127D聚合统计域混淆：CONDITIONALLY READY

已具备：方向不敏感特征、session/root划分、train-only normalization、两个probe、分组bootstrap和Sim任务真值。

限制：

- 测试仅2个Real sessions和2个Sim roots；
- 仅覆盖两个Real配置版本，第三版本完全缺席；
- 物体、动作、接触与时间未配对；
- Real processed合同未知；
- 无Real任务真值。

因此可以评价“某方法在当前冻结数据上的诊断性域可分性”，不能评价普适Sim-to-Real等效性。

### 7.2 真实任务信息保持：BLOCKED

缺少Real逐帧任一接触、多指、拇指对向、接触起点和滑移真值。stage不能替代这些标签。Sim probe高分只能证明Sim标签仍可从特征读取，不能证明真实任务信息被保持。

### 7.3 712D空间迁移：BLOCKED

阻塞项包括：每指两个4×8块的顺序、所有patch块内方向、电气通道顺序、mapping版本hash、raw→processed算法、标定和单位。当前不能构造可信逐taxel损失或空间一致性验收。

## 8. 域混淆度0.90能否被可信证明

当前可以计算点估计，但不能支持正式等效性结论。

`D_confusion>=0.90`相当于BA位于约0.45–0.55附近，但以下做法都不充分：

- 只观察BA接近0.5；
- 只看置信区间包含0.5；
- 因样本少而“未拒绝随机猜测”；
- 只用一个probe或一个seed；
- 把窗口数当作独立样本数。

正式验收应预注册等效区间，并要求：

1. 多个独立Real sessions和Sim roots的grouped CI整体落入等效区间，而非仅包含0.5；
2. Logistic、MLP及至少一个不同假设的probe同时通过；
3. 各配置版本和关键stage不出现系统性失败；
4. 输出合同100%合法；
5. Sim任务保持与Real真实标签保持同时通过。

当前Real test只有2个sessions。其bootstrap `[1,1]`只说明这两个session都被完全识别；若未来BA下降，两个groups也无法稳定估计跨session方差，更无法做跨配置等效性声明。

## 9. 两套互斥后续方案

### 方案A：能够取得受控真实数据——推荐且优先

#### 必须取得的材料

1. 实际使用的`config_mapping_zyhand.json`及hash；
2. `Usb32`、splitter、raw/processed保存和滤波代码版本；
3. calibration、auto-zero、gating、consistency实现与参数；
4. 16块安装图、接口编号、近远端定义及row/column方向；
5. 传感器型号、raw单位/量程、采样协议和session代码版本。

#### 最小配对采集协议

- 16个patch逐块验证；每块中心及四角；
- 标准化接触头、位置、法向方向和至少两个可复现载荷等级；
- 每点包含baseline→接触→hold→release，每条件至少3次重复；
- Real同步保存raw、processed、时间戳、接触事件/载荷和视频；
- Sim使用相同patch、位置、时序和接触条件，保存raw/processed及接触真值；
- session级冻结train/validation/test，不按帧拆分；
- 至少保留多个独立采集session作为test，并让每个配置版本在test中有多个独立groups。

#### 验收

- 映射/方向定位重复一致；
- 接触起点误差、响应单调性、重复性和空间质心可测；
- 输出无负非负统计、比例不越界、分位数有序、activation share和为1；
- 域混淆使用预注册等效区间及grouped CI；
- 同时验证Real/Sim接触、起点和多patch任务信息保持。

### 方案B：暂时无法采集——仅允许一次探索性替代实验

建立有约束的127D统计映射，不使用GAN：

- 非零率使用sigmoid或经验CDF映射，范围严格`[0,1]`；
- 非负幅值/差分使用log1p域的单调分位数映射，输出经softplus或显式非负参数化；
- P50/P90/P95/P99/max使用共享base与非负增量，保证分位数有序；
- autocorrelation使用tanh，范围`[-1,1]`；
- activation share使用softmax，范围`[0,1]`且总和为1；
- 所有参数仅在train sessions拟合；validation只选择预先列出的有限方案；test只评估一次；
- 不输出或声称712D tactile，不接入策略。

停止条件：任何非法输出；任一主要特征族probe仍明显高于随机；Sim任务probe明显退化；不同test groups方向不一致；或grouped CI不能落入预注册等效区间。即使成功，结论也只能是“当前127D聚合统计在当前数据上更难区分”，不能证明真实任务保持或空间迁移。

## 10. 最终决策

### 当前真实数据是否足以支持可信域混淆实验？

**不足以支持正式、可泛化的域等效结论。** 它足以暴露明显域差异和比较探索性127D统计基线，但独立test groups、任务真值、配对条件和空间合同均不足。

### 唯一推荐的下一项实验

**执行逐patch受控配对采集。** 这比继续在现有非配对数据上优化映射更能提高可辨识性，并且是Real任务保持和712D空间迁移共同依赖的前置条件。

### 无法采集时的最小替代

允许一次有严格输出合同的127D单调统计映射基线，用于确认P19失败是否主要来自非法输出和训练目标；其结论边界必须锁定在探索性聚合统计，不得升级为Sim-to-Real完成证据。

本阶段到此停止，未自动开始训练或实现新映射模型。
