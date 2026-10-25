# P18-Tactile：触觉 Sim-to-Real 域迁移基础设施只读审计

日期：2026-10-25  
项目：RM75B + 六驱动欠驱动 Dex Hand  
Git HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`（`main`）  
审计边界：仅只读检查与轻量统计；未训练模型、未运行实机、未修改源数据/代码/checkpoint，未 commit/push

## 1. Executive Summary

- `CONFIRMED`：`outputs/realdata/` 不是空目录，而是 79 个 session、1649 个 trial、224809 个文件、34.16 GB 的真实采集资产。共有 968754 条 `samples.jsonl` 记录，1645 个 trial 有 processed tactile，1646 个有 raw tactile。
- `CONFIRMED`：真实触觉以 16 个二维块保存：5×`7×8`、10×`4×8`、1×`7×16`，合计恰为 712；仿真 Dex Hand 也为相同块形状集合和 712 taxel。
- `BLOCKED`：真实配置引用 `Usb_API_stable_Tujian_/spliter/config_mapping_zyhand.json`，但该映射文件及采集实现均不在数据目录或仓库。现在不能证明真实块 `00…15` 分别对应哪根手指/节段/掌面，也不能证明块内方向与仿真一致。
- `CONFIRMED`：仿真运行时为 `simple_box`，每个 taxel 是附着于手部曲面网格的 MuJoCo `mjSENS_TOUCH` 标量。默认运行配置对原值执行 deadzone、饱和、幂律、低通和邻域串扰，最后归一化到 `[0,1]`。
- `MISSING`：真实 processed 算法、标定文件、物理单位、传感器产品型号/量程和 mapping 文件均缺失。配置值 `sensor_class: Usb32` 只能确认软件类名，不能据此认定产品型号或物理测量单位。
- `BLOCKED`：仓库没有 GAN/CycleGAN、真实数据 loader、域级 split、独立域判别评测或迁移推理接口。现阶段不具备可信 GAN 训练条件。
- `INFERRED`：若找回 mapping 和处理算法，两边很可能可映射到公共 16-patch/712-channel 表示；在此之前兼容性属于方案 **D：缺少必要拓扑/标定，无法可靠映射**，不能因维度相同直接训练 GAN。
- `CONFIRMED`：项目指标不能用 GAN 自带判别器“把 95% 迁移样本判成真实”验收；该结果可能由全判真实、类别失衡或弱判别器产生。正式验收必须使用未参与 GAN 的独立 probe classifier、按 session/trial 分组测试、平衡准确率/AUC/置信区间和任务信息保持指标。

结论：**先补通道拓扑与真实处理合同，再做 Raw Sim 和简单标定基线；当前不应训练 GAN。**

## 2. 真实数据资产盘点

### 2.1 总体规模

| 项目 | 实际结果 |
|---|---:|
| session | 79 |
| trial目录 | 1649 |
| 文件 | 224809 |
| 总大小 | 34,163,917,385 bytes |
| `samples.jsonl` | 1647 files / 968754 rows |
| processed NPZ | 1645 |
| raw NPZ | 1646 |
| JPG / PNG | 211818 / 789 |
| JSONL / JSON / YAML | 6522 / 2231 / 158 |

完整逐文件相对路径、格式、大小和可行范围内的 SHA-256 位于 `outputs/tactile_sim2real_audit/realdata_inventory.json`（约37 MB）。全部3291个NPZ已计算SHA-256，未发现完全相同的NPZ；≤20 KiB元数据有48组完全相同文件，主要说明配置/小元数据复用，不能据此认定 trial 数据重复。大型图像未全量哈希。

### 2.2 数据内容

`CONFIRMED`：每个完整 trial 通常包含：

- `tactile_raw_split.npz`：按帧、按16块保存的原始阵列；
- `tactile_processed.npz`：同一块结构的处理后阵列；
- `samples.jsonl`：sample/tactile时间戳、stage、6轴手状态、7轴机械臂状态、相机索引，较新schema还有sync ID和周期耗时；
- `commands.jsonl`、`stage_events.jsonl`：Close、Lift、Hold、Slip Induce、Release等命令及阶段边界；
- `trial_meta.json`、session manifest/config；
- 部分阶段的彩色/深度图像。

对象标签覆盖16个可见名称；任务标签计数为 stable 900、slip 749、hardness 165（标签可重叠）。主要阶段记录数包括 BASELINE 187180、CLOSE 187241、HOLD_AIR 184845、PUT_DOWN 137131、RECOVER 90089、LIFT 51245、SLIP_INDUCE 44692。

`CONFIRMED`：967102个相邻有效时间间隔的中位数为15.40 ms，P05/P95为8.93/29.26 ms。配置历史并不统一：存在60/100 Hz触觉/同步设置和30/60 Hz相机设置，因此不能把全库视为单一采样率。约961757条记录含sync ID，6997条旧记录不含；1852条早期记录的arm状态标为fake。

### 2.3 真实触觉结构与质量抽样

`CONFIRMED`：16块的形状集合为：

- 5块 `7×8`；
- 10块 `4×8`；
- 1块 `7×16`；
- 总通道数 `5×56 + 10×32 + 112 = 712`。

对77个含完整数据的session各抽一个trial、每trial抽首/中/末帧后的统计：

| 数据 | 抽样值数 | min | max | mean | std | 零值比例 | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 164472 | 0 | 19806 | 240.63 | 675.75 | 42.56% | 0 |
| processed | 164472 | 0 | 9.8967 | 0.00875 | 0.1572 | 86.49% | 0 |

`INFERRED`：高零比例与阶段性无接触、裁剪区域及门控处理都可能有关，当前不能仅凭统计判为坏点。processed最大值大于1，证明它不能直接按仿真归一化 `[0,1]` 解释。是否存在饱和、漂移和永久坏点需要在恢复处理算法和基线标定后按通道/阶段分析。

`CONFIRMED`：6个trial存在局部缺失，包括两个仅有元数据的早期trial，以及若干缺 processed/raw/commands 的trial；详见inventory的 `missing_trial_assets`。这些trial不能直接进入训练/测试。

### 2.4 可确认与未知信息

| 信息 | 状态 | 证据/限制 |
|---|---|---|
| 连续时间序列、trial/session ID | `CONFIRMED` | timestamps、manifest、trial目录 |
| 接触前后/动作阶段 | `CONFIRMED` | BASELINE/CLOSE/HOLD/LIFT/SLIP_INDUCE等 |
| 手和臂状态 | `CONFIRMED` | samples/commands；早期少量fake arm需排除 |
| 真实通道数 | `CONFIRMED` | split NPZ为712值/帧 |
| 软件传感器类 | `CONFIRMED` | `Usb32` |
| 产品型号、材料、量程 | `MISSING` | 无README/硬件说明 |
| 通道到手部解剖映射 | `BLOCKED` | mapping文件缺失 |
| processed算法和物理单位 | `BLOCKED` | 外部采集代码/标定缺失 |
| 压力/法向力/剪切力物理标定 | `MISSING` | 不能由数值范围推断 |
| 明确接触真值 | `MISSING` | stage是动作协议，不等价于逐taxel接触真值 |

## 3. 仿真触觉实际生成链路

### 3.1 调用路径

`CONFIRMED`：

`configs/robot_profiles/rm75b_dex_hand.json`  
→ `source/envs/rl_env.py::RobotGymEnv`  
→ Dex Hand descriptor的factory  
→ `source/sensors/tactile/dex_hand.py::SimpleBoxTactileSensor`  
→ 每个patch网格点创建box site及`mjSENS_TOUCH`  
→ `data.sensordata[address]`  
→ `TactileSignalProcessor.process`  
→ observation `tactile`  
→ grasp episode/LeRobot recorder  
→ DP dataset flatten读取。

### 3.2 空间顺序和语义

`CONFIRMED`：仿真顺序可完整恢复。`DEX_HAND_PATCH_LAYOUT`按 finger 0…4 排列，每指依次 `4×8 segment`、`4×8 segment`、`7×8 fingertip`，最后是 `7×16 palm`；patch内按row-major展平。metadata保留patch名称、形状和flat slice。

`CONFIRMED`：原始值是MuJoCo touch sensor对对应site接触的标量响应，是仿真接触代理信号；不是已证明与真实压阻阵列同单位的压力或剪切力。项目未为每个样本保存独立“物理单位”字段。

默认处理配置：deadzone 0.002、saturation 1.0、exponent 0.7、low-pass alpha 0.45、8邻域crosstalk 0.12、noise 0、normalize=true。由此最终观测为`float32[712]`、范围合同 `[0,1]`；raw可通过sensor接口读取，但标准LeRobot只记录处理后值。

现有P15成功轨迹的轻量抽样为336×712：min 0、max 0.9223、mean 0.00412、zero fraction 93.49%。仿真控制周期为50 ms（20 Hz）。

`CONFIRMED`：已验证episode可同时关联object ID、seed、qpos/qvel/ctrl/action、阶段、接触统计和时间步；现有流程能在不改物理参数的情况下继续生成固定schema仿真触觉。

潜在问题：

- `CONFIRMED`：标准数据只存processed，不存同帧raw，因此难以重新选择处理参数；
- `CONFIRMED`：低通滤波器有跨时间内部状态，episode reset时清零；必须按episode处理，不能随机打乱帧后重建；
- `BLOCKED`：真实端块顺序/方向不明，不能直接对齐仿真flat index；
- `CONFIRMED`：真实约60–100 Hz、仿真策略20 Hz，时序训练前必须以带抗混叠的明确规则重采样，并保留事件边界。

## 4. Sim/Real兼容性矩阵

| 项目 | 仿真 | 真实 | 结论 |
|---|---|---|---|
| 样本结构 | 单个712D向量/20 Hz episode | 16块/约60–100 Hz trial | 可转换但需重采样 |
| 通道数 | 712 | 712 | 数量一致 |
| 块形状 | 10×4×8、5×7×8、1×7×16 | 相同shape multiset | 结构强吻合 |
| 通道语义 | patch/row/col完全可恢复 | key 00…15；mapping缺失 | `BLOCKED` |
| 单位 | MuJoCo touch代理量，经归一化 | raw/processed单位未知 | 不兼容/未知 |
| 范围 | processed `[0,1]` | sampled processed `[0,9.8967]` | 明显不同 |
| 归一化 | 代码明确 | 采集算法缺失、配置版本不同 | `BLOCKED` |
| 空间拓扑 | mesh曲面patch | 可能同布局，但无映射证据 | `BLOCKED` |
| 时间序列 | 完整、20 Hz | 完整但多配置频率 | 可分组重采样 |
| 接触语义 | 可与MuJoCo contact关联 | stage/命令是弱标签 | 只能弱比较 |

当前分类为 **D**。维度和shape集合高度暗示未来可能进入 **A/B**，但在mapping、块方向和标定恢复前不能直接reshape后训练。

## 5. GAN基础设施与方法选择

### 5.1 仓库成熟度

`MISSING`：仓库没有GAN/CycleGAN/条件GAN实现，没有real tactile dataset loader，没有域级train/val/test split，没有独立域probe、checkpoint/恢复、推理适配器或信号保持评测。现有`DiffusionPolicy`的tactile encoder不是域迁移模型。

因此“现有代码足以开始GAN训练”的答案是 **否**。

### 5.2 方法优先级

| 方法 | 前提 | 当前适用性 | 风险 |
|---|---|---|---|
| Raw Sim | 公共表示和固定split | mapping后必须先做 | 域差异大 |
| 分通道/分patch线性或分位数标定 | 拓扑对应、接触阶段可比 | 首选基线 | 可能破坏时序/相关性 |
| 低维公共表示 | 可确认finger/palm级映射 | mapping不完整时较稳健 | 丢失局部taxel信息 |
| 无配对CycleGAN | 两域大量trial、公共拓扑、cycle/identity约束 | 当前不具备 | 模式坍塌、接触语义改变 |
| 单向GAN | 强条件标签或配对监督 | 当前最不适合 | 映射不受约束、容易伪真 |

`INFERRED`：数据没有逐帧配对不排除无配对迁移，但必须先有相同语义空间、可比较阶段和任务保持评价。当前先做简单方法，只有其独立域probe仍明显优于随机且任务保持可靠时，才有理由训练GAN。

## 6. “域混淆度≥90%”的正式定义

### 6.1 不能采用的定义

GAN训练判别器loss、训练集错误率或“95%生成样本被判真实”均不能作为正式指标。后者不测真实样本召回，生成器全输出同一模式或判别器全预测real也可得高分。

### 6.2 建议主指标

使用与GAN完全独立、从未参与生成器训练的域probe，在按真实session/trial和仿真episode分组的held-out测试集上计算：

`D_confusion = clip(1 - 2 * abs(BA - 0.5), 0, 1)`。

90%混淆等价于测试BA位于`[0.45, 0.55]`。但点估计不够，正式通过需同时满足：

1. 至少3种合理容量probe（线性、浅层MLP、1D时序CNN/GRU）均未发现可利用域信息；
2. 5个随机种子，类别/阶段平衡；
3. 以trial/session为cluster bootstrap，`D_confusion`的双侧95%置信区间下界≥0.90，等价地BA等效区间完整落入预注册margin；
4. ROC-AUC及其cluster bootstrap CI也处于预注册随机等效区间（建议0.45–0.55）；
5. 混淆矩阵中两域召回均非退化，拒绝“全判一类”；
6. 用TOST/等效性检验而非普通“不显著”检验。无法拒绝BA=0.5的差异检验不等于证明等价；
7. 与Raw Sim和线性标定做同split对照，并报告样本数、trial数和统计功效。

probe超参数只可用validation sessions选择；test sessions一次性解封。相邻帧、同一trial或同一session不得跨split。

## 7. 任务信息保持

域不可分不等于可用。至少同时验收：

- contact/non-contact：用阶段弱标签，并在一小批人工/实验标定真值上复核；
- contact onset：迁移前后时间偏移及抖动；
- finger/palm patch激活：mapping恢复后比较；
- 每patch峰值、均值、方差、零比例、动态范围和自相关；
- 通道相关矩阵/空间平滑度；
- 多指接触趋势及真实阶段一致性；
- 若未来有物理标定，再评估压力/强度；当前不能声称物理保真；
- 下游接口应接收固定712D、明确范围和20 Hz时序，并记录迁移前/后两个版本，不能覆盖raw。

应增加identity/cycle或变化预算约束，并明确禁止改变接触起始、把non-contact生成成contact、交换finger区域或抹平峰值。没有真实逐taxel接触标签时，空间物理保真只能标为未验证。

## 8. 最小可执行路线与门槛

### Stage 0：恢复真实合同（当前唯一下一实验）

- 找回`config_mapping_zyhand.json`、采集/processed实现、标定说明和硬件规格；
- 选少量完整trial做逐指/掌面控制接触，验证16块对应关系、行列方向和时间同步；
- 门槛：712通道映射100%可追溯，至少每个finger/palm激活区域符合预期；否则停止。

### Stage 1：冻结公共表示与split

- 排除不完整和fake-arm trial；按session/对象/任务/stage分层，session级train/val/test隔离；
- 仿真按episode隔离；只用train统计归一化；
- 统一到20 Hz时必须先低通再降采样；保留原始时间戳和原文件hash；
- 门槛：无group泄漏，stage/object覆盖明确。

### Stage 2：Raw Sim域probe

- 输入：公共712D及短时窗；输出：domain label；
- 建立线性/MLP/时序probe、5 seeds、固定test；
- 产物：BA/AUC/confusion matrix/cluster CI和任务保持基线。

### Stage 3：Linear Calibration

- 只用train trials做每patch稳健中心尺度或分位数映射；
- 同一test和probe协议；
- 若已达到90%且任务保持通过，则GAN没有必要。

### Stage 4：条件性GAN实验

- 仅在简单方法不足、mapping与任务标签可用时启动；
- 优先按patch/阶段条件的时序映射，保留identity、cycle/内容和onset约束；
- 固定预算、多个seed、独立probe；GAN训练判别器不参与验收。

### Stage 5：下游验证

- Raw/Linear/GAN三组在相同触觉任务和最终抓取策略接口比较；
- 域混淆和任务保持必须同时通过，任何一项失败都不能声称完成Sim-to-Real。

## 9. 十个问题的结论

1. **realdata里是什么？** `CONFIRMED`：79 sessions、1649 trials、约96.9万时序样本，含raw/processed 712D触觉、阶段、手臂状态和大量图像；少数trial不完整。
2. **仿真如何生成？** `CONFIRMED`：712个曲面box touch sites → MuJoCo TOUCH → 明确的deadzone/saturation/nonlinear/lowpass/crosstalk → `[0,1]` observation。
3. **两域兼容吗？** `BLOCKED`：维数和patch shapes一致，但通道语义、方向、单位、processed算法和频率不一致或未知。
4. **能开始GAN吗？** `BLOCKED`：不能；mapping、real loader/split、probe和保持指标均未具备。
5. **最小缺失项？** 真实channel mapping和processed/calibration合同；这是首要硬阻塞。
6. **GAN是否首选？** 否。应先Raw与分patch线性/分位数标定；GAN仅在简单方法不足后进入。
7. **90%如何定义？** 独立test probe的`D_confusion`加等效性CI、AUC、混淆矩阵、多probe/多seed；不能用GAN判别器训练结果。
8. **如何防偏置/泄漏/失真？** session/trial/episode分组；test封存；类别/阶段平衡；多容量probe；cluster CI；onset/patch/相关性/任务指标联合门槛。
9. **最小实验几步？** 合同恢复→公共表示/split→Raw probe→Linear→条件性GAN→下游任务，共6个有门槛阶段。
10. **唯一最值得做的下一实验？** 恢复mapping后，用少量逐指/掌面受控接触验证真实712通道的解剖位置和块方向；未通过前不训练任何GAN。

## 10. 检查范围与命令类型

关键文件：

- `source/sensors/tactile/dex_hand.py`
- `source/sensors/tactile/signal_processing.py`
- `source/sensors/tactile/fitting/layout.py`
- `source/envs/rl_env.py`
- `configs/robot_profiles/rm75b_dex_hand.json`
- `source/data/lerobot_recorder.py`
- `source/imitation/dataset.py`
- `source/imitation/diffusion_policy.py`
- `outputs/realdata/*/session_config.yaml`及trial资产

执行了Git只读状态检查、文件/目录统计、JSONL流式统计、NPZ结构抽样、全部NPZ SHA-256重复检查、配置与代码调用追踪。没有运行pytest/Ruff，因为本轮没有源代码修改；没有加载全部34 GB数组到内存，没有训练、生成仿真数据或运行硬件。

本轮仅新增本报告、`outputs/tactile_sim2real_audit/summary.json`和机械生成的`realdata_inventory.json`。原始`outputs/realdata/`未写入、移动、重命名或删除。
