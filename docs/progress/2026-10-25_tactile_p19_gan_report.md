# P19-GAN：基于现有真实触觉数据的 Sim-to-Real 域迁移可行性实验

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
基线 HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`  
结果目录：`outputs/tactile_p19_gan_2026-10-25/`  
执行边界：未修改真实数据、物理参数、Strict、生产触觉处理器或控制策略；未运行实机，未 commit/push

## 1. Executive Summary

P19 完成了第一次小规模、非配对 Sim-to-Real GAN 可行性实验，但结果明确为**未通过**。

- 冻结使用 P18.2-B/C 的 session/root split，SHA-256 为 `2e3d38eb055e9029f22fa75c0db2f0a81a95c413493b9f3c7dab9b996f70630d`；
- 输入不是未经验证排列的 712D taxel，而是 8 帧窗口提取的 127D、方向不敏感 patch 聚合统计；
- 数据包含 Real 184 个窗口、Sim 504 个窗口，覆盖 CLOSE/HOLD/LIFT；训练/验证/测试分别为 Real 124/34/26、Sim 315/63/126；
- 训练了 3 个随机种子，轻量残差生成器每个 49,279 参数，最多 400 epochs、patience 60；实际在 102–212 epochs 停止；
- GAN 输出经完全独立的 Logistic Regression 和浅层 MLP probe 评估，3 个生成器种子 × 3 个 probe 种子全部为 BA=1.000、ROC-AUC=1.000，95% 分组 bootstrap CI 均为 `[1.000, 1.000]`；
- `D_confusion=0`，没有达到预注册目标 `>=0.90`；CLOSE、HOLD、LIFT 各阶段也均为 BA=1.000；
- 仿真接触任务 probe 基本保持：任一接触 BA=1.000，多指接触 BA=0.9919，拇指对向接触 BA=0.9832–0.9874；但这不构成真实侧任务保持证据；
- 模型没有退化为常数或全零输出，但反归一化后的聚合特征有 26.18%–29.63% 为负值，具有明确 `[0,1]` 约束的比例特征中有 3.87%–7.31% 越界；
- 生成器只输出聚合特征，不能生成可供策略消费的 712D tactile frame，也不能评估逐帧接触起点。

因此，本轮证明的是：**在现有非配对、未完全对齐的数据上，简单聚合特征 GAN 既没有降低域可分性，也没有满足输出合同。** 当前不应扩大 GAN 容量或将其接入策略。下一项最有价值的实验仍是逐 patch、位置和阶段严格配对的受控接触采集。

## 2. 基线与数据合同

本轮复用了以下冻结产物：

- `outputs/tactile_p18_2_2026-10-25/mapping_update.json`；
- `outputs/tactile_p18_2_2026-10-25/diagnostic_baseline/split_manifest.json`；
- `outputs/tactile_p18_2_2026-10-25/raw_domain_audit/sim_dual/`；
- P18.2-C 的同步 Sim processed、stage、root 和接触真值。

没有发现与 P18.2-C 历史结论冲突的文件。P19 主路径固定为：

`Sim processed → 8-frame aggregate features → Generator → Real-normalized aggregate features`

真实目标域为 Real processed。由于真实 processed 算法和单位未知，本实验只评价统计域混淆，不声称物理量对齐。

### 2.1 冻结划分

| Domain | Train | Validation | Test | Group unit |
|---|---:|---:|---:|---|
| Real | 124 windows | 34 | 26 | session |
| Sim | 315 windows | 63 | 126 | root |

同一真实 session 的 trial 不跨 split；同一仿真 root 的三个 grasp modes 不跨 split。数据集文件 SHA-256 为 `b6e1628da08a16d6a89906bddfa430c64122e066dc3e118f5f95f2217bd6877f`。

样本单位是 8 个连续控制帧组成的窗口。窗口不在 train/test 之间随机拆分，分类和 bootstrap 均保留 session/root 分组。

### 2.2 特征与可验证范围

127D 特征复用 P18.2-B/C 的方向不敏感定义，包括：

- 每个指尖、同指两个指节的无序聚合、掌面共 11 个 region；
- 非零率、正值总量/均值、中位数、P90/P95/P99、最大值；
- 时间差分均值/P95、lag-1 自相关；
- 五指和掌面的相对激活份额。

它不依赖块内行列方向，也不需要判定每指两个 4×8 patch 的 `_0/_1` 顺序。代价是生成器不能输出 712D 空间触觉，因此本轮不能证明完整空间迁移。

## 3. GAN 架构与训练

### 3.1 架构

生成器为 127D 残差 MLP：

`x → Linear(127,128) → LeakyReLU → Linear(128,128) → LeakyReLU → Linear(128,127) → x + residual`

- 参数量：49,279；
- 判别器：`127→128→64→1`，LeakyReLU；
- 迁移方向：仅 Sim→Real-like；
- 没有使用 CycleGAN。当前不存在可信的 Real→Sim 物理语义和逐 taxel 对应，循环一致性可能把数据集伪特征固化为“内容”，不能自然解决本任务的合同缺口。

### 3.2 损失与超参数

生成器损失为：

`BCE(D(G(sim)), real) + 8 × content_loss`

content loss 只约束非零率和相对激活份额，避免直接对未知对应的 taxel 做 L1/L2。判别器使用标准二元 BCE。

| 项目 | 值 |
|---|---:|
| Seeds | 0, 1, 2 |
| Optimizer | Adam |
| Learning rate | 2e-4 |
| Betas | 0.5, 0.999 |
| Batch size | 64 |
| Epoch上限 | 400 |
| Early-stop patience | 60 |
| Content weight | 8.0 |
| Gradient clipping | 5.0 |

归一化由两个域各自的训练集 median/IQR 拟合；验证选择分数为生成/真实归一化特征的均值—方差差异加 content loss。测试集没有参与归一化、checkpoint 选择或 GAN 训练。

### 3.3 训练结果

| Seed | 完成epochs | 最佳验证选择分数 | Checkpoint SHA-256 |
|---:|---:|---:|---|
| 0 | 212 | 12.7856 | `f20daf401ec8a8879c0a3e0ef209cf8084f20c2fdd72d6c7bffafff9c3d002ee` |
| 1 | 102 | 12.8472 | `05bcbd67846d8bff0c13b49c8711010f1e43c4db5d6296ed32ceccfb67c3588c` |
| 2 | 147 | 12.4208 | `9c6aff3b559e29ea7fd91953fc1b26c306f016ed643c9a3baf94ff1d7d4f33f3` |

所有训练均在固定预算内早停，没有依据测试结果继续调参。

## 4. 独立域判别结果

GAN 自带判别器未用于验收。评估重新训练独立 Logistic Regression 和浅层 MLP，每类 probe 使用 seeds 11/12/13。

| 方法 | Probe | 测试 BA | ROC-AUC | Recall Sim/Real | D_confusion | BA 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Common robust | Logistic/MLP | 1.000 | 1.000 | 1.000/1.000 | 0.000 | [1.000,1.000] |
| Domain robust | Logistic/MLP | 1.000 | 1.000 | 1.000/1.000 | 0.000 | [1.000,1.000] |
| Domain quantile | Logistic/MLP | 1.000 | 1.000 | 1.000/1.000 | 0.000 | [1.000,1.000] |
| GAN，3 seeds | Logistic/MLP | **1.000** | **1.000** | **1.000/1.000** | **0.000** | **[1.000,1.000]** |

GAN 测试混淆矩阵在每个组合中均为 `[[126,0],[0,26]]`。CLOSE、HOLD、LIFT 分层也全部为 BA/AUC=1.000。

结论：GAN 没有带来任何可测域混淆改善，离 `D_confusion>=0.90` 的目标很远。由于 Real test 只有 26 个窗口、LIFT 仅 4 个 Real 测试窗口，分阶段样本量仍有限；但在当前数据上结果不是“置信区间过宽”，而是稳定的完全可分。

## 5. 接触信息保持

使用完全 held-out 的 Sim roots 和原接触真值，对迁移后特征训练独立线性任务 probe：

| 输入 | 任一接触 BA | 多指接触 BA | 拇指对向 BA |
|---|---:|---:|---:|
| Robust/quantile baselines | 1.000 | 0.9919 | 0.9874 |
| GAN seed 0 | 1.000 | 0.9919 | 0.9832–0.9874 |
| GAN seed 1 | 1.000 | 0.9919 | 0.9832–0.9874 |
| GAN seed 2 | 1.000 | 0.9919 | 0.9832–0.9874 |

这些结果表明聚合特征中仍能线性读出三类仿真窗口标签，但不能证明：

- 真实侧接触、多指或拇指对向信息被保持；真实数据没有相应真值；
- 单个 taxel 的空间结构被保持；模型没有生成 712D；
- 接触起始时刻被保持；8 帧聚合生成器没有逐帧输出；
- HOLD/LIFT 的物理稳定性被保持；probe 是推断指标，不是执行指标；
- SLIP 信息被保持；现有两域没有可靠匹配标签。

因此任务信息只能判为“仿真聚合标签可读性未明显下降”，不能判为跨域任务保持通过。

## 6. 坍塌与输出合同检查

| Seed | Finite | 近常数特征比例 | 方差/Real | 负物理特征比例 | 比例特征越界 |
|---:|---|---:|---:|---:|---:|
| 0 | 是 | 0 | 0.306 | 28.22% | 7.31% |
| 1 | 是 | 0 | 0.300 | 29.63% | 3.87% |
| 2 | 是 | 0 | 0.322 | 26.18% | 6.10% |

生成器没有全零、NaN/Inf 或完全常数坍塌，但出现另一类不可接受问题：反归一化后本应非负的统计量大量为负，且非零率/activation share 等有界量越界。输出物理特征范围约为 -35.14 至 33.80。

这说明当前无约束 MLP 没有学习合法统计流形。即便域 probe 将来下降，也必须先加入输出合同约束并验证不会以破坏接触信息换取混淆；本轮没有为了修饰结果而 clip 输出。

## 7. 稳定性与局限

- 三个 GAN seeds 的域结果完全一致，失败不是单一随机种子的偶然现象；
- 三个阶段全部完全可分，未发现只在某一阶段改善；
- 独立 test groups 严格隔离，但真实/仿真任务、动作、物体、采样抖动和 processing 仍不配对；
- 真实配置版本差异仍可能被特征利用；
- 两个 4×8 patch 的顺序和块内方向仍未验证；
- 聚合 GAN 无法回答完整 712D 空间域迁移是否可行；
- Real 没有逐帧任务真值，无法完成最终任务保持验收。

## 8. 工程产物与验证

新增代码：

- `apps/train_tactile_sim2real_gan.py`；
- `apps/evaluate_tactile_sim2real_gan.py`；
- `tests/test_tactile_sim2real_gan.py`。

产物：

- 数据与划分证明：`outputs/tactile_p19_gan_2026-10-25/dataset.npz`、`training_manifest.json`；
- 三组 checkpoint 与曲线：`seed_0/`、`seed_1/`、`seed_2/`；
- 独立域评估：`evaluation/domain_results.json`；
- 任务保持：`evaluation/task_results.json`；
- 坍塌/合法性检查：`evaluation/collapse_checks.json`；
- 汇总与图：`evaluation/summary.json`、`evaluation/training_and_domain_results.png`。

工程验证：定向 pytest 2 passed；相关 Ruff 全部通过。真实数据目录未写入。

## 9. 最终结论

### 是否达到 90% 域混淆度？

**没有。** GAN 的 `D_confusion=0`，目标为至少 0.90；独立 Logistic 与 MLP 仍为 BA/AUC=1.000。

### 是否保留足够接触信息？

**只能确认仿真聚合标签仍可被 probe 读取，不能确认跨域任务保持。** 同时输出存在大量非法负统计量和比例越界，因此整体不合格。

### 是否值得进入下一阶段真实—仿真任务迁移？

**当前不值得。** 该模型不是 712D 迁移器，域混淆失败，输出合同也未通过。扩大容量、切换 CycleGAN 或接入控制策略都缺少证据基础。

### 下一项唯一最值得执行的实验

进行**逐 patch、已知接触位置和阶段的受控 Real/Sim 配对采集**：确定两个 4×8 patch 的近远端关系与块内方向；同步保存 raw/processed；每个位置至少 3 次重复；提供真实接触/非接触和接触起点标签。先在该小数据上验证公共标定、robust/quantile 以及合法输出约束。只有简单方法仍失败且任务标签保持可验证时，才重新设计 712D 条件生成模型。

本阶段到此停止，未自动启动下一轮训练。
