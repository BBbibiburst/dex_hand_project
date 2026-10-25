# P22：基于现有数据的127D触觉域混淆优化

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
基线 HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`  
输出：`outputs/tactile_p22_2026-10-25/`  
执行边界：未修改真实数据、冻结划分、旧checkpoint、物理、Strict或生产策略；未运行实机，未commit/push

## 1. 最终结论

P22实际执行了三条预定路线和一个有限的训练集经验传输补充，共38个验证候选。最终结果为：**验证集达到预注册门槛，但冻结测试集没有达到Logistic与MLP均`D_confusion >= 0.90`的最终标准。**

- 验证集锁定候选：stage条件、任务排序的训练Real经验传输，`thumb_weight=0.5`、`nonzero_weight=1.0`；
- 验证域混淆：Logistic 0.9155、MLP 0.9463、kNN 0.9659；
- 验证任务BA最低0.9828，输出合同零违规；
- 冻结测试集只执行一次。Logistic三个seeds的域混淆为0.7900、0.8913、0.5232；MLP为0.6703、0.4090、0.4170；kNN为0.9560；
- 测试任务保持通过：任一接触0.9960、多指0.9879、拇指对向0.9874；相对原始最大下降0.0041；
- 117个不同输出行，无全零/常数坍塌，所有非负、比例、自相关、分位数顺序及activation-share约束通过；
- 测试域结果对probe seed和仅2个Real sessions/2个Sim roots高度敏感，因此不能声称达到≥0.90。

本轮没有依据测试结果继续调参。成功门槛未通过，应如实记为**探索性验证改善、最终验收失败**。

## 2. 数据与冻结合同

直接复用P19数据：Real 184个窗口、Sim 504个窗口，127D方向不敏感聚合特征，覆盖CLOSE/HOLD/LIFT。划分仍按Real session和Sim root隔离，测试只有2个Real sessions和2个Sim roots。

| 受保护产物 | SHA-256 |
|---|---|
| P19 dataset | `b6e1628da08a16d6a89906bddfa430c64122e066dc3e118f5f95f2217bd6877f` |
| Frozen split | `2e3d38eb055e9029f22fa75c0db2f0a81a95c413493b9f3c7dab9b996f70630d` |
| P19 seed 0 | `f20daf401ec8a8879c0a3e0ef209cf8084f20c2fdd72d6c7bffafff9c3d002ee` |
| P19 seed 1 | `05bcbd67846d8bff0c13b49c8711010f1e43c4db5d6296ed32ceccfb67c3588c` |
| P19 seed 2 | `9c6aff3b559e29ea7fd91953fc1b26c306f016ed643c9a3baf94ff1d7d4f33f3` |

训练、归一化和候选选择未读取测试统计。只有验证候选同时满足域混淆、任务保持和合同合法性后，才执行一次final模式。

## 3. 路线A：条件统计传输

实现位于`apps/run_tactile_constrained_mapping.py`。每个stage独立拟合训练Sim→Real单调分位数映射，并比较带收缩的协方差对齐。共运行5个covariance strengths × 3个shrinkage，共15个候选。

最佳候选为covariance strength 0.75、shrinkage 0.9：

| Validation probe | BA | AUC | D_confusion |
|---|---:|---:|---:|
| Logistic | 0.9265 | 0.9921 | 0.1471 |
| MLP | 0.9265 | 0.9981 | 0.1471 |
| kNN | 0.8235 | 0.8697 | 0.3529 |

任务BA为1.0且合同合法，但域差异仍明显，未进入测试。

## 4. 路线B/C：受约束可学习映射与联合目标

实现位于`apps/train_tactile_constrained_mapper.py`。输出按特征类型参数化：非负量、`[0,1]`比例、`[-1,1]`自相关、softmax activation shares及有序分位数。共运行5个固定候选，包括无对抗Route B和域判别/任务联合Route C。

最佳候选仍被Logistic、MLP和kNN以BA/AUC 1.0区分，`D_confusion=0`；任务最低BA 0.9344，也未通过0.95下限。训练中stage moment项量级远高于其他损失，独立判别器没有得到混淆。该路线未进入测试。

## 5. 有限经验传输候选

为检验现有数据下“合法分布支持”本身能否改善混淆，新增`apps/run_tactile_empirical_transport.py`：

1. 只用Sim train训练三个任务score probe；
2. 按stage计算score分位；
3. 只从对应stage的Real train合法127D原型中选取输出；
4. 用训练Sim推断score排序Real原型，不在transform时读取任务标签；
5. 有限比较6个thumb weights × 3个nonzero weights，共18个候选。

该方法不是712D触觉生成器，也不提供逐taxel映射。它是训练集经验分布上的统计传输，输出天然满足127D特征合同。

锁定候选验证结果：

| Probe | BA | AUC | D_confusion |
|---|---:|---:|---:|
| Logistic | 0.4577 | 0.4953 | 0.9155 |
| MLP | 0.5268 | 0.6359 | 0.9463 |
| kNN | 0.4830 | 0.6863 | 0.9659 |

任务BA为contact 0.9919、multi-finger 0.9918、thumb opposition 0.9828，满足绝对和相对门槛。因此在查看测试前锁定。

## 6. 冻结测试结果

### 6.1 域混淆

| Probe | Seed | BA | ROC-AUC | D_confusion | Grouped BA 95% CI |
|---|---:|---:|---:|---:|---:|
| Logistic | 11 | 0.6050 | 0.6343 | 0.7900 | [0.4821, 0.7421] |
| Logistic | 12 | 0.5543 | 0.6459 | 0.8913 | [0.3204, 0.7520] |
| Logistic | 13 | 0.7384 | 0.7747 | 0.5232 | [0.5972, 0.8550] |
| MLP | 11 | 0.6648 | 0.7527 | 0.6703 | [0.5615, 0.7775] |
| MLP | 12 | 0.7955 | 0.7915 | 0.4090 | [0.6576, 0.8810] |
| MLP | 13 | 0.7915 | 0.8382 | 0.4170 | [0.6617, 0.8770] |
| kNN | — | 0.5220 | 0.7476 | 0.9560 | [0.4603, 0.6022] |

Logistic和MLP均未满足“全部seed的BA位于[0.45,0.55]”。AUC也明显偏离0.5，不能用部分BA接近随机来宣称通过。Grouped CI很宽，反映测试独立组过少，而不是等效性证据。

### 6.2 分阶段

- CLOSE：部分Logistic/MLP达到0.9混淆，但不同seed范围约0.779–1.000，kNN为0.783；
- HOLD：D范围约0.367–1.000；Real测试仅6个窗口，波动最大；
- LIFT：D范围约0.583–1.000；Real测试仅4个窗口；
- 三个阶段都没有在所有预注册probe/seeds上稳定通过。

### 6.3 任务保持与输出合同

| Sim test任务 | 原始BA | 映射后BA | 下降 |
|---|---:|---:|---:|
| 任一接触 | 1.0000 | 0.9960 | 0.0040 |
| 多指接触 | 0.9919 | 0.9879 | 0.0040 |
| 拇指对向 | 0.9874 | 0.9874 | 0.0000 |

三项均高于0.95且下降小于0.03。输出合同全部零违规，finite通过，共117个不同输出行。没有通过常数、全零、clip测试标签或标签反转获取混淆。

## 7. 验收判断与剩余可优化项

| 条件 | 结果 |
|---|---|
| Logistic和MLP测试`D_confusion >= 0.90` | **失败** |
| 三项任务BA≥0.95且下降≤0.03 | 通过 |
| 输出合法且无明显坍塌 | 通过 |
| CLOSE/HOLD/LIFT稳定通过 | **失败** |
| 总体验收 | **失败** |

实际证据表明，经验传输可以在验证集把边际支持和任务排序同时拉近，但只使用训练Real原型无法覆盖两个held-out Real sessions的联合结构。剩余优化项是：增加不读取测试的跨训练session留一法目标、对Real原型做平滑而非离散复制、在训练组内显式惩罚probe-seed方差，以及校正真实/仿真的时间采样差异。但根据预注册纪律，本轮不能依据最终测试表现继续挑选这些方案。

## 8. 工程产物

新增：

- `apps/run_tactile_constrained_mapping.py`；
- `apps/train_tactile_constrained_mapper.py`；
- `apps/run_tactile_empirical_transport.py`；
- `apps/summarize_tactile_p22.py`；
- `tests/test_tactile_constrained_mapping.py`；
- `tests/test_tactile_constrained_mapper.py`；
- `tests/test_tactile_empirical_transport.py`。

关键结果：

- `outputs/tactile_p22_2026-10-25/summary.json`；
- `route_a_candidates.json`、`route_bc_candidates.json`、`empirical_transport_candidates.json`；
- `empirical_transport_validation.json`；
- `empirical_final_test.json`；
- `empirical_mapped_features.npz`；
- `p22_final_results.png`。

定向pytest：3 passed；相关Ruff：通过。旧dataset、split和P19三个checkpoint的SHA-256均与P22前一致。

## 9. 最终回答

P22没有实现最终冻结测试上的`D_confusion >= 0.90`。最好的验证候选确实达标、任务保持与合法性也通过，但其测试结果不能跨probe seed稳定泛化。当前可保留的成果是一个经过实际验证、合同合法、明显优于P19无约束GAN的127D探索性统计传输基线；不能将其描述为已完成触觉Sim-to-Real，更不能描述为712D空间迁移。

本阶段到此停止，未按测试结果继续调参。
