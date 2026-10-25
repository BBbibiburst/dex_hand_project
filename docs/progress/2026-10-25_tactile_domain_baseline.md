# P18.2-B：真实触觉与仿真触觉诊断性域判别基线

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
基线 HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`  
输出：`outputs/tactile_p18_2_2026-10-25/diagnostic_baseline/`  
执行边界：未修改 `outputs/realdata/`、物理参数、Strict、控制模型或 checkpoint；未训练 GAN/控制策略，未运行实机，未 commit/push

## 1. Executive Summary

本轮完成了一个方向不敏感、按 session/root 隔离的诊断性域判别基线。结果不是“域已对齐”，而是相反：**在当前 patch 聚合表示下，真实与仿真仍然极易区分。**

- 使用24个真实 sessions（3种触觉采集配置各8个）和24条仿真 episodes（8个独立 roots、每root 3种候选模式）；
- 冻结划分为真实16/4/4 sessions、仿真5/1/2 roots；同一session、trial、episode及相关root没有跨split；
- 比较 BASELINE、CLOSE、HOLD、LIFT 四个操作阶段；
- 每个 trial/episode-stage 形成一个127维、块内方向不敏感的统计样本；数据清单共326个“信号路径—阶段”单元；
- `Real processed vs Sim processed`：公共robust尺度下，Logistic与MLP的独立测试 balanced accuracy 均为1.000，ROC-AUC为1.000，`D_confusion=0`；
- 按域分别做robust scaling后，线性模型BA仍为0.958（3 seeds范围0.938–0.969），MLP仍为1.000；
- 按域分别做训练集分位数映射后，线性和MLP仍为1.000；
- `Real raw vs Sim processed`在所有组合中均为BA 1.000。该轨道不是raw-to-raw，只证明未知单位的真实raw与保存的仿真processed完全不可比；
- 四个阶段单独评估均可被公共robust线性probe完全区分，并非仅由BASELINE/CLOSE比例造成；
- stage-label保持诊断中，Real processed和Sim processed的`BASELINE`对`CLOSE/HOLD/LIFT`均为BA 1.000。它证明这些预处理没有在当前小样本上抹掉粗粒度阶段信息，但不等于保持了真实接触、多指对向或滑移信息。

因此当前不应启动GAN。优先级应是：保存真正的仿真raw 712D、完成逐patch方向验证，并构造物体/动作/接触状态更严格匹配的数据；同时保留简单patch级robust calibration作为后续方法的必要基线。

## 2. 数据来源与可比性

### 2.1 真实数据

从 `outputs/realdata/` 中选择同时具有：

- `samples.jsonl`；
- `tactile_raw_split.npz`；
- `tactile_processed.npz`；
- 至少一个目标stage的完整trial。

为避免让单一采集版本代表全部真实域，按与触觉处理有关的字段生成配置指纹：同步频率、sensor class/shape、mapping路径、calibration/auto-zero/zero-on-baseline、gated filter和16项consistency。选择了三种主要合同，每种8个sessions，共24个sessions。

配置指纹及数量：

| 配置指纹 | Sessions | 含义 |
|---|---:|---|
| `2c295d254fdd` | 8 | 较新60 Hz、auto-zero类合同 |
| `d89a69f0904e` | 8 | 中期合同 |
| `a585fe7c2bd8` | 8 | 较早100 Hz/gated类合同 |

每session最多读取2个trial，每stage最多均匀抽取64帧。没有通过重复trial或复制帧补样本。

### 2.2 仿真数据

使用P15 Sugar Box三种候选模式的24条完整episode：

`outputs/dp_p15_2026-10-24/candidates/mode_*/ycb_004_sugar_box/p*/candidate_00/episode.npz`

它们覆盖8个独立roots，每root三种相关模式。划分以root为单位，不以episode为单位，避免同一初态的不同模式泄漏。

仿真保存的是经过 `TactileSignalProcessor` 的712D `tactile`。episode没有保存MuJoCo处理前的712D原始touch向量。因此本轮只能建立：

1. `real_processed_vs_sim_processed`：名称相同但算法/单位合同不同；
2. `real_raw_vs_sim_processed`：刻意保留的非同构诊断，**不是raw-to-raw**。

### 2.3 Stage映射限制

| Real | Sim | 本轮处理 |
|---|---|---|
| BASELINE | settle code 0 | 仅操作层面的无动作基线，不声称物理状态完全等价 |
| CLOSE | close code 4 | 纳入 |
| HOLD | hold code 5 | 纳入 |
| LIFT | lift code 6 | 纳入 |
| SLIP_INDUCE | 无语义等价stage | 排除 |

真实数据的物体、手部闭合命令和采样率仍与仿真不同。因此该数据集是“分阶段域诊断”，不是严格配对benchmark。

## 3. 冻结划分

| 域 | Train | Validation | Test | 分组单位 |
|---|---:|---:|---:|---|
| Real | 16 sessions | 4 sessions | 4 sessions | session |
| Sim | 5 roots / 15 episodes | 1 root / 3 episodes | 2 roots / 6 episodes | root |

结构化划分：`diagnostic_baseline/split_manifest.json`  
SHA-256：`2e3d38eb055e9029f22fa75c0db2f0a81a95c413493b9f3c7dab9b996f70630d`

最终清单为326个单元，包含raw与processed两条真实路径。每条域判别轨道的独立测试集包含24个仿真stage单元和16个真实stage单元。置信区间以session/root为组做1000次bootstrap；由于仿真测试仅2个roots，区间只能视为探索性证据。

## 4. 方向不敏感的特征

没有构造未经验证的逐taxel重排。每根手指使用：

- 7×8指尖候选块；
- 两个4×8指节块合并后的无序集合；
- 手掌单独统计。

共11个region。每个region计算：

- 非零比例；
- 正值总量和正值均值；
- median、P90、P95、P99、max；
- region总响应一阶差分的mean/P95；
- lag-1 autocorrelation。

再加入五指与手掌相对激活份额，共127维。块内taxel任意排列不会改变这些特征；单元测试显式验证了这一性质。

## 5. 预处理与模型

所有统计只在train split拟合：

1. `common_robust`：两域共享的训练集median/IQR。它只为数值条件服务，保留域间幅值差；
2. `domain_robust`：Real/Sim分别拟合median/IQR；
3. `domain_quantile`：Real/Sim分别拟合101点经验分位数并映射到标准正态分位数。

后两者在线应用时需要预先知道domain，因此只是oracle式诊断，不是可部署域迁移器。

判别器：

- Logistic Regression：127→1线性probe；
- 浅层MLP：127→64→32→1。

训练seed为0、1、2；同一冻结split、early stopping和类别权重。没有安装或引入新依赖。

## 6. 域判别结果

### 6.1 Real processed vs Sim processed

下表为3 seeds均值；range为balanced accuracy的seed范围。bootstrap CI列给seed 0的grouped 95%区间。

| Preprocess | Probe | BA | Seed range | ROC-AUC | D_confusion | BA 95% CI |
|---|---|---:|---:|---:|---:|---:|
| common robust | Logistic | 1.000 | 1.000–1.000 | 1.000 | 0.000 | [1.000,1.000] |
| common robust | MLP | 1.000 | 1.000–1.000 | 1.000 | 0.000 | [1.000,1.000] |
| domain robust | Logistic | 0.958 | 0.938–0.969 | 1.000 | 0.083 | [0.812,1.000] |
| domain robust | MLP | 1.000 | 1.000–1.000 | 1.000 | 0.000 | [1.000,1.000] |
| domain quantile | Logistic | 1.000 | 1.000–1.000 | 1.000 | 0.000 | [1.000,1.000] |
| domain quantile | MLP | 1.000 | 1.000–1.000 | 1.000 | 0.000 | [1.000,1.000] |

公共robust Logistic的测试混淆矩阵为：

| Actual / Predicted | Sim | Real |
|---|---:|---:|
| Sim | 24 | 0 |
| Real | 0 | 16 |

按stage分别计算时，BASELINE、CLOSE、HOLD、LIFT均为BA 1.000；每个stage测试混淆矩阵为`[[6,0],[0,4]]`。测试集中出现的两个真实配置版本，Real recall均为1.000。第三种配置没有产生本轮测试单元，只出现在train/validation，因此不能声称三种配置都经过独立测试。

### 6.2 Real raw vs Sim processed

所有3种预处理、2种probe、3个seeds的BA和ROC-AUC均为1.000，`D_confusion=0`。该结果主要确认单位、基线和时序合同完全不同，不能用于评价任何正式对齐方法优劣。

### 6.3 可区分特征

公共robust线性probe的绝对权重显示，差异不是单一总幅值：

- processed轨道：palm P95、ring/palm相对激活份额、index-tip P90、多块lag-1 autocorrelation和middle-tip差分；
- raw轨道：多个指尖/指节的lag-1 autocorrelation、median/P90/P95以及差分P95。

这些权重只能作为probe解释，不是因果重要性。它们说明后续需要同时匹配空间激活比例和时间响应，单一全局缩放不足。

完整逐seed结果、两类recall、confusion matrices、stage和配置分层结果见：

- `diagnostic_baseline/results.json`
- `diagnostic_baseline/results.csv`
- `diagnostic_baseline/preprocessing/`

## 7. 域混淆指标解释

按预注册公式：

`D_confusion = max(0, 1 - 2 * abs(BA - 0.5))`

当前processed轨道最好看的线性结果是domain robust的`D_confusion=0.083`，而非接近1。其余多数配置为0。

这不表示触觉任务一定无法迁移，只表示在当前分阶段统计表示和小规模独立测试组上，域差异远大于随机水平。反过来，即使未来`D_confusion`升高，也必须同时检查任务信息、多个probe和grouped CI，不能单独验收。

## 8. 任务信息保持

真实数据没有逐taxel接触真值，也没有可证明的多指接触或拇指对向标签。因此本轮只定义一个弱标签任务：

- negative：BASELINE；
- positive：CLOSE、HOLD或LIFT。

这实际测量“粗粒度操作阶段可分性”，不是接触检测准确率。

| Track | Preprocess | Real BA | Sim BA |
|---|---|---:|---:|
| processed/processed | common robust | 1.000 | 1.000 |
| processed/processed | domain robust | 1.000 | 1.000 |
| processed/processed | domain quantile | 1.000 | 1.000 |
| real raw/sim processed | common robust | 0.958 | 1.000 |
| real raw/sim processed | domain robust | 1.000 | 1.000 |
| real raw/sim processed | domain quantile | 0.958 | 1.000 |

因此没有观察到粗粒度stage信息被消除。但因为stage与动作时序、采集配置和域高度相关，这不能证明多指接触、接触起点或滑移信息得到保持。以下指标按停止条件标记为“不可计算”：

- 真实接触/非接触真值；
- 接触起始时间偏差；
- 多指接触识别；
- 拇指对向接触识别；
- 跨域SLIP_INDUCE保持（仿真无等价stage）。

完整结果：`diagnostic_baseline/task_preservation.json`。

## 9. 失败/受限实验

| 项目 | 状态 | 原因 |
|---|---|---|
| Raw-to-raw域判别 | 未执行 | 仿真episode没有保存处理前712D raw |
| SLIP_INDUCE跨域对照 | 未执行 | 仿真无语义等价stage |
| 712D spatial discriminator | 未执行 | segment顺序和块内方向未验证 |
| 接触/对向/滑移任务保持 | 停止 | 真实逐taxel接触真值缺失 |
| 配置版本完整独立对照 | 不充分 | 测试仅4个real sessions，只有两个版本产生有效单元 |
| 统计等效性 | 未成立 | 域probe远高于随机，无需声称等效 |

## 10. 工程验证与交付

新增：

- `apps/run_tactile_domain_baseline.py`
- `tests/test_tactile_domain_baseline.py`
- 本报告及独立输出目录。

验证：

- 定向pytest：4 passed；
- 相关Ruff：All checks passed；
- split manifest、预处理统计、全部逐seed结果均成功写入；
- 没有修改 `outputs/realdata/`。

运行命令：

```bash
/home/zmy/miniconda3/envs/mujoco/bin/python apps/run_tactile_domain_baseline.py
```

主要产物：

- `diagnostic_baseline/split_manifest.json`
- `diagnostic_baseline/dataset_inventory.json`
- `diagnostic_baseline/results.json`
- `diagnostic_baseline/results.csv`
- `diagnostic_baseline/task_preservation.json`
- `diagnostic_baseline/summary.json`
- `diagnostic_baseline/domain_and_task_summary.png`

## 11. 下一阶段建议

### 是否需要立即使用GAN？

**不需要，也不具备可信条件。** 当前probe完全可分首先暴露的是单位、处理算法、采样时序和任务分布不匹配。直接训练GAN很可能学会配置/阶段转换或抹除接触信号，而不是物理域迁移。

### 简单标定是否值得做？

**值得保留为必要基线，但当前证据表明它不够。** 按域robust scaling只能把线性BA从1.000降到0.958；MLP仍为1.000。后续任何生成式方法必须超过这一简单基线，同时保持任务信息。

### 下一项最小实验

优先做两个合同补全动作，而不是扩大模型：

1. 在仿真采集链路中同时保存处理前和处理后的712D tactile，建立真正的raw→processed成对基线；
2. 完成P18.2-A提出的逐patch受控按压，冻结4×8 segment排列和块内方向。

随后采集或筛选物体、动作阶段和接触强度更加匹配的小规模Real/Sim数据，重新运行本脚本。若简单patch calibration后域probe仍显著可分、且可靠接触任务指标保持，再评估GAN或其他生成式方法。

本阶段到此停止，未启动生成式迁移训练。
