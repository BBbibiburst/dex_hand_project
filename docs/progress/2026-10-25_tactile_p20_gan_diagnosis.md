# P20：P19-GAN 失败归因与域差异诊断

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
基线 HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`  
结果目录：`outputs/tactile_p20_gan_diagnosis_2026-10-25/`  
执行性质：只读取 P18/P19 数据和三个既有 checkpoint，未调用训练入口、未训练或修改任何模型

## 1. Executive Summary

P19 失败不是因为生成器未更新、checkpoint 未加载或独立 probe 误用了 GAN 判别器。三个生成器都显著改变了输入，但改变方向没有落到 Real 合法统计流形，反而制造了新的、非常容易识别的域标志。

证据如下：

1. 三个生成器的归一化残差 L2 均值达到输入 L2 均值的 1.13–1.32 倍；输入与物理域生成特征的逐维中位相关仅 0.25–0.38。生成器不是恒等映射。
2. 训练中判别器 loss 从约 1.39–1.49 降至 0.64–0.74，生成器 loss 从约 0.88–0.90 上升至 1.67–1.98。GAN 对抗在后期明显由判别器占优。
3. checkpoint 选择使用“验证集均值/方差距离 + content loss”，而不是冻结外部域 probe。该选择分数从约20降到12，但独立 test probe 仍为 BA/AUC=1.000，说明选择代理与最终目标不一致。
4. GAN 输出中 26.18%–29.63% 的聚合统计为负，3.87%–7.31% 的有界比例特征越界。Seed 0 的 `index_segments.median` 被生成为均值 -0.731，而 Real 与原 Sim 均为0；该单特征即可实现 AUC=1.000 的域区分。
5. 即使按特征族单独训练诊断 probe，GAN 后的幅值族 BA=1.000，时间族 BA=0.9496。多个单独 region 仍达到约0.88–0.98。可分性不是只来自一个异常值。
6. 非零率族从原 BA=0.9524 降至0.6099，activation share 的线性 probe也不再稳定区分，说明 GAN 确实对少数统计族产生了混淆；但幅值、尾部和时间统计仍完整暴露域标签。
7. 没有检测到 session/root split 泄漏。独立 Logistic/MLP probe 重新初始化并仅用 train/validation 拟合；test 未用于归一化、checkpoint 选择或 probe 训练。BA、AUC、混淆矩阵和 `D_confusion` 公式实现正确。
8. 测试 Real 只包含两个独立 sessions，且恰好各来自一个配置版本；这足以确认当前冻结测试集完全可分，但不足以可靠分离“配置版本效应”和“真实—仿真传感器效应”。

根因排序为：**无约束生成输出离开合法统计流形与目标/选择代理不充分**（高置信度）→ **现有域差异在幅值、尾部和时间统计上过强且数据未配对**（高置信度）→ **真实配置、stage与采样条件混杂**（中等置信度）。没有证据支持“训练根本没发生”或“评价泄漏”作为原因。

## 2. P19 链路核实

### 2.1 模型与 checkpoint

生成器为 `127→128→128→127` 残差 MLP，共49,279参数；判别器为 `127→128→64→1`。三个 checkpoint 均能严格加载，feature names、hidden size、normalizer和seed完整。

| Seed | Checkpoint SHA-256 | 训练epochs | 最佳epoch | 报告使用一致 |
|---:|---|---:|---:|---|
| 0 | `f20daf401ec8a8879c0a3e0ef209cf8084f20c2fdd72d6c7bffafff9c3d002ee` | 212 | 152 | 是 |
| 1 | `05bcbd67846d8bff0c13b49c8711010f1e43c4db5d6296ed32ceccfb67c3588c` | 102 | 42 | 是 |
| 2 | `9c6aff3b559e29ea7fd91953fc1b26c306f016ed643c9a3baf94ff1d7d4f33f3` | 147 | 87 | 是 |

checkpoint 保存的是验证选择分数最低时的生成器，而 training curve 持续记录到 early stopping。P19 报告中的最佳分数与曲线最小值一致。

### 2.2 损失与梯度路径

代码中的梯度路径有效：`fake = generator(sim_batch)`在生成器步骤未 detach，随后通过冻结参数状态但未冻结计算图的 discriminator 计算 BCE，`g_loss.backward()`可回传至 generator。content loss也直接依赖fake。

当前日志没有保存逐层 gradient norm，也没有保存 discriminator checkpoint，因此不能事后逐步复算每个 epoch 的梯度；但非零且很大的生成残差、训练损失变化和三个不同checkpoint足以排除“生成器完全未更新”。

| Seed | 初始 D/G loss | 最佳epoch D/G loss | 最终 D/G loss |
|---:|---:|---:|---:|
| 0 | 1.494/0.884 | 0.743/1.860 | 0.712/1.978 |
| 1 | 1.400/0.905 | 1.110/1.095 | 0.724/1.665 |
| 2 | 1.388/0.875 | 0.851/1.608 | 0.639/1.882 |

判别器 loss 持续降低、生成器 loss 上升，符合判别器重新分离假样本而生成器未达到平衡的表现。

### 2.3 Normalization与checkpoint选择

Sim 与 Real 各自使用仅由训练集拟合的median/IQR，transform再clip至±20。训练集被clip的比例为 Real 2.75%、Sim 0.31%；测试集为 Real 1.06%、Sim 0.31%。clip存在但不是大多数样本的主要信息损失。

Sim normalizer 有47.24%的特征因训练IQR过小而回退到scale=1；Real为10.24%。这反映Sim大量稀疏/近常数特征，也使两域归一化坐标具有不同的数值分辨率。

验证选择分数主要由mean/variance距离主导：最佳epoch的distribution项约12.18–12.67，`8×content`仅约0.18–0.27。content不是数值上的首要主导项，但它固定非零率与activation share，使这些维度不能自由匹配真实域；更重要的问题是选择指标不约束特征合法性，也不直接衡量held-out域混淆。

### 2.4 独立评价协议

- Logistic/MLP不是GAN内部判别器，而是在生成结果冻结后重新训练；
- probe train/validation/test仍按原session/root split；
- 同一Real session和Sim root只出现在一个split；
- P19 dataset中的16个Real sessions是冻结manifest中24个sessions的有效子集，并非额外重划分；
- 所有8个Sim roots与manifest一致；
- test未进入GAN normalizer、checkpoint选择或probe训练；
- `D_confusion=max(0,1-2*abs(BA-0.5))`、AUC和混淆矩阵复核正确。

未发现实现错误、标签反转、checkpoint错配或split泄漏。

## 3. 生成器改变了什么

### 3.1 总体改变量

| Seed | 残差L2/输入L2 | 逐维中位相关 | 负特征比例 | 有界特征越界 |
|---:|---:|---:|---:|---:|
| 0 | 1.265 | 0.250 | 28.22% | 7.31% |
| 1 | 1.128 | 0.376 | 29.63% | 3.87% |
| 2 | 1.318 | 0.356 | 26.18% | 6.10% |

三组结果一致说明：生成器大幅改写了分布，但没有学会合法的Real-like统计约束。它不是“主要保持输入不变”，也不是全零/常数坍塌。

### 3.2 哪些差异被改善

以seed 0进行固定诊断：

| 特征族 | 原Sim vs Real BA | GAN vs Real BA | 解释 |
|---|---:|---:|---|
| nonzero ratio | 0.9524 | 0.6099 | 明显接近混淆，但未稳定达到随机 |
| activation share | 0.8053 | 0.3877 | 线性方向发生反转/失稳，不应单独视为成功 |
| activation amplitude | 0.9808 | **1.0000** | 没有改善，成为最强识别来源 |
| temporal | 0.9881 | **0.9496** | 仅小幅改善，仍高度可分 |

GAN主要改变了非零率和手指间相对激活，但没有同时匹配幅值尾部与时间统计。

### 3.3 最明显的残留与新生域标志

- `index_segments.median`：原Sim和Real测试均为0，GAN均值为-0.731，单特征AUC=1.000；这是纯粹由非法负值制造的新域标志。
- `palm.nonzero`：Sim 0.010、GAN 0.055、Real 0.023，GAN越过目标分布，单特征AUC=1.000。
- `index_tip.positive_sum`：Sim 1.096、GAN 3.654、Real 0.009，改变方向远离Real，AUC=0.937。
- `middle_segments.max`：Sim接近0、GAN -0.636、Real 2.267，负值违反统计定义，AUC=0.921。
- 多个region的lag-1 autocorrelation被推到负值或大于合理相关范围，成为时间域标志。

因此完全可分并非“GAN完全没改变”，而是“改变后仍保留强差异，同时新增非法特征”。

## 4. Probe识别依据

### 4.1 Region消融

仅使用一个region的11维统计，GAN后测试BA如下：

| Region | BA |
|---|---:|
| ring tip | 0.9841 |
| thumb segments | 0.9570 |
| index tip | 0.9444 |
| middle segments | 0.9371 |
| index segments | 0.8822 |
| palm | 0.8770 |
| ring segments | 0.8770 |
| little segments | 0.8550 |

域标志分散在多个手指和掌面，不是单个坏patch即可解释。

### 4.2 Stage分层

GAN后每个阶段仍有多个单特征AUC=1.000：

- CLOSE：Real 16、Sim 30个测试窗口；主要包括指节median、指尖幅值/尾部；
- HOLD：Real 6、Sim 30；主要包括index tip sum、指节median、时间差分和palm nonzero；
- LIFT：Real 4、Sim 66；主要包括指尖sum/p99、指节median、时间相关和palm nonzero。

HOLD/LIFT的Real独立组样本很少，因此不能比较各阶段根因强弱；但足以确认当前三阶段都没有实现混淆。

### 4.3 配置版本

P19有效Real数据包含两个配置版本，各8个sessions；test中每版本只有1个session。配置版本没有与域标签完全混杂——两个版本都只属于Real，但这本身意味着任何真实配置共同具有、而Sim没有的采集特征都可成为域标志。

由于每版本test仅1个session，不能进行可靠的版本内bootstrap，也不能把BA=1唯一归因于某个配置版本。当前证据只支持“配置与采样差异是未控制混杂”，不支持其为唯一根因。

## 5. 根因证据排序

### 1. 输出合同无约束、生成结果离开合法统计流形——高置信度

- 证据：26%–30%特征为负，4%–7%有界比例越界；若干非法单特征AUC=1；多个seed一致。
- 影响：外部probe获得比原始域差异更简单的新判别规则。
- 反证：无NaN/Inf、无常数坍塌，说明是“离流形”而非普通数值崩溃。
- 未解：加入严格输出参数化后能否降低域差异，当前未训练验证。

### 2. 对抗训练与checkpoint代理未覆盖最终验收目标——高置信度

- 证据：D loss降低而G loss升高；mean/variance选择分数改善，但外部probe完全不变；没有合法性项和独立域混淆项。
- 影响：最佳checkpoint只在有限矩统计上较好，不保证完整分布或合法统计匹配。
- 反证：生成器确实更新，不能归为代码未执行。
- 未解：日志未保存梯度范数和训练中独立probe，无法精确定位首次失衡epoch。

### 3. 幅值、尾部与时间域差异过强——高置信度

- 证据：幅值族单独BA=1，时间族BA=0.9496，多region独立可分；P18 raw/processed两路也均完全可分。
- 影响：只对非零率/activation share施加content约束不能覆盖主要差异。
- 未解：差异中传感器物理、真实processing、任务动作和采样抖动各占多少，现有非配对数据不能因果拆分。

### 4. 域独立归一化与稀疏特征退化——中高置信度

- 证据：47.24%的Sim特征IQR退化到scale=1，Real仅10.24%；部分样本被±20 clip。
- 影响：同名特征在两个归一化坐标中具有不同分辨率，残差MLP很难学习统一合法映射。
- 反证：即使domain quantile基线也完全可分，归一化不是唯一原因。

### 5. 配置、stage、动作与采样混杂——中等置信度

- 证据：Real test每配置仅1 session；Real/Sim物体、动作与采样时序未配对；三个stage样本明显不平衡。
- 影响：probe可能识别数据采集协议，而非纯触觉材料/传感器差异。
- 未解：没有受控配对数据，无法量化各混杂贡献。

### 6. 评价实现或数据泄漏——低可能性

- 支持该假设的证据：未发现。
- 反对证据：root/session隔离成立、test不参与拟合、probe独立训练、指标复算一致、两个probe和三个seed结果一致。

## 6. 下一步候选与推荐

### 推荐：受控逐patch配对采集 + 有约束统计映射

先对同一解剖patch、同一已知接触位置/阶段完成Real/Sim配对，提供真实接触起点与重复试验。用该数据建立非负、有界、单调或分位数约束的统计映射，并以独立session/root及任务标签验收。

推荐理由：它同时减少数据混杂、补齐输出合法性和真实任务真值，能够判断是否真的需要生成模型。当前非配对GAN无法区分这些因素。

### 候选二：只在现有数据上实现有约束的统计基线

为nonnegative、ratio、autocorrelation分别使用softplus、sigmoid、tanh参数化；按feature family而非统一MLP映射。它能修复输出合同，但不能解决任务/配置不匹配，因此只适合作为配对采集前的工程基线。

### 候选三：空间映射确认后再研究712D条件生成

逐patch方向和4×8顺序确认、真实processed合同明确、配对集建立后，再考虑patch条件生成器。现阶段切换CycleGAN或扩大127D MLP不具备证据基础。

## 7. 工程与完整性验证

新增：

- `apps/diagnose_tactile_p19_gan.py`；
- `tests/test_tactile_p19_gan_diagnosis.py`；
- `outputs/tactile_p20_gan_diagnosis_2026-10-25/summary.json`；
- `outputs/tactile_p20_gan_diagnosis_2026-10-25/feature_diagnostics.json`；
- `outputs/tactile_p20_gan_diagnosis_2026-10-25/p19_failure_diagnosis.png`；
- 本报告。

验证结果：定向pytest 1 passed；相关Ruff通过。诊断脚本只加载既有dataset/checkpoint并进行纯推理和新的诊断probe拟合，没有调用P19训练入口。

| 受保护产物 | P20前SHA-256 | P20后SHA-256 | 结果 |
|---|---|---|---|
| Seed 0 checkpoint | `f20daf...02ee` | `f20daf...02ee` | 未改变 |
| Seed 1 checkpoint | `05bcbd...588c` | `05bcbd...588c` | 未改变 |
| Seed 2 checkpoint | `9c6aff...33f3` | `9c6aff...33f3` | 未改变 |
| Frozen split | `2e3d38...0630d` | `2e3d38...0630d` | 未改变 |

`outputs/realdata/`仅被只读加载，没有写入、移动或删除。未commit、未push、未清理工作区。

## 8. 最终回答

1. **最可能原因**：生成器在不充分的分布代理下学到离开合法统计流形的变换，同时幅值/尾部/时间域差异仍很强。
2. **GAN是否改变输入**：是，而且改变很大；残差约为输入范数的1.13–1.32倍，并非恒等映射。
3. **probe为何仍能区分**：非法负值/越界提供新标志，幅值族仍BA=1，时间族仍约0.95，多个region独立可分。
4. **实现或评价错误**：未发现checkpoint错配、split泄漏、probe复用或指标公式错误；训练目标和选择代理存在设计不足。
5. **合法性与任务信息**：输出合同失败；仿真聚合接触标签仍可读，但不能外推到真实侧或712D空间保持。
6. **最推荐实验**：逐patch受控Real/Sim配对采集，并先验证有约束统计映射；它比继续扩GAN更能消除当前不可辨识的混杂。

P20到此停止，未新增训练或扩大分析范围。
