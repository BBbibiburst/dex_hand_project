# P6/P8 模型容量与过度抬升审计

日期：2026-10-14。本文为只读审计；没有修改模型定义、checkpoint、训练数据或控制参数。

## 1. 审计方法

所有数字均通过实际 checkpoint 加载对应类后，遍历 `named_parameters()` 获得。`total` 与 `trainable` 不包含 normalization buffer；文件大小为磁盘实际字节数。层数采用“直接持有参数的 leaf module”口径，同时列出结构深度，避免把 SiLU、Flatten 等无参数算子误报为模型层。

实际加载路径：

- XYZ9 BC：`source/imitation/action_chunk_bc.py::ActionChunkBC.from_checkpoint`
- Lift-only：`source/imitation/lift_temporal.py::LiftTemporalController.from_checkpoint`
- ACT：`lerobot.policies.act.modeling_act.ACTPolicy(checkpoint["config"])` 后加载 state dict
- DP：`source/imitation/diffusion_policy.py::DiffusionPolicy.from_checkpoint`

## 2. 实际容量统计

| 模型/checkpoint | 输入 → 输出 | Hidden/结构 | 参数化 leaf 层 | Total params | Trainable | 文件大小 |
|---|---|---|---:|---:|---:|---:|
| P6 XYZ9 BC | RGB 3×120×160 + state39 → 16×9 | feature 256；3 Conv；3-layer MLP head 512→1024→1024→144 | 12 | 1,861,520 | 1,861,520 | 7,457,013 B / 7.112 MiB |
| P8 Single MLP | 当前帧 `(state39+past_xyz3)=42` → 4×3 | hidden 256；3 Linear | 3 | 79,884 | 79,884 | 324,341 B / 0.309 MiB |
| P8 History MLP | 8×42=336 → 4×3 | hidden 256；3 Linear | 3 | 155,148 | 155,148 | 625,397 B / 0.596 MiB |
| P8 GRU | 8×42 → 4×3 | 1-layer GRU hidden 256 + 3 Linear | 4 | 365,068 | 365,068 | 1,465,589 B / 1.398 MiB |
| Official ACT | RGB + state39 → 16×13 | ResNet18；dim 128；2 encoder、1 decoder、2 VAE encoder；FFN 512 | 58 | 12,319,117 | 12,319,117 | 49,435,087 B / 47.145 MiB |
| DP Vision+State | RGB + state39 → 16×13 | feature 256；50 diffusion steps；1024-wide denoiser | 14 | 2,337,232 | 2,337,232 | 9,363,317 B / 8.930 MiB |
| DP Vision+Tactile+State | RGB + tactile712 + state39 → 16×13 | 同上，增加 tactile 712→256 | 16 | 2,585,808 | 2,585,808 | 10,364,789 B / 9.885 MiB |

Checkpoint 证据：

- `outputs/dp_p6_2026-10-11/xyz9_bc/checkpoint.pt`
- `outputs/dp_p8_2026-10-14/models/{single_mlp,history_mlp,gru}/checkpoint.pt`
- `outputs/dp_p3_2026-10-11/act_single_sugar/checkpoint.pt`
- `outputs/dp_p1_2026-10-11/train_{vs,vts}_formal30/checkpoint.pt`

## 3. 模块参数分解

| 模型 | Visual encoder | Temporal encoder | Action head/denoiser | 其他条件编码 |
|---|---:|---:|---:|---:|
| XYZ9 BC | 128,256 | 0 | 1,722,512 | state 10,752 |
| Single MLP | 0 | 0 | 79,884 | 0 |
| History MLP | 0 | 0（历史直接 flatten） | 155,148 | 0 |
| GRU | 0 | 230,400 | 134,668 | 0 |
| ACT | 11,232,576 | 1,057,920 | 1,677 | state/latent/embedding 26,944 |
| DP Vision+State | 128,256 | 0 | 2,001,104 | state+condition 142,080；time 65,792 |
| DP Vision+Tactile+State | 128,256 | 0 | 2,001,104 | state+condition 207,616；tactile 183,040；time 65,792 |

ACT 的 temporal 项包含 Transformer encoder、decoder 与 VAE encoder；visual 项包含 ResNet18 backbone 和图像投影。DP 没有 observation-history encoder；time embedding 是 diffusion timestep 编码而不是观测时序编码。

## 4. GRU 是否欠拟合

| 模型 | 最佳 epoch | 最佳 train | 最佳 validation | epoch120 train | epoch120 validation |
|---|---:|---:|---:|---:|---:|
| Single MLP | 119 | 0.01274 | 0.02096 | 0.01261 | 0.02360 |
| History MLP | 108 | 0.00670 | 0.03986 | 0.00590 | 0.04402 |
| GRU state+action | 42 | 0.01093 | 0.03902 | 0.00265 | 0.04535 |
| GRU state-only | 17 | 0.05577 | 0.07744 | 0.00530 | 0.10319 |

结论：现有 GRU 没有明显容量欠拟合证据。epoch 42 后训练误差继续下降约 4.1 倍，而 validation 从 0.03902 恶化至 0.04535；state-only 更明显，从 0.07744 恶化至 0.10319。继续训练或直接扩容更可能加重 root-level 过拟合。现 checkpoint 已保存最佳 validation epoch，因此不是“训练太早停止”。

闭环上 GRU 1/2、Single MLP 0/2，说明时序归纳偏置有价值，但不能据此证明需要更多参数。当前更强的证据指向：只有 7 个 fit roots、BC Hold 状态分布外、缺少可部署 Lift phase/termination 信号。

## 5. 是否应该扩容

当前不建议直接扩容，原因：

1. GRU 训练误差已经很低，validation 先降后升；
2. 365k 参数相对于 1,750 个 fit 样本已不小；
3. 专家 Hold 可成功而 BC Hold 失败，主要是输入状态分布，不是模型无法拟合训练映射；
4. 过度抬升源于闭环 OOD 和缺少停止条件，增加 hidden size 不会自动修复；
5. ACT 已有 12.3M 参数、DP 2.34–2.59M 参数，历史失败不能归因于模型太小。

扩容只能在以下前提后作为受控消融：增加独立 BC-derived recoverable Hold roots、定义可部署终止器，并保持 root-level split。

## 6. Small / Medium / Large 容量消融配置

下表仅为配置建议和实例化容量核算，本次未训练。

### Lift-only GRU

| 级别 | History | Hidden | GRU layers | Horizon | 参数量 |
|---|---:|---:|---:|---:|---:|
| Small | 8 | 128 | 1 | 4×3 | 100,620 |
| Medium | 8 | 256 | 1 | 4×3 | 365,068（当前） |
| Large | 8 | 512 | 1 | 4×3 | 1,385,484 |

首轮只改变 hidden size，避免同时改变历史长度、层数和容量。若未来增加多层 GRU，应作为另一项独立消融。

### XYZ9 BC

| 级别 | Feature dim | Head width | Horizon | 参数量 |
|---|---:|---:|---:|---:|
| Small | 128 | 1024 | 16×9 | 1,577,488 |
| Medium | 256 | 1024 | 16×9 | 1,861,520（当前） |
| Large | 512 | 1024 | 16×9 | 2,429,584 |

### DP Vision+State

| 级别 | Feature dim | Denoiser width | Steps | 参数量 |
|---|---:|---:|---:|---:|
| Small | 128 | 1024 | 50 | 1,905,488 |
| Medium | 256 | 1024 | 50 | 2,337,232（当前） |
| Large | 512 | 1024 | 50 | 3,495,632 |

### ACT

| 级别 | dim_model / FFN | Encoder / Decoder / VAE layers | 参数量 |
|---|---|---|---:|
| Small | 64 / 256 | 1 / 1 / 1 | 11,380,941 |
| Medium | 128 / 512 | 2 / 1 / 2 | 12,319,117（当前） |
| Large | 256 / 1024 | 4 / 1 / 4 | 18,727,437 |

ACT 参数主要来自 ResNet18，因此 Small 仍超过 11M。正式容量消融必须使用相同 root split、训练预算和 early stopping，不能按单模型单独调数据。

## 7. 约 260 mm 过度抬升专项审计

### 7.1 实际行为

未见 root 8：

- 250 control steps，63 个 chunk；
- wrist 总上升 281.09 mm；object 最大抬升 259.76 mm；
- step 0–39 平均预测 Δz 0.11 mm；
- step 40–129 平均预测 Δz 3.25 mm；
- step 130–249 平均预测 Δz 4.85 mm；
- 对应训练标签 Hold/Lift/Verify 平均 Δz 约 0.33/3.05/0.59 mm；
- 0 次 IK failure、0 次 joint-limit violation（原 P8 summary/trace）。

root 9 同样在后 120 步持续输出平均 4.63 mm 正 Δz，最终因物体尾段角速度 0.2499 rad/s 超过 FINAL 0.1 rad/s 而拒绝，并非接触丢失。

### 7.2 是否为动作累计 bug

单个 chunk 内不是累计 bug。`evaluate_lift_temporal_controller.py:84-96` 在每次推理开始读取一次 anchor，随后 4 个 XYZ 都相对同一个 anchor 解码，符合 P6 XYZ9 合同。

跨 chunk 会累计，但这是 receding-horizon 相对控制的预期语义：每 4 步重新以当前实测 pose 建立新 anchor。问题是模型进入训练状态范围外后仍不断输出正 Δz，导致每个新 chunk 把正增量继续叠加。

### 7.3 是否为终止条件问题

是重要诱因。环境由 `apps/trace_diffusion_rollout.py::_make_env` 创建，并明确设置 `terminate_on_success=False`。Lift-only evaluator 又固定运行 `max_steps=250`，没有基于可部署机器人状态/触觉的 phase completion 或停止器。Strict verification 仅在轨迹完成后执行，不能在线停止过抬升。

### 7.4 是否为控制执行/IK错误

现有证据不支持。成功和失败 root 都是 0 次 IK failure；动作经过 `ContinuousCartesianAdapter.decode`、joint limits 和环境 action bounds。腕部与物体同步上升，说明命令被实际执行。问题发生在模型输出和 rollout 生命周期，而非底层 IK 没执行目标。

### 7.5 根因排序

1. **缺少可部署 Lift 完成/停止条件**：策略被强制执行完整 250 步；
2. **闭环 OOD 正反馈**：腕部超过专家训练高度后，GRU 仍输出 4–5 mm 正 Δz；
3. **状态/动作历史缺少明确 phase-progress 可观测量**：39D actuator state 可以间接编码高度，但泛化不足；
4. **训练 root 数量过少**：仅 7 个 fit roots；
5. 动作累计实现错误或 IK 错误：当前证据不支持。

## 8. 建议的下一项最小实验

暂不扩容。先冻结当前 GRU，加入不依赖物体真值的有界停止器诊断，例如腕部相对 Lift-start 上升上限、低速度稳定窗口和触觉保持条件，并在相同 root 8/9 重放。随后扩充 BC-derived、可恢复的独立 Hold roots。只有 Medium 在更大 root split 上表现为 train/validation 同步平台期且闭环受表达能力限制时，才运行 Small/Medium/Large 容量消融。
