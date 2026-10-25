# P10-Vision：1920×1080 相机与视觉输入审计

日期：2026-10-24  
范围：固定 Sugar Box、冻结 XYZ9 BC、低成本相机/预处理/ROI/计算代价审计。未修改物理参数、Strict profile、成功阈值和既有 checkpoint；未启动高分辨率正式训练。

## 1. Executive Summary

MuJoCo 已成功稳定输出 1920×1080 RGB。高清图像显著增加了手指与物体的可见像素：Close 帧中物体由 160×120 下约 36 px 增至 1920×1080 下 2939 px，局部边缘更清楚。但冻结 160×120 XYZ9 BC 的配对闭环没有从“1080p 渲染后缩小”中获益：两组接触率同为 311/340，Strict 均失败，最大抬升从 11.24 mm 降至 4.66 mm。

固定相机 ROI 在 10 条独立扰动根、40 个 Close/Hold 帧中完整覆盖物体及接触邻域（40/40，最小覆盖率 100%），因此 ROI 比直接把整张 1080p 输入现有编码器更值得继续验证。现有 BC 直接输入 1080p 时，RTX 4060 Laptop GPU 上稳态推理 P50/P95 为 183.76/230.80 ms，已经超过 20 Hz 的 50 ms 控制预算；同时编码器最终使用全局平均池化，更多像素不等于局部接触信息一定能被保留。

结论：视觉细节确实不足，但目前没有证据表明它是承载抓持失败的主要原因。本轮不启动全帧 1080p 训练。下一步应做固定 ROI 双分支的小规模单轨迹对照；若仍无承载收益，返回 Close 阶段时序接触形成与 arm/hand 协同主线。

## 2. 基线保护与可复现性

- Git HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`
- 开始前工作区已有 P0–P9 修改及未跟踪产物；完整快照见 `outputs/dp_p10_vision_2026-10-24/provenance/baseline.txt`。
- XYZ9 BC SHA256：`671a091347b1c8c2c8e06bcff8153cc3113b00541846713ef1b9dd4605718d9d`
- P8 GRU SHA256：`765d64fc310e662a91524c4b6814866310d2510d10cbb22ad4b96eca55dc796d`
- Supervisor 是确定性代码/配置，不存在独立学习 checkpoint。
- P4 基线：`outputs/dp_p4_2026-10-11/`
- P9 产物：`outputs/dp_p9_2026-10-15/`
- 本轮根目录：`outputs/dp_p10_vision_2026-10-24/`

重启后原先使用的 `dex-hand` conda 环境路径已不存在；经实际导入检查，本轮使用现存 `mujoco` 环境（PyTorch 2.10.0+cu128、MuJoCo 3.10.0）。

## 3. 阶段 A：相机与预处理链路

实际代码链路：

`MuJoCo Renderer RGB uint8 HWC → torch CHW → float32 / 255 → VisionEncoder`

审计结果：

- 后端：`MUJOCO_GL=egl`；当前机器打印 `/dev/dri` 权限警告，但 RGB 渲染成功。
- 相机：`agentview`，固定 world camera；FOV 55°。
- 相机位置：`[1.025, -0.995929, 1.363953]` m。
- near/far：约 0.1061/530.49 m。
- RGB：`uint8`、HWC、RGB 顺序，范围有效，无黑屏和异常颜色。
- 原训练和推理路径均没有裁剪、resize、颜色空间转换或图像均值方差归一化；原数据本身就是 160×120。
- 冻结 BC 的 `VisionEncoder` 为 3 层卷积后 `AdaptiveAvgPool2d(1)`，形状上可接受任意分辨率，但会把最终空间特征全局汇聚。
- 1920×1080 是 16:9，原 160×120、320×240、640×480 是 4:3。相同垂直 FOV 下，1080p 帧水平视野更宽；直接缩成 4:3 会产生几何压缩。这是本轮必须保留的混杂因素。
- MuJoCo 默认 offscreen framebuffer 为 640×480。为生成 1080p，仅增大 `model.vis.global_.offwidth/offheight`；没有改变物理、相机或任务参数。

相关代码：

- `source/envs/manipulation/arenas.py`
- `source/imitation/diffusion_policy.py`
- `source/imitation/vision_input.py`
- `apps/audit_vision_resolution.py`
- `apps/trace_cartesian_action_chunk_bc.py`

## 4. 四种分辨率静态对照

同一专家 episode、同一精确物理状态、同一相机，仅改变输出分辨率。

| 分辨率 | RGB shape | 单帧缓冲 | Close物体像素 | 物体bbox |
|---|---:|---:|---:|---:|
| 160×120 | 120×160×3 | 0.055 MiB | 36 | 5×13 px |
| 320×240 | 240×320×3 | 0.220 MiB | 145 | 10×27 px |
| 640×480 | 480×640×3 | 0.879 MiB | 580 | 19×55 px |
| 1920×1080 | 1080×1920×3 | 5.933 MiB | 2939 | 43×123 px |

本机 EGL 渲染计时在四种分辨率下均约 83–98 ms，且没有随像素数正常缩放。这与当前 `/dev/dri` 权限/软件回退环境一致，不能作为正式 GPU 渲染吞吐结论。实际帧缓冲字节数和进程 RSS 增量（整轮约 435 MiB）作为可复现成本证据保留。

图像：

- `outputs/dp_p10_vision_2026-10-24/static_audit/resolution_stage_comparison.png`
- `outputs/dp_p10_vision_2026-10-24/static_audit/direct_downsample_roi_comparison.png`
- `outputs/dp_p10_vision_2026-10-24/figures/detail_vs_inference_cost.png`

1080p 的确显示了更完整的指缘与物体边缘，但固定 `agentview` 仍存在手指/物体互相遮挡；分辨率不能消除视角遮挡。

## 5. 阶段 B：冻结 160×120 XYZ9 BC 配对闭环

两次评估使用相同 manifest、exact reset、seed=2011、action_steps=4、控制频率、IK 与 Strict profile。唯一输入差异是：

- A：直接渲染 160×120；
- B：渲染 1920×1080，再用 Pillow bilinear 缩为 160×120。

| 指标 | 160直接渲染 | 1080p→160 | 结论 |
|---|---:|---:|---|
| 接触帧 | 311/340 | 311/340 | 相同 |
| 多指接触率 | 88.53% | 87.06% | 未改善 |
| 拇指对向接触率 | 82.06% | 81.47% | 未改善 |
| 最大抬升 | 11.24 mm | 4.66 mm | 下降 |
| 最终抬升 | 1.82 mm | 0.08 mm | 下降 |
| object-following ratio | 0.0046 | 0.0371 | 两者绝对承载均接近零 |
| IK失败 | 0/340 | 0/340 | 相同 |
| Strict Lift | 0/1 | 0/1 | 均失败 |
| resize P50/P95 | 0.013/0.018 ms | 7.29/8.54 ms | 额外开销 |

1080p→160 与直接 160 的 Close 图像 MAE 为 16.10/255、RMSE 33.25/255，约 94.34% 像素不同。这一差异同时包含抗锯齿、宽高比/视野变化和重采样，不能只归因于“高清细节”。

视频与详细轨迹：

- `outputs/dp_p10_vision_2026-10-24/paired/direct_160/bc_rollout.mp4`
- `outputs/dp_p10_vision_2026-10-24/paired/render_1920_to_160/bc_rollout.mp4`
- `outputs/dp_p10_vision_2026-10-24/figures/paired_frozen_bc_comparison.png`

本实验只有一个严格配对 reset，只能证明该固定条件下没有收益，不能声称统计显著。

## 6. 阶段 C：固定局部 ROI

ROI 由一条专家轨迹 Close+Hold 的离线分割标定得到，归一化坐标：

`[x0, y0, x1, y1] = [0.493229, 0.415741, 0.568229, 0.782407]`

在 1920×1080 下对应 `[947, 449, 1091, 845]`。在线裁剪只需要固定相机标定坐标，不读取 MuJoCo 物体真值、未来帧或阶段标签。分割只用于离线覆盖审计。

独立根验证：10 条真实扰动成功根 × Close/Hold start/mid = 40 帧。

- 物体完整覆盖：40/40；平均/最小均 100%。
- 物体邻域机器人像素覆盖：40/40；平均/最小均 100%。
- ROI 明显放大指尖、物体轮廓和接触边缘。
- 当前结果仅覆盖 Sugar Box 与现有扰动范围；不能外推到其他物体、相机位移或更大 reset 分布。

证据：

- `outputs/dp_p10_vision_2026-10-24/roi_independent_roots/summary.json`
- `outputs/dp_p10_vision_2026-10-24/figures/fixed_roi_independent_roots.jpg`

## 7. 原生高分辨率模型输入成本

实际加载 P6 XYZ9 BC checkpoint，在 RTX 4060 Laptop GPU 上直接输入各分辨率。参数量恒为 1,861,520；这里只改变输入张量尺寸。

| 输入 | 推理P50 | 推理P95 | 推理峰值增量 | 单样本前反向 | 训练峰值增量 |
|---|---:|---:|---:|---:|---:|
| 160×120 | 16.04 ms | 30.67 ms | 1.54 MiB | 18.72 ms | 11.48 MiB |
| 320×240 | 15.40 ms | 29.84 ms | 6.16 MiB | 33.74 ms | 24.54 MiB |
| 640×480 | 59.00 ms | 133.47 ms | 18.75 MiB | 82.89 ms | 65.52 MiB |
| 1920×1080 | 183.76 ms | 230.80 ms | 128.00 MiB | 384.80 ms | 399.02 MiB |

推理数据均经过 10 次 warmup、40 次测量；前反向经过 2 次 warmup、5 次测量，表中为 P50。1080p 稳态推理仍明确超出 50 ms 控制周期。GPU 调度存在抖动，所以结论以数量级和是否跨越 50 ms 控制预算为主。

这些是 batch=1 增量值，不是正式 batch=64 训练显存。直接按 batch 线性外推不可靠，因此未伪报完整训练成本。现有全局池化编码器也不适合用“直接增加所有像素”来提取局部接触信息。

## 8. 是否启动高清训练

本轮没有启动 640×480 或 1920×1080 正式训练，理由：

1. 冻结策略的 1080p→160 配对闭环没有任何承载收益；
2. 全帧 1080p 原生推理 P95 230.80 ms，不满足 20 Hz；
3. 原编码器全局池化，架构上缺少显式局部接触保留机制；
4. ROI 虽在 10 个独立根上覆盖可靠，但尚未经过学习对照，证据只支持“小规模 ROI 可行性训练”，不支持全帧高清大训练。

因此止步于 A–C 和计算成本检查，符合预设止损条件。

## 9. 八个必须回答的问题

1. **MuJoCo 是否成功生成 1920×1080？** 是。RGB shape、dtype、颜色与连续渲染均正常；需显式扩大 offscreen framebuffer。
2. **160×120 是否损失关键细节？** 是，Close 物体仅约 36 px、bbox 约 5×13 px，指缘与接触边缘严重压缩；1080p 有 2939 个物体像素。但“细节损失”尚不等于“主要控制瓶颈”。
3. **高清渲染缩到原尺寸是否变化？** 图像明显变化，但固定 reset 下接触不变、最大/最终抬升更差，Strict 仍 0/1。
4. **ROI 是否更适合？** 当前证据支持“比全帧 1080p 更值得下一步测试”：10 个独立根 40/40 完整覆盖，且计算可局部化；尚未证明闭环收益。
5. **真正高清输入是否值得训练？** 全帧 1080p 不值得；固定 ROI 双分支值得一次有界小规模训练。640×480 全局输入可作为成本对照，但优先级低于 ROI。
6. **延迟、显存与训练成本？** 1080p 原生推理 P50/P95 183.76/230.80 ms；batch=1 推理/单步训练峰值增量约 128/399 MiB；稳态单样本前反向 P50 约 384.80 ms。OpenGL 渲染显存当前无法由 torch 读取。
7. **视觉不足是承载失败主因吗？** 没有证据。高清缩小后接触仍充足但物体不随腕部运动，说明 Close/Hold 接触力学及时序协同仍是更直接的瓶颈。
8. **下一步？** 先做固定 ROI + 全局低分辨率双分支的单轨迹、同预算对照；若不能提升 object-following 与 Strict Lift，立即回到 Close 阶段时序接触形成优化，不扩大全帧高清训练。

## 10. 修改文件与测试

新增：

- `source/imitation/vision_input.py`
- `apps/audit_vision_resolution.py`
- `apps/audit_fixed_vision_roi_roots.py`
- `apps/benchmark_bc_vision_resolution.py`
- `apps/plot_vision_audit_summary.py`
- `tests/test_vision_input.py`

修改：

- `apps/trace_cartesian_action_chunk_bc.py`：新增独立渲染/模型输入分辨率、显式 resize、分项延迟记录；默认行为保持不变。

定向验证：

- P10 涉及文件 `ruff check`：通过。
- `pytest -q tests/test_vision_input.py`：6 passed。
- 全量 `pytest -q`：193 passed，3 个既有 `torch.jit.script` 弃用警告。
- 仓库级 `ruff check apps source tests tools` 仍被两份既有 PPT 工具脚本的 67 个格式问题阻断：`tools/build_p3_p10_ppt.py`、`tools/render_pptx_preview.py`。本轮没有修改这两个无关文件；P10 文件自身无 Ruff 错误。

没有 commit 或 push。

## 11. 关键命令

```bash
conda run -n mujoco python -m apps.audit_vision_resolution \
  --manifest outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p00/candidate_00/manifest.json \
  --output outputs/dp_p10_vision_2026-10-24/static_audit

conda run -n mujoco python -m apps.trace_cartesian_action_chunk_bc \
  --checkpoint outputs/dp_p6_2026-10-11/xyz9_bc/checkpoint.pt \
  --manifest outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p00/candidate_00/manifest.json \
  --mode xyz9 --action-steps 4 --seed 2011 --image-width 160 --image-height 120 \
  --render-width 1920 --render-height 1080 \
  --output outputs/dp_p10_vision_2026-10-24/paired/render_1920_to_160

conda run -n mujoco python -m apps.audit_fixed_vision_roi_roots \
  --source-a outputs/dp_p8_2026-10-14/dataset_audit/source_a.json \
  --roi-summary outputs/dp_p10_vision_2026-10-24/static_audit/summary.json \
  --output outputs/dp_p10_vision_2026-10-24/roi_independent_roots

conda run -n mujoco python -m apps.benchmark_bc_vision_resolution \
  --checkpoint outputs/dp_p6_2026-10-11/xyz9_bc/checkpoint.pt \
  --frames outputs/dp_p10_vision_2026-10-24/static_audit/frames \
  --output outputs/dp_p10_vision_2026-10-24/model_resolution_benchmark \
  --device cuda:0 --repeats 40 --warmup 10
```

