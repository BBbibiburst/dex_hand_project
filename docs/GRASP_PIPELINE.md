# 抓取生成与验证流水线

## 目标

为物体目录生成不仅能在最终姿态保持物体，而且能从无接触状态沿实际路径闭合，并尽可能由
完整机械臂完成 Lift 的抓取配置。

## 流程

1. 闭链 GraspQP 适配器以 MuJoCo 有限差分运动学联合优化手腕位姿和六个驱动量；
   force-closure 使用官方 QP metric，并在种子目标中加入桌面半空间惩罚。
2. MuJoCo evolutionary refinement 变异腕部位姿与六个执行器状态。
3. Direct-hold 仿真快速筛选动力学稳定候选。
4. 对进化候选重新规划 approach 与 closure；每个候选使用独立 seed 搜索方向。
5. 检查完整手部网格、物体碰撞和桌面硬约束。
6. 实际执行 approach、closure 和 hold，产生 trajectory validation 结果。
7. 在 RM75B 完整场景执行 IK 预检及 scripted Lift，单独记录 Robot Lift 结果。

完整 Robot Lift 前会在一个复用的 RM75B 场景中批量检查所有 trajectory-stable 候选的
全部腕部路点。IK 不可达或关节插值会碰桌面的候选被后置并跳过动态执行；可行候选按 IK
残差、桌面余量和进化 fitness 排序。这样不会再为确定不可执行的候选反复运行最长 900
步仿真。多次独立搜索全部结束后，报告选中的最佳尝试会重新原子写回对应 grasp JSON，
保证 benchmark 行与后续示教实际读取的配置一致。

进化 worker 会按物体 mesh、scale 和手型结构复用已编译的 MuJoCo `MjModel`；每个
候选只创建新 `MjData` 并重置相对位姿，不再重复解析 XML 和编译 mesh。

这里借鉴 DexEvolve 的 simulator-in-the-loop、无梯度进化思想，但后端是 MuJoCo，不声称
复现 Isaac Sim 的 GPU 吞吐或论文中的完整硬件参数。

Dex Hand 的六驱动闭链不能直接交给 GraspQP 的通用开链 URDF `HandModel`。因此前端复用
官方 force-closure metric，但由 MuJoCo 解闭链 FK/Jacobian；优化后的完整网格还必须通过
桌面 5 mm 硬过滤。该阶段只生成种子，最终稳定性仍由后续进化和物理执行决定。

## 标准命令

```bash
python -m tools.grasping.benchmark_catalog --full-pipeline
```

全量运行前建议先执行代表性诊断：

```bash
python -m tools.grasping.benchmark_catalog --pilot
```

Pilot 固定单进程，逐物体输出累计 Lift 成功率和失败原因。默认在至少 4 个结果后检查：
Lift 成功率低于 25%，或同一失败连续出现 3 次时，以退出码 2 提前结束并保留
`configs/grasps/dex_hand/pilot_benchmark.json`。阈值可通过 `--pilot-min-results`、
`--pilot-min-lift-rate` 和 `--pilot-max-repeated-failure` 调整。

预设包括 GraspQP、20 代进化、完整轨迹验证、Robot Lift 验证和最多 5 次独立搜索。
运行 `--help` 查看当前参数，不要从旧日志复制超长参数列表。
当前完整预设以 `robot_lift_verified` 为最终目标。某次搜索得到
`trajectory_stable` 但 `lift=FAIL` 时，会继续使用剩余独立搜索 seed（最多 5 次），
而不是把仅通过固定手轨迹验证的配置视作完成。所有尝试仍失败时，报告保留整机执行阶段
进展最远的结果，便于后续定向补跑。

### 小规模测试

```bash
python -m tools.grasping.benchmark_catalog \
  --full-pipeline \
  --object-id ycb:003_cracker_box \
  --output configs/grasps/dex_hand/cracker_pipeline.json
```

### 恢复与补跑

```bash
# 只继续未写入报告的物体
python -m tools.grasping.benchmark_catalog --full-pipeline --resume

# 重算 trajectory 或 Robot Lift 未完成的行；完整预设默认启用该语义
python -m tools.grasping.benchmark_catalog \
  --full-pipeline --resume --retry-incomplete
```

## 状态定义

| Benchmark status | 含义 | 能否进入示教采集 |
| --- | --- | --- |
| `trajectory_stable` | 完整 approach/closure/hold 通过 | 可以 |
| `direct_hold_only` | 终态可保持，但完整轨迹未通过 | 不可以 |
| `unstable` | 动力学保持失败 | 不可以 |
| `validation_error` | 配置生成后验证异常 | 不可以 |
| `search_error` | 未生成可用候选 | 不可以 |

`robot_lift_verified` 是独立布尔字段，不是 benchmark status。这样可以区分：

```text
手部相对物体轨迹可执行
≠
完整机械臂在特定场景中可达并成功 Lift
```

## 输出结构

```text
configs/grasps/dex_hand/
├── full_pipeline_benchmark.json
├── graspqp_seeds/
│   └── <object>.json
└── dexevolve/
    └── <object>.json
```

最终配置可能包含：

- 最终手腕位姿和执行器值
- approach 与 closure waypoints
- `direct_hold_stable`
- `trajectory_collision_free`
- `trajectory_hold_stable`
- `trajectory_stable_candidates`
- 桌面间隙与动力学指标
- `robot_lift_verified`
- `robot_table_collision`

Benchmark 报告保存逐物体状态、配置路径、搜索/进化摘要、Lift 尝试、耗时和总体统计。
总体 `failure_reasons` 还会汇总轨迹碰撞、IK 不可达、机器人碰桌和各 Lift
阶段失败，便于在全量任务未结束时分析。
每个物体的 `phase_seconds` 分别记录 `search`、`evolution`、
`trajectory_replan_and_validation` 和 `robot_lift_validation`，用于判断真实瓶颈。

## 运行输出

```text
[15/127] TRAJECTORY_STABLE ycb:016_pear lift=PASS object=18m42s avg=2m31s eta=4h42m
```

- `object`：该物体累计处理时间。
- `throughput`：本轮按并行吞吐计算的平均墙钟时间/物体。
- `eta`：至少完成一整批 worker 前显示 `warming_up`，之后才给出动态剩余时间。
- `lift=PASS/FAIL`：完整机械臂验证结果；没有进入验证时不显示。

结束统计：

```text
completed=127/127 generated=127/127 trajectory_stable=... trajectory_stable_rate=...
status_counts=...
robot_lift_verified=.../... robot_lift_verified_rate=...
total_elapsed=...
```

## 失败诊断

### `direct_hold_only`

常见原因是接近方向不可行、闭合时刚性结构碰撞或桌面间隙不足。不能将其当作最终稳定抓取。

### `robot_ik_unreachable_waypoint_*`

手部相对轨迹有效，但机械臂无法达到 waypoint。IK 预检会立即淘汰候选，避免空跑完整 episode。

### `robot_table_collision_waypoint_*`

机械臂或手在完整场景中碰桌，需要换候选或重新生成，不能通过降低桌面硬约束解决。

### Lift 到 `approach/grasp/lift/verify` 失败

- `approach`：通常为可达性或跟踪问题。
- `grasp`：闭合过程、候选选择或桌面碰撞。
- `lift`：抓住后提升滑落。
- `verify`：短暂成功但未持续满足最终条件。

不要盲目放宽 `v3_base_link` 等刚性几何碰撞、桌面间隙或稳定阈值。这些失败应优先通过新
候选、不同重规划 seed 或执行可达性解决。

## Schema 兼容

当前写入：

```text
benchmark schema_version=4
validation_semantics=trajectory-hold-v2
```

可视化层可以读取旧的歧义 `stable` 并显示为 `legacy_stable`，但生成器和 collector 不会
继续写入或执行旧语义报告。
