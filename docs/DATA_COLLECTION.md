# 多样化示教数据采集

## 输入与目标

输入是 `full_pipeline_benchmark.json` 及其中每个物体指向的抓取配置。采集器在随机化 Lift
环境中实际执行：

```text
approach → grasp → lift → verify
```

只有 `strategy_verified_success=true` 的 episode 才作为成功示教保存。静态抓姿本身不会被
直接当作示教。

## 仅评测成功率

不创建 LeRobotDataset：

```bash
python -m apps.collect_scripted_lerobot \
  --task lift \
  --grasp-benchmark-report configs/grasps/dex_hand/full_pipeline_benchmark.json \
  --trials-per-object 10 \
  --evaluation-output configs/grasps/dex_hand/lift_task_evaluation.json
```

## 多样化采集

```bash
python -m apps.collect_scripted_lerobot \
  --task lift \
  --grasp-benchmark-report configs/grasps/dex_hand/full_pipeline_benchmark.json \
  --coverage-search \
  --target-successes-per-object 20 \
  --max-coverage-seeds 100 \
  --max-coverage-candidates 16 \
  --output datasets/scripted_lift_diverse \
  --repo-id local/dex-hand-scripted-lift-diverse \
  --evaluation-output configs/grasps/dex_hand/lift_diversity_evaluation.json
```

如果 127 个物体都为 `trajectory_stable` 且每个都达到目标，20 条/物体对应 2540 条成功
episode。

## 多样性策略

- 不同 seed 随机化物体位置与 yaw。
- 每个 seed 最多保存一条成功轨迹，避免相同初始位姿的近重复数据。
- 候选尝试顺序随 seed 轮换，避免候选 0 支配数据集。
- 最多使用 `max_coverage_candidates` 个 `trajectory_stable_candidates`。
- 达到该物体的目标成功数后才进入下一个物体。
- 失败 episode 默认不保存。

因此必须满足：

```text
target_successes_per_object <= max_coverage_seeds
```

## 运行输出

```text
[1/127] ycb:002_master_chef_can attempt=1 seed=0 candidate=0 FAILED phase=approach
[1/127] ycb:002_master_chef_can attempt=2 seed=0 candidate=1 SUCCESS phase=verify
[1/127] ycb:002_master_chef_can attempt=3 seed=1 candidate=1 SUCCESS phase=verify
```

同一个 seed 成功后不会继续保存其他候选，而是进入下一个随机位姿。

## 输出

LeRobotDataset：

```text
datasets/scripted_lift_diverse/
```

评测与覆盖报告：

```text
configs/grasps/dex_hand/lift_diversity_evaluation.json
```

当前采集评测报告 `schema_version=5`；它将 `object_ids` 和 `limit` 也纳入恢复
参数，避免将不同物体选择的运行结果混合。

逐物体统计包括：

- `successes`、`trials`、`success_rate`
- `coverage_success`
- 成功的 seed/candidate 组合
- `unique_successful_seeds`
- `unique_successful_candidates`
- `candidate_coverage_rate`
- `initial_position_span_xyz`
- `initial_yaw_bins`

总体重点指标：

```text
objects_with_success
object_coverage_rate
successful_episodes
micro_success_rate
macro_object_success_rate
```

“所有物体至少一条”应检查 `objects_with_success`；“每个物体达到 N 条”应检查逐物体
`coverage_success`。

## 恢复

```bash
python -m apps.collect_scripted_lerobot \
  ...原有参数... \
  --resume-evaluation
```

恢复要求 source report、schema 和采集参数一致。已经完成的物体会跳过。LeRobot recorder
也会检查已有数据集的 FPS、shape 和 feature schema，避免不兼容数据混写。
在 coverage 模式下，只跳过已达到 `target_successes_per_object` 的物体；未达标物体
保留已有成功 episode，并从尚未尝试的 seed 继续。

## 限定对象

```bash
# 单物体
--object-id ycb:025_mug

# 只采集 YCB
--dataset ycb

# 快速检查前 10 个匹配对象
--limit 10
```

Collector 只读取同时满足 `trajectory_stable` 和 `robot_lift_verified=true` 的行，并复现
benchmark 保存的成功 `task_scene`。若物体只有 `direct_hold_only` 或未通过 Robot Lift，
应先用 `--resume --retry-incomplete` 修复，而不是在采集阶段绕过验证。

## 数据字段

每帧主要包含：

- MuJoCo `qpos/qvel/ctrl` 拼接状态
- IK/手部 action
- agent-view RGB 图像或视频
- 触觉观测
- 从 scripted action 构造的 glove 与 Vive pose 表示
- `lift:<object_id>` task 标签

Lift 的最终 verify 依赖触觉，不要为该流程传 `--no-tactile`。
