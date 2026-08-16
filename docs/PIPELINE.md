# Ultra Prior、Wrist Lattice 与 MJWarp PPO 验证

本文描述当前 Dex Hand 主流水线。所有命令从项目根目录运行，不需要第三方源码 checkout
或 Git submodule。

## 流水线阶段

1. **Dex Hand 标定**：从权威 MJCF 采样六个驱动下的手掌与指面轨迹，拟合可微 surrogate。
2. **Ultra Prior**：优化物体相对手掌位姿、六个手驱动以及接触力闭合指标。
3. **Wrist Lattice**：围绕优先候选展开局部腕部平移和旋转，并用 RM75B IK 排除不可达姿态。
4. **MJWarp PPO**：策略离散选择腕部模板，同时连续编辑六个手驱动，在 GPU 上并行评估。
5. **C MuJoCo 复验**：重新执行最佳控制序列，验证末段持续抬升，并检查轨迹与权威
   MuJoCo 模型、当前控制维度是否一致。

## 环境与模型检查

```bash
python -m tools.ultradexgrasp.probe \
  --strict \
  --mjwarp \
  --device cuda:0

python -m pytest -q tests/test_dex_hand_mjcf.py
python -m tools.grasping.benchmark_hand_physics --steps 4000
```

修改 `assets/grippers/dex_hand/dex_hand.xml` 后使用：

```bash
python -m tools.ultradexgrasp.probe \
  --strict \
  --mjwarp \
  --device cuda:0 \
  --recalibrate-surrogate
```

## 缓存规则

一次有效的新手模型测试必须同时满足：

- surrogate 由当前 MJCF 重新标定；
- Ultra Prior 不复用旧手模型生成的 episode；
- Wrist Lattice 不复用旧手模型执行过的轨迹；
- PPO 使用新的输出目录和 checkpoint。

`--force` 只会忽略部分 benchmark/PPO 结果。手模型变化后应使用新的输出根目录，同时把
`--ultra-root` 和 `--lattice-root` 指向该目录下的新位置。

## Pilot

建议先覆盖盒体、柱体、容器、带柄和不规则对象：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -m tools.grasping.batch_grasp_edit \
  --object-id ycb:002_master_chef_can \
  --object-id ycb:003_cracker_box \
  --object-id ycb:024_bowl \
  --object-id ycb:025_mug \
  --object-id egad:C0 \
  --output outputs/dex_hand_pilot \
  --ultra-root outputs/dex_hand_pilot/ultra \
  --lattice-root outputs/dex_hand_pilot/lattice \
  --device cuda:0 \
  --num-envs 64 \
  --initial-updates 2 \
  --mid-updates 4 \
  --max-updates 6 \
  --train-ultra-success \
  --train-lattice-success \
  --verbose-child
```

Pilot 应满足：没有 traceback、CUDA OOM、NaN、MJWarp capacity overflow 或
`PIPELINE_ERROR`，并且有可达模板的对象实际产生 `rl_updates`。

## 127 对象全量运行

以下参数适合单张 24 GB GPU。显存不足时把 `--num-envs` 降为 32：

```bash
MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
python -m tools.grasping.batch_grasp_edit \
  --dataset all \
  --expect-count 127 \
  --output outputs/dex_hand_ppo127 \
  --ultra-root outputs/dex_hand_ppo127/ultra \
  --lattice-root outputs/dex_hand_ppo127/lattice \
  --device cuda:0 \
  --num-envs 64 \
  --initial-updates 5 \
  --mid-updates 10 \
  --max-updates 15 \
  --base-candidates 3 \
  --lattice-max-templates 12 \
  --lattice-max-executions 32 \
  --train-ultra-success \
  --train-lattice-success
```

两个 `--train-*-success` 参数用于压力测试：即使 Ultra 或 Lattice 已经成功，只要模板可用，
仍然运行 PPO。正常的分层生产流程应移除这两个参数，让已经解决的对象提前退出。

任务被中断时，使用完全相同的参数重新运行。匹配的每对象结果会被自动复用。

## 结果状态

| 状态 | 含义 |
| --- | --- |
| `ULTRA_SUCCESS` | Ultra episode 已成功；压力测试中仍可能包含 PPO 指标 |
| `LATTICE_SUCCESS` | 某个可达 Wrist Lattice 模板已成功 |
| `RL_SUCCESS` | MJWarp PPO 产生成功轨迹，需要 C MuJoCo 复验 |
| `RL_PROMISING` | 有抬升或成功信号，但预算内未达到成功标准 |
| `DIRECT_FAILED` | Ultra、Lattice 和当前 PPO 预算均无有效进展 |
| `NO_ULTRA_PRIOR` | 无法生成完整 Ultra episode |
| `NO_REACHABLE_TEMPLATE` | 腕部候选无法通过 IK/轨迹执行 |
| `PIPELINE_ERROR` | 程序、依赖、CUDA 或数据错误 |

检查汇总：

```bash
jq '{
  count,
  status_counts,
  ppo_eligible: ([.results[] | select(.lattice_templates > 0)] | length),
  ppo_ran: ([.results[] | select(.rl_updates > 0)] | length)
}' outputs/dex_hand_ppo127/summary.json
```

批处理返回 0 只表示没有 `PIPELINE_ERROR`，不等于 127 个对象全部抓取成功。压力测试还应
确认 `ppo_ran == ppo_eligible`。

## C MuJoCo 回放

MJWarp 成功轨迹必须在 C MuJoCo 中复验：

```bash
find outputs/dex_hand_ppo127/rl \
  -path '*/best_trajectory/manifest.json' -print0 |
while IFS= read -r -d '' manifest; do
  python -m tools.rl.replay_trajectory "$manifest" ||
    echo "REPLAY_FAIL $manifest"
done
```

当前回放成功判据为：最后一帧满足 Lift 任务条件，且末尾最多 20 个控制帧中至少 80% 满足
该条件。Lift 条件是物体底部高于桌面 4 cm。这个返回码不会单独统计接触数量、物体转动、
滑移或几何穿透；需要这些指标时，应另外保存接触和物体姿态时序，不能仅从回放成功推断。

最终报告应同时保存 benchmark commit、MJCF、配置、`summary.json` 和失败对象日志，以便后续
比较手指包络、抓取覆盖率和 C MuJoCo/MJWarp 一致性。
