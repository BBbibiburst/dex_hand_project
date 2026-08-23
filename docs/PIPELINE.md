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

## 自动示教与 Diffusion Policy 数据

只有通过最终 C MuJoCo 复验、状态为 `FINAL_VERIFIED` 的 Ultra/Lattice/PPO 轨迹，才会默认
进入自动示教数据集。采集入口在权威 MuJoCo 环境中重新执行控制序列，并同步记录机器人状态、
相机图像、触觉和动作到 LeRobot：

```bash
python -m apps.collect_generated_lerobot \
  --input-root outputs/dex_hand_ppo127 \
  --output datasets/ultra_lerobot \
  --repo-id local/dex-hand-ultra-demonstrations
```

随后训练与评估 Diffusion Policy：

```bash
python -m apps.train_diffusion \
  --dataset datasets/ultra_lerobot \
  --repo-id local/dex-hand-ultra-demonstrations

python -m apps.evaluate_diffusion \
  --checkpoint checkpoints/diffusion_policy.pt \
  --task lift
```

`--allow-unverified` 只用于诊断，不能用于正式 DP 训练集。自动生成的数据不伪造 Vive 或手套
字段；遥操作采集入口仍单独保留这些 operator metadata。

## 接触模型与摩擦

Dex Hand 指垫使用 `condim=4`、滑动摩擦 `1.3`、扭转摩擦 `0.02`，并以更高 contact
`priority` 提供手—物体接触的柔顺参数。这样刚性 YCB 网格不会再把机械臂默认的硬接触参数
混入软指垫接触。所有动态加入的任务物体也会显式写入 `condim`、摩擦、`solref`、
`solimp`、`priority` 和 `solmix`，不再依赖当前机器人 MJCF 的默认 class。

MuJoCo MultiCCD 必须保持启用；它为凸网格接触提供多接触点，是当前刚体模型近似有限接触斑
的关键部分。不要用极高滑动摩擦掩盖抓取几何或控制问题。对扁盒应优先增加合理的接触力臂、
拇指/掌面限位和持续预紧，并同时检查 `tail_max_speed` 与
`tail_max_angular_speed`。默认最终阈值分别为 `0.10 m/s` 和 `0.10 rad/s`；专家池角速度
阈值放宽为 `0.20 rad/s`，但通过专家池仍不代表最终验证成功。

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

## 随机摆放可达区

默认任务按物体中心采样，不再把整个桌面或料箱的大部分面积都作为出生区。区域根据当前
RM75B 的 C MuJoCo IK 探针向机械臂侧（较小的全局 X）偏置；自定义
`placement_sampler` 和严格回放使用的 `FixedTablePlacementSampler` 不受影响。

| 任务 | 物体中心范围（米） |
| --- | --- |
| Lift / 通用桌面任务 | 全局 X `[0.49, 0.57]`，Y `[-0.05, 0.05]` |
| Stack | 全局 X `[0.49, 0.57]`，Y `[-0.08, 0.08]` |
| PickPlace 源料箱 | 全局 X `[0.43, 0.48]`，Y `[-0.225, -0.175]` |
| NutAssembly | 全局 X `[0.58, 0.62]`，Y `[-0.085, 0.085]`；朝向限制为 `±20°` |
| Push | 物体 X `[0.47, 0.53]`、Y `[-0.16, -0.10]`；目标 X `[0.50, 0.58]`、Y `[0.10, 0.16]` |

Stack 和 NutAssembly 仍执行完整布局重试与最小中心间距检查。NutAssembly 额外避免物体在
reset 时与固定桩或另一螺母穿插。全机器人 benchmark 的初始 task scene 也从 Lift 的同一
可达区采样；后续 fallback 只会把物体继续向机械臂方向拉近。

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

以下参数适合单张 24 GB GPU。调度器会读取 `CUDA_VISIBLE_DEVICES`、GPU 空闲显存、启动时
利用率和 CPU 核心数；在空闲的 24 GB 3090、`--num-envs 64` 下通常会选择两个对象 worker
和两个通用 GPU 任务槽，使低占用 Ultra、CPU Wrist Lattice 与 PPO 在对象间形成流水线。
实测 64-env PPO 会达到 100% GPU 利用率，因此默认仍只允许一个 PPO；显存紧张或启动时 GPU
已高负载时，通用 GPU 槽也会自动退回一个。显存不足时可把 `--num-envs` 降为 32：

```bash
MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
python -m tools.grasping.batch_grasp_edit \
  --dataset original127 \
  --expect-count 127 \
  --output outputs/dex_hand_ppo127 \
  --ultra-root outputs/dex_hand_ppo127/ultra \
  --lattice-root outputs/dex_hand_ppo127/lattice \
  --device cuda:0 \
  --gpus auto \
  --workers-per-gpu auto \
  --gpu-jobs-per-gpu auto \
  --ppo-jobs-per-gpu auto \
  --num-envs 64 \
  --initial-updates 5 \
  --mid-updates 10 \
  --max-updates 15 \
  --base-candidates 3 \
  --lattice-max-templates 12 \
  --lattice-max-executions 32 \
  2>&1 | tee -a outputs/dex_hand_ppo127/console.log
```

多卡时只需扩大可见设备，例如 `CUDA_VISIBLE_DEVICES=0,1`；`--gpus auto` 会为每张卡建立
独立槽位。自动模式最多给每张卡两个 worker/两个通用 GPU 任务，避免为了追求并发无限增加
进程。`--ppo-jobs-per-gpu auto` 会把满算力 PPO 限制为一路，但不阻止另一个 worker 同时执行
Ultra；只有在实测双 PPO 能提高总吞吐时才建议把它显式改为 `2`。

启动和每个对象完成时都会打印资源计划与动态 ETA：

```text
[resource] gpu=0 workers=2 gpu_jobs=2 ppo_jobs=1 estimated_worker=2.0GiB memory=free=22.0/24.0GiB ...
[estimate] cached=5 pending=122 samples=5 avg=7m07s/object workers=2 gpu_parallelism=1 eta=14h28m
```

启动时 ETA 使用当前签名下已完成对象的实际 `runtime_sec` 均值，并除以真正允许并发的 GPU
PPO 阶段数，所以预热前会偏保守。所有 worker 至少完成一个对象后，ETA 会切换为本轮墙钟
吞吐率，从而把 Ultra/PPO 重叠带来的单卡加速计入估计。前几个对象差异较大时仍只是粗估，
样本增加后会自动收敛。相同信息写入
`summary.json.progress` 和 `summary.json.resource_plan`。

两个 `--train-*-success` 参数用于压力测试：即使 Ultra 或 Lattice 已经成功，只要模板可用，
仍然运行 PPO。正常的分层生产流程应移除这两个参数，让已经解决的对象提前退出。

任务被中断时，使用完全相同的语义参数重新运行。GPU 数量和 worker 数只影响调度，不会让
已经完成且签名匹配的每对象结果失效。

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
  python -m tools.verification.replay_trajectory "$manifest" ||
    echo "REPLAY_FAIL $manifest"
done
```

当前回放成功判据为：最后一帧满足 Lift 任务条件，且末尾最多 20 个控制帧中至少 80% 满足
该条件。Lift 条件是物体底部高于桌面 4 cm。这个返回码不会单独统计接触数量、物体转动、
滑移或几何穿透；需要这些指标时，应另外保存接触和物体姿态时序，不能仅从回放成功推断。

最终报告应同时保存 benchmark commit、MJCF、配置、`summary.json` 和失败对象日志，以便后续
比较手指包络、抓取覆盖率和 C MuJoCo/MJWarp 一致性。
