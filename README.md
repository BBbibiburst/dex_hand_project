# Dex Hand Project

基于 MuJoCo 的 RM75B 机械臂与灵巧手操作项目。目前工作的主线是：为 YCB/EGAD 物体生成
可执行抓取，经过轨迹与完整机械臂 Lift 验证，再将成功执行过程记录为多样化 LeRobot
示教数据。

当前抓取流水线为：

```text
GraspQP 初始姿态
→ MuJoCo evolutionary refinement
→ 无接触 approach / closure / hold 验证
→ 完整 RM75B + Dex Hand Lift 验证
→ 随机化成功示教采集
```

项目还包含 Pika 平行夹爪、触觉阵列、Vive/蓝牙手套遥操作、基础操作任务与 Diffusion
Policy 实验入口，但当前全量生成和自动示教以 `RM75B + dex_hand + lift` 为主。

## 当前状态

- 物体目录：YCB 与 EGAD，共约 127 个对象，实际数量以
  `assets/maniskill/manifest.json` 为准。
- 抓取生成：官方 GraspQP force-closure metric + Dex Hand 闭链联合姿态优化 +
  MuJoCo simulator-in-the-loop 进化优化。
- 安全约束：完整手部网格、桌面硬约束、接近/闭合轨迹碰撞检查。
- 验证分层：direct hold、trajectory hold、完整机器人 Lift 分开记录。
- 整机候选先批量检查 RM75B IK 与桌面碰撞，再运行动态 Lift；最终报告与写入的 grasp
  config 始终对应同一次最佳搜索。
- 完整预设跨独立搜索累积并去重可执行抓取，默认每物体目标 3 个；最多搜索 5 次、保存
  24 个轨迹候选、每次动态测试 6 个候选，并设置 45 分钟物体预算。
- 长任务支持：自动推断并行度、逐物体原子保存、ETA、断点恢复和失败物体定向重试。
- 示教采集：按物体、随机 seed 和抓取候选搜索成功轨迹；每个 seed 最多保存一条，优先保证多样性。

项目仍处于全量验证阶段。`trajectory_stable` 不等于机械臂一定能完成 Lift，最终示教可用性
应结合 `robot_lift_verified` 和随机化采集报告判断。

## 快速开始

### 1. 克隆

```bash
git clone --recurse-submodules <repository-url> dex_hand_project
cd dex_hand_project
```

若已经普通 clone：

```bash
git submodule update --init --recursive
```

### 2. 创建环境

推荐 Python 3.10：

```bash
conda create -n mujoco python=3.10 -y
conda activate mujoco

python -m pip install --upgrade pip
python -m pip install -e ".[dev,assets]"
python -m pip install -e "deps/graspqp/graspqp[lite]" --no-build-isolation
```

采集 LeRobot 数据时再安装：

```bash
python -m pip install -e ".[learning]"
```

遥操作硬件依赖：

```bash
python -m pip install -e ".[hardware]"
```

### 3. 准备物体资产

如果压缩包或服务器目录已经包含 `assets/maniskill`，可以跳过下载。否则运行：

```bash
python tools/download_maniskill_objects.py
```

### 4. 验证部署

```bash
python -m pytest -q
python -m tools.run_smoke_checks --steps 2
```

## 全量抓取生成

首次运行：

```bash
python -m tools.grasping.benchmark_catalog --full-pipeline
```

程序会根据 CPU affinity、可用内存和 cgroup 限制自动选择 1～8 个 worker。也可以显式指定：

```bash
python -m tools.grasping.benchmark_catalog \
  --full-pipeline --jobs 8 --evolution-jobs 1
```

服务器后台运行：

```bash
mkdir -p logs
nohup python -m tools.grasping.benchmark_catalog \
  --full-pipeline \
  > logs/full_pipeline.log 2>&1 &

tail -f logs/full_pipeline.log
```

断电或任务中止后恢复：

```bash
python -m tools.grasping.benchmark_catalog --full-pipeline --resume
```

全量运行前可先执行代表性 pilot。它串行测试盒体、曲面、容器、带柄和不规则物体，
实时打印 Lift 成功率与失败类型；完成至少 4 个物体后，成功率低于 25% 或同一错误连续
出现 3 次会保存报告并提前退出：

```bash
python -m tools.grasping.benchmark_catalog --pilot
```

重新计算未通过完整轨迹或 Robot Lift 的物体（`--full-pipeline` 当前默认采用该语义，
显式参数保留用于普通非预设运行）：

```bash
python -m tools.grasping.benchmark_catalog \
  --full-pipeline --resume --retry-incomplete
```

输出目录：

```text
configs/grasps/dex_hand/graspqp_seeds/          GraspQP 初始配置
configs/grasps/dex_hand/dexevolve/              进化与轨迹验证后的配置
configs/grasps/dex_hand/full_pipeline_benchmark.json
```

详细参数、状态解释和恢复策略见 [抓取流水线](docs/GRASP_PIPELINE.md)。

## 多样化示教采集

全量生成后，采集每个物体最多 20 条成功 Lift 示教：

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

采集器只读取 benchmark 中的 `trajectory_stable` 物体。每个随机 seed 最多保存一条成功
轨迹，并轮换候选尝试顺序，避免相同初始位姿或候选支配数据集。

完整说明见 [示教数据采集](docs/DATA_COLLECTION.md)。

## 结果语义

抓取稳定性不能用一个裸 `stable` 表示：

| 字段或状态 | 含义 |
| --- | --- |
| `direct_hold_stable` | 直接设置最终抓姿后可以保持，只用于候选快速筛选 |
| `trajectory_collision_free` | 无接触 approach 与 closure 满足碰撞约束 |
| `trajectory_hold_stable` | 完整轨迹执行后仍能稳定保持 |
| `trajectory_stable` | benchmark 最终通过完整轨迹验证 |
| `robot_lift_verified` | 完整机械臂策略完成抓取、提升和验证且无桌面碰撞 |
| `strategy_verified_success` | scripted strategy 的完整流程验证成功 |
| `info["task_success"]` | 当前环境状态满足 task 自身成功条件 |

当前 benchmark schema：

```text
schema_version=4
validation_semantics=trajectory-hold-v2
```

旧报告中的 `stable` 是歧义状态，只能作为 `legacy_stable` 显示，不能继续写入新结果。

## 常用调试命令

单物体生成：

```bash
python -m tools.grasping.benchmark_catalog \
  --full-pipeline --object-id ycb:003_cracker_box \
  --output configs/grasps/dex_hand/cracker_test.json
```

验证已有抓取轨迹：

```bash
python -m tools.grasping.validate_grasp \
  configs/grasps/dex_hand/dexevolve/ycb_003_cracker_box.json
```

在 MuJoCo Viewer 中查看 scripted Lift：

```bash
python -m tools.grasping.validate_scripted_strategy \
  --object-id ycb:003_cracker_box \
  --grasp-config configs/grasps/dex_hand/dexevolve/ycb_003_cracker_box.json
```

渲染抓取目录总览：

```bash
MUJOCO_GL=egl python -m tools.grasping.render_grasp_catalog
```

## 文档

- [安装与服务器部署](docs/DEPLOYMENT.md)
- [抓取生成、验证和结果解释](docs/GRASP_PIPELINE.md)
- [多样化示教数据采集](docs/DATA_COLLECTION.md)
- [代码结构与语义约定](docs/ARCHITECTURE.md)

## 目录结构

```text
apps/                    数据采集应用
assets/                  机器人、手、场景和 ManiSkill 物体资产
configs/                 机器人配置及本地抓取结果
deps/graspqp/            固定版本的 GraspQP submodule
docs/                    部署、流水线、采集和架构文档
examples/                Viewer、控制和任务示例
source/                  环境、控制、抓取、策略、传感器和学习代码
tests/                   单元测试、回归测试和语义契约测试
tools/                   抓取、触觉、资产和诊断命令
```

## 开发检查

```bash
python -m ruff check source apps tests examples tools --exclude deps
python -m pytest -q
git diff --check
```

`configs/grasps/`、`datasets/` 和日志属于运行产物，通常不纳入 Git。跨机器迁移时可以直接
复制整个 `configs/grasps/`；配置内的旧绝对 mesh 路径会通过 `assets/...` 自动重定位。
