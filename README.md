# Dex Hand Project

基于 MuJoCo 的 RM75B 机械臂与灵巧手操作项目。当前主线是独立的
UltraDexGrasp 风格抓取示教生成：直接优化物体相对手掌位姿与 Dex Hand 六个物理驱动，
再由完整 RM75B 场景执行并记录成功 Lift episode。

当前默认抓取流水线为：

```text
MuJoCo 闭链手模型标定
→ UltraDexGrasp/BODex 风格批量可微抓取优化
→ RM75B 实际 IK 可达性排序
→ 同步接近与闭合
→ 抬升和保持验证
→ manifest.json + episode.npz
```

旧 GraspQP + DexEvolve 实现仍保留为回退和结果对照，但新流水线不读取旧 grasp config、
不调用旧搜索模块，也不输出旧 schema。

项目还包含 Pika 平行夹爪、触觉阵列、Vive/蓝牙手套遥操作、基础操作任务与 Diffusion
Policy 实验入口，但当前全量生成和自动示教以 `RM75B + dex_hand + lift` 为主。

## 当前状态

- 物体目录：YCB 与 EGAD，共约 127 个对象，实际数量以
  `assets/maniskill/manifest.json` 为准。
- 手模型：从权威 MuJoCo 闭链模型标定六驱动可微曲面代理，当前 RMS 约 0.7 mm。
- 物体模型：使用与 MuJoCo 单 mesh geom 一致的凸碰撞壳，不把视觉凹腔误当成可进入空间。
- 抓取生成：批量优化 6D 手根位姿、六个驱动、接触与摩擦锥力闭合。
- 安全约束：最坏点穿透、桌面间隙、五指接触分离和接近轨迹联合约束。
- 整机验证：使用 RM75B 实际 IK 排序候选，执行张开、接近、同步闭合、抬升和保持。
- 输出：独立 episode contract，包含控制、状态、物体位姿、阶段、奖励、成功和接触诊断。

新流水线已经完成单物体端到端验证，仍需开展全目录统计。下文中的
`trajectory_stable`、`robot_lift_verified` 等字段仅属于保留的旧流水线；新输出直接使用
episode 的 `success`、`terminal_stage` 和逐帧 `task_success`。

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
python -m pip install -e ".[dev,assets,ultradexgrasp]"
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

## UltraDexGrasp 示教生成

检查固定版本的上游参考仓库与本机 CUDA/Torch 环境：

```bash
python -m tools.ultradexgrasp.bootstrap --clone-missing
python -m tools.ultradexgrasp.probe --strict
```

为单个物体生成全新抓取并执行 Lift：

```bash
MUJOCO_GL=egl python -m tools.ultradexgrasp.generate \
  --object-id ycb:002_master_chef_can \
  --seed 11 \
  --output outputs/ultradexgrasp/ycb_002_master_chef_can/seed_0011
```

只生成物体相对抓取候选、不启动机械臂：

```bash
python -m tools.ultradexgrasp.generate \
  --object-id ycb:024_bowl --synthesis-only
```

成功目录包含 `manifest.json`、`episode.npz`、`candidates.json` 和运行摘要。详细设计、
环境构建及坐标约定见 [UltraDexGrasp 新流水线](docs/ULTRADEXGRASP.md)。

## 旧全量抓取生成（回退）

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

采集器只读取同时满足 `trajectory_stable` 和 `robot_lift_verified=true` 的物体，并从
benchmark 记录的成功 `task_scene` 开始执行。每个 seed 最多保存一条成功轨迹，并轮换候选
尝试顺序。

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
- [UltraDexGrasp 新流水线](docs/ULTRADEXGRASP.md)
- [抓取生成、验证和结果解释](docs/GRASP_PIPELINE.md)
- [多样化示教数据采集](docs/DATA_COLLECTION.md)
- [代码结构与语义约定](docs/ARCHITECTURE.md)

## 目录结构

```text
apps/                    数据采集应用
assets/                  机器人、手、场景和 ManiSkill 物体资产
configs/                 机器人配置及本地抓取结果
deps/graspqp/            固定版本的 GraspQP submodule
deps/ultradexgrasp/      固定版本的 UltraDexGrasp/BODex 参考仓库
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
