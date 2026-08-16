# Dex Hand Project

基于 MuJoCo 的 RM75B 机械臂与六驱动欠驱动灵巧手项目。当前主线是使用项目原生实现完成
Ultra Prior 抓取生成、Wrist Lattice 可达姿态扩展、MJWarp 并行 PPO 优化，以及最终的
C MuJoCo 物理复验。

本项目不再保存第三方源码副本、Git submodule 或 `deps/` 目录。运行依赖统一通过
`pyproject.toml` 安装，Ultra Prior 的实现完整位于 `source/ultradexgrasp/`。

## 核心流程

```text
Dex Hand MJCF 与六驱动肌腱标定
→ Ultra Prior 优化物体相对手掌位姿和六个手驱动
→ RM75B IK 与 Wrist Lattice 生成可达抓取轨迹
→ MJWarp Hybrid PPO 选择腕部模板并优化六维手部动作
→ C MuJoCo 回放验证末段持续抬升及其与权威模型的一致性
```

当前物体目录包含 127 个 Lift 对象：78 个 YCB 和 49 个 EGAD。实际对象列表以
`assets/maniskill/manifest.json` 为准。

## 快速开始

### 1. 获取项目

```bash
git clone <repository-url> dex_hand_project
cd dex_hand_project
```

项目没有 submodule，不需要执行额外的第三方仓库初始化。

### 2. 创建环境

推荐 Python 3.10 和支持 CUDA 的 PyTorch 环境：

```bash
conda create -n dex-hand python=3.10 -y
conda activate dex-hand

python -m pip install --upgrade pip
python -m pip install -e ".[dev,assets,ultradexgrasp,mjwarp]"
```

按需安装其他功能：

```bash
# Diffusion Policy、LeRobot 数据集和学习工具
python -m pip install -e ".[learning]"

# Vive、串口和蓝牙手套
python -m pip install -e ".[hardware]"
```

### 3. 准备对象资产

如果 `assets/maniskill/manifest.json` 和对应 mesh 已存在，可跳过下载：

```bash
python -m tools.download_maniskill_objects
```

### 4. 检查运行环境

下面的探针同时验证 CUDA、原生 Ultra Prior、Dex Hand surrogate 和 MJWarp：

```bash
python -m tools.ultradexgrasp.probe \
  --strict \
  --mjwarp \
  --device cuda:0
```

每次修改 `assets/grippers/dex_hand/dex_hand.xml` 后，应重新标定 surrogate：

```bash
python -m tools.ultradexgrasp.probe \
  --strict \
  --mjwarp \
  --device cuda:0 \
  --recalibrate-surrogate
```

## 验证 Dex Hand 建模

```bash
python -m pytest -q tests/test_dex_hand_mjcf.py
python -m tools.grasping.benchmark_hand_physics --steps 4000
```

该检查覆盖六驱动肌腱结构、自适应弯曲、碰撞几何、食指与拇指最大闭合，以及从最大闭合
释放回零的能力。

## 单物体 Ultra Prior

```bash
MUJOCO_GL=egl python -m tools.ultradexgrasp.generate \
  --object-id ycb:002_master_chef_can \
  --seed 11 \
  --output outputs/ultradexgrasp/ycb_002_master_chef_can/seed_0011
```

成功目录包含 `manifest.json`、`episode.npz`、`candidates.json` 和 `run.json`。

## Ultra Prior + Wrist Lattice + MJWarp PPO

先用少量代表性对象检查完整三阶段流程：

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
  --train-lattice-success
```

127 对象全量压力测试：

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
  --train-ultra-success \
  --train-lattice-success
```

自动调度会根据可见 GPU、空闲显存、启动利用率、环境数和 CPU 核数，为单卡或多卡分配对象
worker；24 GB GPU 配合 64 个环境时通常采用同卡双流水线，让 Ultra/CPU 工作与单路满载 PPO
重叠，并输出动态 ETA。详细资源规则见 `docs/PIPELINE.md`。

移除最后两个 `--train-*-success` 参数后，流水线会在 Ultra 或 Lattice 已成功时提前停止，
只把尚未解决的对象交给 PPO。中断后使用相同参数重新运行即可继续。

详细的状态解释、缓存规则和 C MuJoCo 复验方法见
[全量流水线与验证](docs/PIPELINE.md)。

### GraspM3-lite 单物体多抓取模式验证

需要对一个物体跑“Ultra Prior → Wrist Lattice → 低维时序 CEM/MJWarp → C MuJoCo”闭环时，
使用：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -m apps.run_graspm3_lite_single \
  --object-id ycb:005_tomato_soup_can \
  --output-root outputs/graspm3_lite_single \
  --template-root outputs/graspm3_lite_single/lattice \
  --ultra-root outputs/graspm3_lite_single/ultra \
  --population-size 64 --iterations 4 --grasp-modes all --device cuda:0
```

`all` 会尝试八个抓取 family：Power Wrap、Pinch、Tripod、Spherical、Hook、Cradle、
Lateral 和 Table-assisted。它们是候选 seed/prior，仍然只输出六个欠驱动 actuator 控制，
不是八个独立策略。最终是否成功只看 C MuJoCo 的持续 lift/hold；桌面香蕉需要额外的推、滚或
翻转物体阶段，`TABLE_ASSISTED_CANDIDATE_ONLY` 不等于已验证成功。重跑同一对象目录时使用
`--overwrite-output`，避免复用旧候选。

## 输出目录

```text
outputs/dex_hand_ppo127/
├── objects/             每个对象的可恢复结果
├── logs/                每个对象的完整日志
├── lattice/             Wrist Lattice 轨迹与 index.json
├── rl/                  PPO 配置、checkpoint、metrics 和最佳轨迹
├── ultra/               本次运行生成的 Ultra Prior
├── summary.csv
└── summary.json

```

`outputs/`、数据集、checkpoint 和生成缓存默认不纳入 Git。

## 其他功能

- `apps/collect_scripted_lerobot.py`：自动策略示教采集。
- `apps/collect_teleop_lerobot.py`：遥操作示教采集。
- `source/imitation/`：Diffusion Policy 训练和评估。
- `source/sensors/tactile/`：Dex Hand 与 Pika 的触觉建模和标定。
- `tools/grasping/benchmark_catalog.py`：CPU 抓取搜索和物理 benchmark。
- `tools/grasping/validate_scripted_strategy.py`：在完整机械臂场景中验证策略。

## 项目结构

```text
apps/                    长运行训练、采集和目录任务入口
assets/                  MJCF、mesh、场景、机器人和对象资产
configs/                 机器人、遥操作和 Ultra Prior 配置
docs/                    当前流水线与验证文档
examples/                Viewer、IK 和控制示例
source/                  可复用的环境、控制、抓取、RL、传感器代码
tests/                   单元测试、物理回归和架构边界测试
tools/                   诊断、资产处理、抓取和验证命令
```

模块边界与依赖方向见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 开发检查

```bash
python -m ruff check source apps tests examples tools
python -m pytest -q
git diff --check
```
