# Dex Hand Project

基于 MuJoCo 的 RM75B 机械臂与六驱动欠驱动灵巧手项目。当前主线是使用项目原生实现完成
GraspQP + DexEvolve 抓取生成、Wrist Lattice 可达姿态扩展、MJWarp 并行 PPO 优化，以及最终的
C MuJoCo 物理复验。

运行依赖统一通过 `pyproject.toml` 安装；`third_party/DexEvolve` 仅作为官方算法参考
子模块，项目运行实现位于 `source/grasping/`，不会直接依赖 Isaac Lab 源码。

## 核心流程

```text
Dex Hand MJCF 与六驱动肌腱标定
→ GraspQP + DexEvolve 优化物体相对手掌位姿和六个手驱动
→ RM75B IK 与 Wrist Lattice 生成可达抓取轨迹
→ MJWarp Hybrid PPO 选择腕部模板并优化六维手部动作
→ C MuJoCo 回放验证末段持续抬升及其与权威模型的一致性
```

完整本地物体目录包含 1160 个 Lift 对象：78 个 YCB、49 个 EGAD 和 1033 个 GSO；历史
`original127` 基准固定为前两者。实际对象列表以各数据集 manifest 为准。

物体先按数据集单位恢复物理尺度，再统一限制到最大水平直径 75 mm。全新 clone 先执行
第3节的 YCB/EGAD 下载，再用仓库锁定的 v2 清单恢复当前 Top100 所需的 GSO：

```bash
python -m tools.download_gso_objects \
  --selection configs/underactuated_top100_v2.json \
  --workers 12
```

只有重新研究完整母池时才需要下载全部1033个 GSO：

```bash
python -m tools.download_gso_objects --all --workers 12
```

可以生成几何排序作为预筛，但它不代表已经通过抓取：

```bash
python -m tools.rank_underactuated_candidates \
  --count 100 \
  --minimum-prior 0.50

python -m tools.render_object_catalog \
  --selection configs/underactuated_top100_v2.json \
  --output-dir outputs/object_catalog/top100_v2 \
  --include-collision \
  --collision-mode multi
```

该清单按 GSO/YCB/EGAD 数据集配额、抓取形状族和近重复约束生成，只是物理测试候选。
最终 100 物体列表仍必须结合 GraspQP + DexEvolve 全量结果、手指接触增长率、扰动
鲁棒性和 C MuJoCo 抬升/保持复验生成。

## 快速开始

### 1. 获取项目

```bash
git clone --recurse-submodules <repository-url> dex_hand_project
cd dex_hand_project
```

已有检出目录使用 `git submodule update --init --recursive` 初始化官方 DexEvolve 参考实现。

### 2. 创建环境

推荐 Python 3.10 和支持 CUDA 的 PyTorch 环境：

```bash
conda create -n dex-hand python=3.10 -y
conda activate dex-hand

python -m pip install --upgrade pip
python -m pip install -e ".[dev,assets,grasping,mjwarp]"
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
python -m tools.download_gso_objects \
  --selection configs/underactuated_top100_v2.json \
  --workers 12
```

### 4. 检查运行环境

下面的探针同时验证 CUDA、原生 GraspQP + DexEvolve、Dex Hand surrogate 和 MJWarp：

```bash
python -m tools.grasp_generation.probe \
  --strict \
  --mjwarp \
  --device cuda:0
```

每次修改 `assets/grippers/dex_hand/dex_hand.xml` 后，应重新标定 surrogate：

```bash
python -m tools.grasp_generation.probe \
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

## 单物体 GraspQP + DexEvolve

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -m tools.grasp_generation.graspqp_evolve \
  --object-id ycb:002_master_chef_can \
  --output outputs/grasp_generation/master_chef_can \
  --device cuda:0 --seed 11

MUJOCO_GL=glfw python -m tools.grasp_generation.visualize_episode \
  --manifest outputs/grasp_generation/master_chef_can/manifest.json \
  --play --viewer-speed 0.4 --loop
```

该路线用 GraspQP 细化解析候选，以适配六驱动欠驱动闭链，再由 DexEvolve 搜索并在
MJWarp 中批量执行扰动评价；最终结果必须通过 C MuJoCo 的持续拇指—对侧 skin 接触复验。
默认正式预算为 GraspQP 64 个种子、每个150步、执行12个候选；DexEvolve 使用24个初始
个体、每代12个子代、16代（共216次评价），并向 Wrist Lattice 导出 Top 6。

## GraspQP + DexEvolve + Wrist Lattice + MJWarp PPO

先用少量代表性对象检查完整三阶段流程：

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python -m tools.grasping.batch_grasp_edit \
  --object-id ycb:002_master_chef_can \
  --object-id ycb:003_cracker_box \
  --object-id ycb:024_bowl \
  --object-id ycb:025_mug \
  --object-id egad:C0 \
  --output outputs/dex_hand_pilot \
  --grasp-root outputs/dex_hand_pilot/grasp \
  --lattice-root outputs/dex_hand_pilot/lattice \
  --device cuda:0 \
  --num-envs 64 \
  --initial-updates 2 \
  --mid-updates 4 \
  --max-updates 6 \
  --train-lattice-success
```

当前 Top100 v2 全量正式生成（已剔除上一轮 A/B 失败并补入通过初始稳态检查的对象）：

```bash
MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
python -m tools.grasping.batch_grasp_edit \
  --selection configs/underactuated_top100_v2.json \
  --expect-count 100 \
  --output outputs/dex_hand_top100_v2 \
  --grasp-root outputs/dex_hand_top100_v2/grasp \
  --lattice-root outputs/dex_hand_top100_v2/lattice \
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
  --lattice-max-executions 32
```

自动调度会根据可见 GPU、空闲显存、启动利用率、环境数和 CPU 核数，为单卡或多卡分配对象
worker；24 GB GPU 配合 64 个环境时通常采用同卡双流水线，让 Grasp/CPU 工作与单路满载 PPO
重叠，并输出动态 ETA。详细资源规则见 `docs/PIPELINE.md`。

这一条命令就是完整正式流程：先生成 GraspQP + DexEvolve Top-K 候选并展开默认 65 mm
Wrist Lattice；成功便退出。失败对象会自动改用独立缓存的 85 mm 恢复轨迹；恢复 Lattice
仍失败才以 0.20 的保守手部编辑范围进入 5→10→15 更新的 MJWarp PPO。无需再维护失败清单
或第二条重跑命令。中断后使用相同参数重新运行即可继续。

详细的状态解释、缓存规则和 C MuJoCo 复验方法见
[全量流水线与验证](docs/PIPELINE.md)。

## 输出目录

```text
outputs/dex_hand_top100_v2/
├── objects/             每个对象的可恢复结果
├── logs/                每个对象的完整日志
├── lattice/             Wrist Lattice 轨迹与 index.json
│   └── recovery_lift_085mm/  仅失败对象的恢复轨迹
├── rl/                  PPO 配置、checkpoint、metrics 和最佳轨迹
├── grasp/               本次运行生成的 GraspQP + DexEvolve
├── summary.csv
└── summary.json

```

`outputs/`、数据集、checkpoint 和生成缓存默认不纳入 Git。

## 其他功能

- `apps/collect_generated_lerobot.py`：把最终验证通过的 Grasp/Lattice/PPO 轨迹重放为 LeRobot 自动示教数据。
- `apps/collect_teleop_lerobot.py`：遥操作示教采集。
- `source/imitation/`：Diffusion Policy 训练和评估。
- `source/sensors/tactile/`：Dex Hand 与 Pika 的触觉建模和标定。

自动示教数据生成示例：

```bash
python -m apps.collect_generated_lerobot \
  --input-root outputs/dex_hand_top100_v2 \
  --output datasets/grasp_lerobot \
  --repo-id local/dex-hand-grasp-demonstrations
```

## 项目结构

```text
apps/                    长运行训练、采集和目录任务入口
assets/                  MJCF、mesh、场景、机器人和对象资产
configs/                 机器人、遥操作和 GraspQP + DexEvolve 配置
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
