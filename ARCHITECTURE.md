# 项目架构

## 设计原则

项目代码只保留自身实现，不在仓库中 vendor 第三方源码，也不使用 Git submodule。
Python/CUDA 依赖由 `pyproject.toml` 管理，大型对象资产和运行结果放在 Git 忽略目录中。

依赖方向固定为：

```text
apps / tools / examples
          ↓
       source
          ↓
 assets / configs / Python packages
```

`source/` 不得导入 `apps/`、`tools/`、`examples/` 或测试代码。长运行任务只负责组织参数、
子进程和输出，可复用实现必须位于 `source/`。

入口与模块名称以当前实现为准，不保留只转发到新名称的兼容脚本，也不为测试单独维护
monkeypatch facade；测试直接覆盖实际模块 API。

## 运行层

- `source/robots`：机械臂、底座、末端执行器描述和装配。
- `source/control`：机械臂、手和组合控制器。
- `source/envs`：Gymnasium/MuJoCo 环境、任务、对象和场景绑定。
- `source/ultradexgrasp`：项目原生 Ultra Prior、Dex Hand 可微 surrogate 和 episode contract。
- `source/rl/grasp_edit`：Wrist Lattice、六维手动作和 Hybrid PPO。
- `source/grasp_pipeline`：Ultra/PPO 共享的参考轨迹、结果契约和 C MuJoCo 回放。
- `source/verification`：生成轨迹的严格 C MuJoCo 最终复验。
- `source/sensors`：触觉接口、Dex Hand/Pika 传感器和曲面拟合。
- `source/data`：LeRobot 记录器和数据 schema。
- `source/teleop`：Vive、蓝牙手套、映射、会话和轨迹处理。
- `source/viz`：可视化和报告绘制，不参与核心物理逻辑。

## 主流水线

```text
source/ultradexgrasp
  生成 Ultra Prior episode
          ↓
source/rl/grasp_edit/templates.py
  构建并执行 Wrist Lattice
          ↓
source/rl/grasp_edit/env.py + ppo.py
  MJWarp 并行 Hybrid PPO
          ↓
source/grasp_pipeline/replay.py 或 source/verification/strict_replay.py
  C MuJoCo 权威复验
          ↓
apps/collect_generated_lerobot.py + source/data
  重放最终验证轨迹并写入 LeRobot
          ↓
apps/train_diffusion.py + apps/evaluate_diffusion.py + source/imitation
  训练与评估视觉-触觉 Diffusion Policy
```

对应入口：

- `python -m tools.ultradexgrasp.probe`
- `python -m tools.ultradexgrasp.generate`
- `python -m tools.ultradexgrasp.batch_generate`
- `python -m apps.train_grasp_edit_rl`
- `python -m tools.grasping.batch_grasp_edit`
- `python -m tools.verification.replay_trajectory`
- `python -m apps.collect_generated_lerobot --input-root <pipeline-output>`
- `python -m apps.train_diffusion`
- `python -m apps.evaluate_diffusion`

## 资产与生成数据

- `assets/grippers/dex_hand/dex_hand.xml` 是 Dex Hand 权威物理模型。
- `configs/ultradexgrasp/default.json` 是 Ultra Prior 默认参数。
- `configs/ultradexgrasp/cache/` 保存可重建的手部 surrogate。
- `assets/maniskill/` 保存可重新下载的 YCB、EGAD 和 GSO 资产。
- `outputs/` 保存 Ultra、Lattice、PPO、日志和汇总报告。
- `datasets/`、`recordings/`、`checkpoints/` 保存训练和采集产物。

这些生成目录不作为源代码提交。修改 Dex Hand MJCF 后必须重新标定 surrogate，并使用新的
Ultra、Lattice 和 PPO 输出，避免把旧手模型的轨迹当作当前结果。

## 外部依赖策略

项目不依赖本地参考仓库路径。核心依赖分组如下：

- 默认依赖：MuJoCo、NumPy、SciPy、Gymnasium、Trimesh 和可视化工具。
- `ultradexgrasp`：PyTorch，用于原生可微抓取优化。
- `mjwarp`：MuJoCo Warp 和 Warp，用于 GPU 并行仿真。
- `assets`：ManiSkill 对象资产下载。
- `learning`：LeRobot、TorchVision 和 Diffusion Policy 相关工具。
- `hardware`：Vive、串口和蓝牙设备。
- `dev`：pytest 和 Ruff。

## 架构约束

`tests/test_architecture_boundaries.py` 检查：

- `source/` 不反向依赖入口层。
- `source` 一级包依赖图无环。
- robots/control、sensors/viz 等边界不回退。
- 已迁移的旧单文件实现不会重新出现。
- 旧入口别名和抓取搜索兼容 facade 不会重新出现。
- 仓库中不存在 `deps/` 和 `.gitmodules`。
