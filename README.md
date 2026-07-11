# Dex Hand Project

基于 MuJoCo 的 RM75B 单臂、Dex Hand / Pika 夹爪操作环境，包含视觉触觉示教采集和多模态 Diffusion Policy 模仿学习。

## 功能

- 单臂任务：`lift`、`stack`、`pick_place`、`nut_assembly`、`door`
- 末端执行器：六维 Dex Hand、一维 Pika 平行夹爪
- 触觉：Dex Hand taxel 阵列
- 遥操作：Vive 六维位姿 + 六维拉伸手套
- 数据：LeRobotDataset v3（Parquet + MP4）
- 模仿学习：RGB、触觉、机械状态融合的条件扩散动作策略
- 验证：策略在任务环境中闭环 rollout，以 `task_success` 统计成功率

## 环境

建议使用 Python 3.10，并安装：

```powershell
pip install mujoco gymnasium numpy scipy torch torchvision "lerobot>=0.4"
```

当前机器人由 [configs/current_robot.json](configs/current_robot.json) 选择。

## 查看任务

```powershell
python -m source.demos.manipulation_task_playback --task lift
python -m source.demos.manipulation_task_playback --task stack
python -m source.demos.manipulation_task_playback --task pick_place
python -m source.demos.manipulation_task_playback --task nut_assembly
python -m source.demos.manipulation_task_playback --task door
```

## 收集示教

正弦假设备测试：

```powershell
python -m source.demos.collect_teleop_lerobot `
  --task lift `
  --device sine `
  --episodes 20 `
  --output datasets/lift
```

Viewer 操作：

| 按键    | 功能                    |
| ------- | ----------------------- |
| `Space` | 开始/暂停记录帧         |
| `C`     | 校准 Vive 相对零位      |
| `N`     | 确认并保存当前 episode  |
| `R`     | 丢弃当前 episode 并重置 |
| `Q`     | 退出；未确认帧会被丢弃  |

真实设备接入后使用 `--device hardware`。厂商 API 的适配点位于 [devices.py](source/teleop/devices.py)。设备驱动只需实现 `connect()`、`read()` 和 `close()`。

经典蓝牙手套与 Vive OpenVR 已接入：

```powershell
pip install openvr
pip install git+https://github.com/pybluez/pybluez.git#egg=pybluez
python -m source.demos.collect_teleop_lerobot `
  --task lift `
  --device hardware `
  --glove-mac 20:20:11:11:16:22 `
  --vive-serial YOUR_TRACKER_SERIAL `
  --dry-run
```

启动时按提示先握拳、再张开手完成手套标定。手套的 5 路顺序按
`拇指、食指、中指、无名指、小指` 处理；Dex Hand 有 6 个执行器，因此同一个拇指值同时
驱动拇指旋转与抓握，无法用这款手套独立控制这两个自由度。当前 Vive 驱动读取 OpenVR，
使用 PC 接收器时需要先启动 VIVE Hub 和 SteamVR。

每帧记录：

- `observation.images.agentview`：RGB
- `observation.tactile`：Dex Hand 触觉数组
- `observation.state`：完整 `qpos + qvel + ctrl`
- `observation.operator.glove`：六维原始手套值
- `observation.operator.vive_pose`：原始 Vive `[xyz, quaternion_wxyz]`
- `action`：实际下发的末端目标位姿与手部动作
- `task`：任务文本

## Diffusion Policy

策略以当前 RGB、触觉和机械状态为条件，生成未来 `horizon` 步动作：

```text
RGB ───── CNN ─────────┐
触觉 ─── MLP ─────────┼─ 融合条件 ─ 条件 DDPM ─ 未来动作序列
机械状态 ─ MLP ───────┘
```

训练时给动作序列加入不同时间步的高斯噪声，网络预测噪声；推理时从随机噪声迭代去噪。策略一次预测动作块，但闭环只执行前 `action_steps` 步，然后重新读取视觉、触觉和状态再规划。

### 训练

```powershell
python -m source.imitation.train_diffusion `
  --dataset datasets/lift `
  --repo-id local/dex-hand-demonstrations `
  --output checkpoints/lift_diffusion.pt `
  --horizon 16 `
  --epochs 100 `
  --batch-size 32
```

训练程序会：

1. 用 LeRobot 的 `delta_timestamps` 构造未来动作序列；
2. 划分训练/离线验证帧；
3. 计算 state、tactile、action 归一化统计；
4. 按 validation diffusion loss 保存最佳 checkpoint。

离线 loss 用于选模型，但不代表任务完成能力。

### 闭环任务验证

```powershell
python -m source.imitation.evaluate_diffusion `
  --checkpoint checkpoints/lift_diffusion.pt `
  --task lift `
  --episodes 20 `
  --max-steps 500 `
  --action-steps 4
```

输出每个 episode 的：

- `success`：环境的 `task_success`
- return
- 完成或终止步数

最终汇总任务成功率、平均回报和平均步数。验证环境必须使用与训练数据相同的机器人、末端执行器、触觉配置、相机分辨率以及动作维度。

## 目录

```text
assets/                         本地 MuJoCo 资产
configs/                        机器人配置
source/control/                 机械臂 IK 与末端执行器控制
source/envs/manipulation/       操作任务
source/sensors/tactile/         触觉模型
source/teleop/                  设备接口、映射、LeRobot 写入
source/imitation/               Diffusion Policy、训练和闭环验证
source/demos/                   Viewer 与示教采集程序
datasets/                       本地示教数据（Git 忽略）
checkpoints/                    训练 checkpoint
```

## 注意

- `--device sine` 是测试输入，不是实际设备控制。
- 正式数据只在 viewer 显示 `REC` 时写入，并需按 `N` 确认 episode。
- 不同任务的 MuJoCo `qpos/qvel` 维度可能不同，建议每个任务分别训练 checkpoint。
- 目前视觉输入为 `agentview` 单相机 RGB；可以按相同 feature 约定扩展多相机或深度图。
