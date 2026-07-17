# Dex Hand Project

基于 MuJoCo 的 RM75B 单臂、Dex Hand / Pika 夹爪操作环境，包含视觉触觉示教采集和多模态 Diffusion Policy 模仿学习。

## 功能

- 单臂任务：`lift`、`stack`、`pick_place`、`nut_assembly`、`door`
- 末端执行器：六维 Dex Hand、一维 Pika 平行夹爪
- 触觉：末端执行器注册的 taxel 阵列（Dex Hand、Pika 夹爪等）
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

### 下载 ManiSkill 物体资产

项目提供一个可重复运行的资产工具：通过 ManiSkill 下载全部可用 YCB 真实扫描日常
物体，并从 EGAD 官方资源下载完整的 7×7 评测子集（A0–G6，共 49 个），统一整理供
抓取实验使用。脚本会校验 EGAD 的 49 个 OBJ，不会用其他模型补齐。ManiSkill 版本间
YCB 模型目录可能是 77 或 78 个，因此最终总数以清单记录的实际数量为准，不会静默
删除一个模型来硬凑 126。大型原始缓存和整理后的模型不会提交到 Git。

```powershell
python -m pip install --upgrade mani_skill
python tools/download_maniskill_objects.py
```

YCB 和 EGAD 的规范化来源分别位于 `assets/maniskill/ycb/models/` 和
`assets/maniskill/egad/models/`，统一对象位于 `assets/maniskill/models/`；实际选择
分别写入 `configs/ycb_objects.lock.txt` 和 `configs/egad_objects.lock.txt`，最终来源
和模型文件索引写入 `assets/maniskill/manifest.json`。原始压缩包统一放在
`assets/maniskill/downloads/`。Windows 默认复制文件；Linux 可传
`--mode symlink` 节省空间。已有原始缓存后可用 `--dry-run` 只查看固定选择、不复制或
下载。ManiSkill 文档说明其分发资产采用 CC BY-NC 4.0，商业使用或再分发前应再次核对
各原始数据集许可。

## 自动无渲染测试

每次修改代码后运行一个命令即可检查核心仿真路径：

```powershell
python -m source.demos.smoke_test
```

测试不会打开 MuJoCo Viewer、OpenCV 或 Matplotlib 窗口，默认检查 Dex Hand 与 Pika
模型编译、position/IK 控制、触觉观测与曲面点位、八邻域信号扩散，以及所有已注册操作任务的
reset/step。任一项失败会打印 traceback 并返回非零退出码。它不测试相机、蓝牙手套、Vive、
真实机械臂、数据集或训练流程；这些路径仍需相应硬件或数据做集成测试。

开发时只想快速检查机器人与触觉，不编译任务场景：

```powershell
python -m source.demos.smoke_test --skip-tasks
```

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

首次测试手套时，建议先运行不依赖 MuJoCo、Vive 和机械臂的独立诊断：

```powershell
python -m source.teleop.bluetooth_glove.bluetooth_glove_test

# Vive 控制手的六维位姿，蓝牙手套控制五指弯曲
python -m source.teleop.vive.vive_glove_hand_control
```

程序会引导完成系统配对确认、握拳/张手校准，只采集一次全张手基线，再把拇指、食指、
中指、无名指、小指和完整握拳拆成六个确认姿势逐项测试，输出每项的目标通道变化、
其他通道变化与接收频率。
最后两路均为同一个拇指传感器，这是五维手套映射到
Dex Hand 六维动作的预期行为。配对 PIN 为 `1234`。程序默认使用实机验证过的
PyBluez RFCOMM，并在退出时执行 `shutdown` 和 `close`；Windows COM 自动发现仍可作为
备用，因此重新配对导致 COM 号改变也无需修改配置。默认 MAC、PIN、传输方式、波特率和校准时长配置在
[configs/teleop.json](configs/teleop.json)，命令行参数仍可临时覆盖。

独立测试全部通过后，会把五路原始张手/握拳校准边界自动写回同一配置。正式遥操作会
直接加载它们，不再重复要求握拳和张手；需要重新校准时再次运行测试即可，临时测试但
不覆盖配置可传入 `--no-save`。

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
- `observation.tactile`：当前末端执行器的触觉数组
- `observation.state`：完整 `qpos + qvel + ctrl`
- `observation.operator.glove`：六维原始手套值
- `observation.operator.vive_pose`：原始 Vive `[xyz, quaternion_wxyz]`
- `action`：实际下发的末端目标位姿与手部动作
- `task`：任务文本

## 自动触点检测

通用检测程序会逐个沿 taxel 法向放置探针，依次检查瞬时压力、持续压力、邻点串扰和释放残留；没有底层碰撞面的裁角点会记录为 `inactive`。完整检测 Pika 夹爪：

```powershell
python -m source.demos.tactile_contact_validation `
  --robot-config configs/robot_profiles/rm75b_pika_gripper.json
```

也可以分 patch 或分段检测：

```powershell
python -m source.demos.tactile_contact_validation `
  --robot-config configs/robot_profiles/rm75b_pika_gripper.json `
  --patch left `
  --start-index 0 `
  --max-taxels 100
```

程序默认只打开按真实坐标绘制的 3D 触点云，不保存文件：颜色表示 `pass`、`inactive`、`no_response`、`unstable_hold`、`crosstalk` 或 `release_residual`，点大小表示瞬时压力。需要图片时显式传入 `--plot tactile_validation.png`；无界面批处理或 CI 使用 `--no-show`；只有需要原始指标时才传入 `--csv tactile_validation.csv`。出现实际失败时返回非零退出码，可直接接入 CI 或硬件验收流程。

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
