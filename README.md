# Dex Hand Project

基于 MuJoCo 的 RM75B 单臂、Dex Hand / Pika 夹爪操作环境，包含视觉触觉示教采集和多模态 Diffusion Policy 模仿学习。

## 功能

- 单臂任务：`lift`、`stack`、`pick_place`、`nut_assembly`、`push`
- 末端执行器：六维 Dex Hand、一维 Pika 平行夹爪
- 触觉：末端执行器注册的 taxel 阵列（Dex Hand、Pika 夹爪等）
- 遥操作：Vive 六维位姿 + 五路蓝牙弯曲传感器（映射为六维手部控制）
- 数据：LeRobotDataset v3（Parquet + MP4）
- 模仿学习：RGB、触觉、机械状态融合的条件扩散动作策略
- 验证：策略在任务环境中闭环 rollout，以 `task_success` 统计成功率

## 环境

建议使用 Python 3.10。完整开发环境直接安装：

Linux 上的 PyBluez 需要系统提供 BlueZ 的头文件。Ubuntu / Debian 请先安装：

```bash
sudo apt-get update
sudo apt-get install -y libbluetooth-dev
```

Fedora / RHEL 使用 `sudo dnf install bluez-libs-devel`。不需要蓝牙手套时，可以只安装
下方的仿真依赖，避免构建 PyBluez。

```powershell
python -m pip install -r requirements.txt
```

也可以按功能安装当前项目：

```powershell
# 仿真、任务、可视化
python -m pip install -e .

# 追加模仿学习、硬件或资产下载支持
python -m pip install -e ".[learning,hardware,assets,dev]"
```

当前机器人由 [configs/current_robot.json](configs/current_robot.json) 选择。`lift`、
`pick_place`、`stack`和`push`依赖整理后的 YCB/EGAD 清单，首次运行这些任务前需要执行
下一节的资产下载命令；`nut_assembly`使用程序化物体，不需要下载数据集。

### 下载 ManiSkill 物体资产

项目提供一个可重复运行的资产工具：通过 ManiSkill 下载全部可用 YCB 真实扫描日常
物体，并从 EGAD 官方资源下载完整的 7×7 评测子集（A0–G6，共 49 个），统一整理供
抓取实验使用。脚本会校验 EGAD 的 49 个 OBJ，不会用其他模型补齐。ManiSkill 版本间
YCB 模型目录可能是 77 或 78 个，因此最终总数以清单记录的实际数量为准，不会静默
删除一个模型来硬凑 126。大型原始缓存和整理后的模型不会提交到 Git。

```powershell
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

## 可运行程序总览

所有命令均在项目根目录执行。模块形式的程序可先运行
`python -m <模块名> --help` 查看当前版本的完整参数；直接位于 `tools/` 下的程序使用
`python tools/<文件名>.py --help`。带 Viewer 或 Matplotlib 窗口的程序需要图形桌面。

### 抓取搜索、验证与 Lift 策略

这组程序构成当前推荐的抓取工作流：

```text
物体 STL/OBJ
  → search_mesh_force_closure（搜索手腕位姿、手指关节和接触）
  → validate_standalone_grasp（只加载手和物体做稳定性验证）
  → validate_scripted_strategy（在完整机械臂环境中逐阶段验证）
  → collect_scripted_lerobot（无 Viewer 批量收集成功轨迹）
```

#### `source.demos.search_mesh_force_closure`

只使用物体 mesh 和 `assets/grippers/dex_hand/` 中的灵巧手模型搜索抓取，不加载机械臂、
桌子或完整任务环境。程序会将物体放入手掌工作空间，联合搜索手腕相对位姿、手指闭合量
和接触点，并对穿透、接触分布及近似力闭合质量评分。

```bash
# 使用资产清单中的物体
python -m source.demos.search_mesh_force_closure \
  --object-id ycb:005_tomato_soup_can \
  --output configs/grasps/ycb_005_tomato_soup_can.json \
  --preview-image configs/grasps/ycb_005_tomato_soup_can.png

# 直接使用一个 STL/OBJ
python -m source.demos.search_mesh_force_closure \
  --mesh path/to/object.stl \
  --target-size 0.09 \
  --output configs/grasps/custom.json \
  --viewer
```

主要参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--object-id ID` | Lift 默认物体 | 从项目物体资产清单选择 mesh；不能和 `--mesh` 同时使用 |
| `--mesh PATH` | 无 | 直接指定 STL/OBJ 等 Trimesh 支持的文件 |
| `--points N` | `2048` | 从物体表面采样的点数；增大可提高几何分辨率但会变慢 |
| `--joint-candidates N` | `128` | 搜索的手型候选数 |
| `--seed N` | `0` | 随机种子，用于复现实验 |
| `--target-size M` | `0.09` | 将物体最长边归一化到该尺寸，单位为米 |
| `--output PATH` | `configs/grasps/<object_id>.json` | 输出正式抓取配置，供验证程序和 Lift 策略读取 |
| `--preview PATH` | 无 | 用 Trimesh 导出物体、手和接触点组成的 3D 场景，如 `.glb` |
| `--preview-image PATH` | 无 | 保存点云、接触点和法向量的 PNG 预览图 |
| `--viewer` | 关闭 | 打开交互式 Matplotlib 3D 窗口查看搜索结果 |

几何评分只是候选生成器，不能代替动力学验证；搜索结果至少应通过下一程序验证。

程序化调用不要导入 demo，使用无窗口 API：

```python
from source.grasping import generate_grasp_config, plan_approach_path

config_path = generate_grasp_config("ycb:002_master_chef_can")
```

`source.grasping.grasp_config_search`负责物体 mesh、抓取搜索和版本化配置，
`source.grasping.approach_path_search`负责完整手点云的无碰撞 waypoint 搜索；两者均不会
加载 Matplotlib。只有独立运行 `source.demos.search_mesh_force_closure` 时才进行轨迹、
接触点和手模型可视化。

#### `source.demos.validate_standalone_grasp`

加载 Dex Hand XML、抓取 JSON 对应的物体 mesh 和自由物体关节，不加载机械臂、桌子及任务
场景。程序先执行抓取手型和预紧，再观测接触保持、物体位移与转动，最后打印
`stable=True/False`。适合快速排除穿透严重、没有夹紧或一受力就脱落的候选。

```bash
python -m source.demos.validate_standalone_grasp \
  configs/grasps/ycb_005_tomato_soup_can.json \
  --seconds 3 \
  --grip-preload 0.35

python -m source.demos.validate_standalone_grasp \
  configs/grasps/ycb_005_tomato_soup_can.json \
  --viewer --viewer-speed 1
```

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `grasp` | 必填 | `search_mesh_force_closure` 生成的 JSON |
| `--seconds S` | `3.0` | 施加抓取后的仿真观察时间 |
| `--grip-preload X` | `0.25` | 在优化手型基础上增加的闭合预紧量 |
| `--viewer` | 关闭 | 打开交互式 Viewer |
| `--viewer-speed X` | `1.0` | Viewer 播放倍率；`1` 表示与真实时间一致 |

#### `source.demos.validate_scripted_strategy`

在完整任务环境中运行脚本策略，但不记录数据。Lift 策略按
`approach → descend → adjust → make_gripper_hand_form → grasp → lift → check`
执行；Viewer 中会显示末端目标位姿、旋转轴、抓取中点和手部指令。每个阶段结束后暂停，
按空格确认进入下一阶段，按 `Q` 退出。机械臂目标使用限速插值，不会瞬间跳变。
策略会优先读取 `configs/grasps/<object_id>.json`；如果当前物体尚无配置，会自动调用
点云抓取搜索一次并缓存配置，后续验证和数据收集直接复用，不会重复搜索。

```bash
python -m source.demos.validate_scripted_strategy --task lift
python -m source.demos.validate_scripted_strategy \
  --task lift --seed 3 --max-steps 900 --viewer-speed 0.5
```

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--task NAME` | `lift` | 选择已注册的脚本策略 |
| `--seed N` | `0` | 环境和物体随机种子 |
| `--max-steps N` | `900` | 单次验证允许的最大控制步数 |
| `--fps N` | `20` | 控制循环刷新频率 |
| `--viewer-speed X` | `1.0` | Viewer 播放倍率；`1` 表示与真实时间一致，可传 `0.5` 半速观察 |
| `--robot-config PATH` | 当前机器人配置 | 覆盖 `configs/current_robot.json` |
| `--arm-name/--hand-name/--base-name` | 配置值 | 临时覆盖机器人组件 |

#### `source.demos.collect_scripted_lerobot`

无 Viewer 的批量数据收集入口。它重复运行与验证程序相同的脚本策略，只将成功轨迹写入
LeRobotDataset；验证视觉效果请使用上一程序。默认失败轨迹会丢弃，不计入
`--episodes`。

```bash
# 先跑一条策略但不创建数据集
python -m source.demos.collect_scripted_lerobot \
  --task lift --episodes 1 --max-steps 500 --dry-run

# 正式收集 20 条成功轨迹
python -m source.demos.collect_scripted_lerobot \
  --task lift \
  --repo-id local/dex-hand-scripted-lift \
  --output datasets/scripted_lift \
  --episodes 20 --max-attempts 100 --max-steps 500
```

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--task NAME` | `lift` | 已注册的脚本策略 |
| `--repo-id ID` | `local/dex-hand-scripted-demonstrations` | LeRobot 数据集标识 |
| `--output PATH` | `datasets/scripted_lerobot` | 本地数据集目录 |
| `--episodes N` | `20` | 要保存的成功 episode 数 |
| `--max-attempts N` | `100` | 包含失败在内的最大尝试次数 |
| `--max-steps N` | `400` | 每次尝试的最大控制步数 |
| `--fps N` | `20` | 数据集帧率 |
| `--camera NAME` | `agentview` | RGB 相机名 |
| `--image-width/--image-height` | `640` / `480` | 保存图像分辨率 |
| `--seed N` | `0` | 第一条轨迹的随机种子，后续尝试依次递增 |
| `--save-failures` | 关闭 | 同时保存失败轨迹 |
| `--dry-run` | 关闭 | 执行策略和成功判定，但不创建数据集 |
| `--no-video` | 关闭 | 不编码视频，适合只检查状态采集路径 |

### 仿真、机器人和任务演示

| 程序 | 功能 | 常用参数和运行示例 |
| --- | --- | --- |
| `source.demos.smoke_test` | 无窗口检查机器人模型、控制、触觉和注册任务，失败时返回非零状态码 | `python -m source.demos.smoke_test --steps 2`；`--skip-tasks` 跳过任务，`--fail-fast` 首次失败即停止 |
| `source.demos.manipulation_task_playback` | 在 Viewer 中查看任务、物体随机化和手动/预设动作 | `python -m source.demos.manipulation_task_playback --task lift --object-id ycb:005_tomato_soup_can`；完整选项见 `--help` |
| `source.demos.robot_preview_demo` | 预览当前机器人和关节控制 | `python -m source.demos.robot_preview_demo`；`--no-scene` 只加载机器人 |
| `source.demos.ik_sine_demo` | 让末端沿椭圆轨迹运动，用于检查 IK、坐标系和控制平滑度 | `python -m source.demos.ik_sine_demo --frequency 0.08 --radius-x 0.1 --radius-y 0.1`；`--no-realtime` 最快运行 |
| `source.demos.random_demo` | 向 position/IK 控制器发送可复现随机动作 | `python -m source.demos.random_demo --seed 0`；`--no-realtime` 取消实时限速 |

上述机器人演示均支持公共覆盖参数 `--robot-config`、`--arm-name`、`--hand-name` 和
`--base-name`；支持触觉的入口还可用 `--no-tactile`。

### 触觉程序

| 程序 | 功能 | 运行方式和关键参数 |
| --- | --- | --- |
| `source.demos.tactile_preview` | 显示 taxel 布局、编号和表面方向 | `python -m source.demos.tactile_preview --help` |
| `source.demos.tactile_probe_demo` | 用探针接触 taxel 并实时显示热力图 | `python -m source.demos.tactile_probe_demo --fps 60`；`--no-heatmap` 关闭热图，`--debug-tactile` 打印原始响应 |
| `source.demos.tactile_surface_fitting` | 可视化从 mesh/后端生成的触觉曲面和拟合点 | `python -m source.demos.tactile_surface_fitting --save tactile_surface.png`；`--backend` 选择后端 |
| `source.demos.tactile_contact_validation` | 自动逐点探测，检查即时响应、保持、串扰和释放 | `python -m source.demos.tactile_contact_validation --csv tactile.csv`；阈值参数见下文及 `--help` |
| `source.sensors.tactile.surface_fitting` | 底层曲面拟合模块的独立调试入口 | `python -m source.sensors.tactile.surface_fitting --help`；日常可视化优先使用上面的 demo |

### 遥操作、数据收集与硬件检查

| 程序 | 功能 | 运行方式 |
| --- | --- | --- |
| `source.demos.collect_teleop_lerobot` | 用 Vive 位姿和蓝牙弯曲手套收集 LeRobotDataset | `python -m source.demos.collect_teleop_lerobot --task lift --episodes 10 --output datasets/lerobot` |
| `source.teleop.vive.vive_link_test` | 检查 Vive 设备发现和位姿数据 | `python -m source.teleop.vive.vive_link_test --help` |
| `source.teleop.vive.vive_glove_hand_control` | 联调 Vive、蓝牙手套和手部控制映射 | `python -m source.teleop.vive.vive_glove_hand_control --help` |
| `source.teleop.bluetooth_glove.bluetooth_glove_test` | 单独检查蓝牙手套通道 | `python -m source.teleop.bluetooth_glove.bluetooth_glove_test --help` |

遥操作采集的主要数据参数是 `--task`、`--repo-id`、`--output`、`--episodes`、
`--episode-frames`、`--fps`、`--camera`、`--image-width` 和 `--image-height`；空间映射
使用 `--position-scale`，设备可用 `--vive-device-index` 或 `--vive-serial` 精确选择。
蓝牙地址、通道映射、标定、死区和反向选项以
`python -m source.demos.collect_teleop_lerobot --help` 输出为准。

### 模仿学习

| 程序 | 功能 | 运行方式和关键参数 |
| --- | --- | --- |
| `source.imitation.train_diffusion` | 从 LeRobotDataset 训练多模态 Diffusion Policy | `python -m source.imitation.train_diffusion --dataset datasets/lerobot --repo-id local/dex-hand-demonstrations --output checkpoints/diffusion_policy.pt`；常用参数：`--horizon 16 --epochs 100 --batch-size 32 --lr 1e-4 --device cuda` |
| `source.imitation.evaluate_diffusion` | 在任务环境闭环评估 checkpoint 并统计成功率 | `python -m source.imitation.evaluate_diffusion --checkpoint checkpoints/diffusion_policy.pt --task lift --episodes 20`；`--max-steps` 限制长度，`--action-steps` 设置每次重规划前执行的动作数 |

### 资产与开发工具

| 程序 | 功能 | 运行方式和关键参数 |
| --- | --- | --- |
| `tools/download_maniskill_objects.py` | 下载、校验并整理 YCB/EGAD 资产和 manifest | `python tools/download_maniskill_objects.py`；`--dry-run` 只查看计划，`--mode symlink` 使用软链接，目录可由 `--cache-dir`、`--output-dir`、`--manifest`、`--lock-dir` 覆盖 |
| `tools/render_object_catalog.py` | 根据 manifest 渲染物体目录图 | `python tools/render_object_catalog.py --output-dir assets/maniskill/catalog`；可设置 `--columns`、`--tile-width`、`--tile-height` |
| `tools/code_summary.py` | 生成代码结构/统计摘要，便于审查项目组成 | `python tools/code_summary.py --help` |

安装项目后还可使用 `dex-smoke-test`、`dex-vive-test`、`dex-vive-glove`、
`dex-glove-test`、`dex-collect-teleop` 和 `dex-task-playback` 这些等价的命令行入口。

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
python -m source.demos.manipulation_task_playback --task lift --object-id egad:G6
python -m source.demos.manipulation_task_playback --task stack
python -m source.demos.manipulation_task_playback --task pick_place --object-id ycb:025_mug
python -m source.demos.manipulation_task_playback --task nut_assembly
python -m source.demos.manipulation_task_playback --task push --object-id ycb:006_mustard_bottle
```

任务覆盖关系：

| 任务 | 类型 | 可选物体 |
| --- | --- | ---: |
| `lift` | 抓握并抬升 | 127 |
| `pick_place` | 抓握、搬运、放置 | 123 |
| `stack` | 堆叠 | 25 |
| `nut_assembly` | 精密对孔装配 | 程序化方螺母、圆螺母 |
| `push` | 桌面平面推动 | 117 |

只输出一张`agentview`相机图片、不打开MuJoCo Viewer或运行控制循环：

```powershell
python -m source.demos.manipulation_task_playback `
  --task pick_place `
  --snapshot docs/pick_place.png
```

`--snapshot`不填写路径时默认保存到`docs/manipulation_snapshot.png`。
可用`--camera`选择其他命名相机，使用`--image-width`和`--image-height`设置图片大小。

`lift`可选择全部127个目录物体，`pick_place`使用一个物体在源箱和目标箱两个完整格子
之间搬运（不再使用四物体、四目标分区），并排除链条、弹珠等4个不适合单刚体搬运的
模型，共123个候选。`stack`使用25个具有相对稳定支撑面的候选。`push`使用桌面目标区，
排除链条、弹珠和球类后仍覆盖117个适合平面推动的物体。物体在环境构造时由
`task_config`选择，例如：

```python
env = make_manipulation_env(
    "pick_place",
    task_config={"object_id": "ycb:025_mug", "reward_shaping": True},
)
```

MuJoCo模型编译后不会在`reset()`中替换网格。并行训练时应给不同worker传入不同
`object_id`，需要切换课程阶段时重建对应worker。`nut_assembly`的螺母几何、碰撞体和
定位site由代码直接构造，并复用`assets/textures`材质，不依赖旧的`assets/objects`目录。

## 收集示教

流程策略的调试与数据收集使用两个独立入口。交互验证不会创建数据集，每个阶段完成后
都会暂停；在 MuJoCo Viewer 中按 `Space` 确认进入下一阶段，按 `Q` 退出：

```bash
python -m source.demos.validate_scripted_strategy \
  --task lift --viewer-speed 1.0
```

确认所有阶段后，再使用无 Viewer 的收集入口自动写入 LeRobot 数据：

```bash
python -m source.demos.collect_scripted_lerobot \
  --task lift --episodes 20 --output datasets/scripted_lift
```

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
  --glove-mac 20:19:08:21:31:03 `
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
assets/maniskill/               下载后生成的 YCB/EGAD 物体与清单（Git 忽略）
assets/textures/                场景和程序化任务物体的共享纹理
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
- `requirements.txt`是包含学习、硬件和开发工具的完整环境；只做仿真可使用
  `pip install -e .`避免安装可选组件。
