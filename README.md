# 描述符驱动的机械臂 + 灵巧手 MuJoCo 仿真框架

一个基于 **MuJoCo** 和 **Gymnasium** 的机器人仿真框架。核心设计思想是"描述符驱动"（descriptor-driven）：
机械臂、末端执行器（手/夹爪）、移动底座都用一个轻量的数据类（dataclass）描述"是什么"（XML 路径、
执行器名字、挂载点……），框架再根据描述符自动完成模型拼装、控制器构建、触觉传感器注入等工作。
新增一款手爪或机械臂，理论上只需要新写一个描述符文件并注册，不需要改动控制器、环境、传感器等任何框架代码。

---

## 一、项目结构

```
source/
├── code_summary/                # 代码合并/导出小工具
│   └── code_summary.py
│
├── control/                     # 控制层：位置控制 / IK 控制
│   ├── controllers.py
│   └── __init__.py
│
├── demos/                       # 可直接运行的演示脚本
│   ├── random_demo.py           # 随机动作 demo
│   ├── ik_sine_demo.py          # 末端 IK 圆形轨迹 demo
│   ├── tactile_preview.py       # 触觉 taxel 站点可视化
│   └── tactile_sampling_plot.py # 触觉采样网格离线绘图（matplotlib）
│
├── environments/                 # 环境层：Gym 环境、模型拼装、场景、任务、传感器接口
│   ├── assets.py                # 资源路径常量
│   ├── robot_builder.py         # MjSpec 拼装（手臂+手+底座）
│   ├── rl_env.py                # RobotGymEnv（Gymnasium 环境）
│   ├── scene.py                 # 地面/灯光等场景装饰
│   ├── overlays.py              # 被动查看器标记/文字覆盖层
│   ├── tactile_sensors.py       # TactileSensorBase 抽象接口 + NullTactileSensor
│   ├── tasks.py                 # RobotTask 抽象接口 + NoopTask
│   └── transforms.py            # 四元数/旋转数学工具
│
└── robots/                       # 描述符与注册表
    ├── descriptors.py           # ArmDescriptor / EndEffectorDescriptor / BaseDescriptor
    ├── registry.py              # 注册与查找（get_arm/get_hand/get_base）
    ├── defaults.py              # 默认组合：RM75B + dex_hand + rethink_minimal_mount
    ├── arms/rm75b.py            # RM75B 机械臂描述符
    ├── bases/rethink_minimal_mount.py
    └── hands/
        ├── dex_hand.py          # 灵巧手描述符
        ├── dex_hand_tactile.py  # 灵巧手触觉传感器实现（DexHandTouchSensor）
        └── _surface_fitting.py  # 私有：STL 网格 -> taxel 网格的曲面拟合算法
```

### 分层原则

| 层 | 职责 | 关键点 |
|---|---|---|
| `robots/descriptors.py` | 定义"设备是什么" | 只含路径、执行器名、挂载点、工厂函数，**不含**任何传感器实现细节 |
| `robots/registry.py` | 组件注册与查找 | `register_arm/hand/base` + `get_arm/hand/base`，内置描述符懒加载，避免循环导入 |
| `environments/robot_builder.py` | 模型拼装 | 直接操作 `mujoco.MjSpec` 对象完成挂载/触觉注入，**不做任何 XML 文本拼接** |
| `control/controllers.py` | 运动控制 | 位置控制 + 带阻尼最小二乘的解析 IK，可分别构建再组合成 `CompositeRobotController` |
| `environments/tactile_sensors.py` | 触觉传感器**接口** | 4 个生命周期方法：`augment_spec → bind → reset → read`，框架完全不知道传感器怎么实现 |
| `robots/hands/dex_hand_tactile.py` | 触觉传感器**具体实现** | 只有灵巧手才需要的 STL 曲面拟合逻辑，全部私有，其他手爪可以有完全不同的实现 |
| `environments/rl_env.py` | Gym 环境 | 组合控制器 + 任务 + 触觉传感器，负责计时、渲染、统计 |
| `environments/tasks.py` | 任务逻辑 | `RobotTask` 抽象类：obs / reward / termination，默认 `NoopTask` |

---

## 二、核心组件说明

### 1. 描述符（Descriptors）

```python
ArmDescriptor(name, xml_path, position_actuator_names, ee_site_name,
              hand_attach_body_name, hand_attach_rot_xyz_deg, controller_factory=None)

EndEffectorDescriptor(name, xml_path, position_actuator_names=(), default_prefix="",
                       tactile_sensor_factory=None, controller_factory=None)

BaseDescriptor(name, xml_path, arm_mount_site_name, mount_prefix="mount_")
```

默认组合（`source/robots/defaults.py`）：**RM75B 机械臂 + dex_hand 灵巧手 + rethink_minimal_mount 底座**。

### 2. 控制层（`control/controllers.py`）

- `ArmPositionIkController`：单臂控制器，支持 `position`（直接给关节目标位置）和 `ik`（笛卡尔空间位姿目标）两种模式。
  IK 求解特性：
  - 自适应阻尼最小二乘（damping 随奇异值自适应），提升近奇异位形附近的稳定性
  - 零空间姿态保持（nullspace posture）
  - 关节步长限幅 + 速度低通滤波，避免抖动
  - 末端目标位姿一阶低通滤波 + 四元数 Slerp 插值
- `EndEffectorPositionController`：手/夹爪的直接执行器位置控制。
- `CompositeRobotController`：把上面两者拼成环境需要的单一动作空间（`[arm_action, hand_action]`）。
- `build_robot_controller(...)`：根据描述符自动选择/构建控制器（描述符可自带 `controller_factory`）。

### 3. 模型拼装（`environments/robot_builder.py`）

`build_robot_spec()` 依次完成：底座挂载 → 手部触觉传感器注入（`augment_spec`）→ 手臂末端挂载手部，
全程只操作 `mujoco.MjSpec` 对象，无 XML 字符串拼接/临时文件。`build_robot_model()` 在此基础上加场景并编译出
`MjModel`/`MjData`。

### 4. 触觉传感器（灵巧手专属实现）

灵巧手每根手指的 3 个皮肤贴片（近节/中节用"手指段"椭圆柱拟合，指尖用椭球帽拟合）+ 手掌（mesh-UV 采样）
共同组成 taxel 触觉阵列。每个 taxel 是一个带 `mjSENS_TOUCH` 传感器的小球体 site，压力值即 MuJoCo 接触力。

### 5. Gym 环境（`environments/rl_env.py`）

`RobotGymEnv`：

- Observation：`{qpos, qvel, ctrl, tactile, ...task_obs}`
- Action：`controller.action_space`（位置模式或 IK 模式，形状不同）
- 支持 `render_mode="human"`（被动查看器）或 `"rgb_array"`
- 内置仿真统计（物理步频率、实时倍率、控制频率）

---

## 三、运行方式

> 依赖：`mujoco`、`gymnasium`、`numpy`、`scipy`（IK/曲面拟合用）、`matplotlib`（仅采样绘图 demo 需要）。

### 1. 预览默认机器人装配（不含任务/RL）

```bash
python -m source.environments.robot_builder
```
效果：弹出 MuJoCo 被动查看器窗口，展示 RM75B + 灵巧手 + 底座拼装好的静态/自由落体场景，可用于检查挂载是否正确。

### 2. 随机动作 Demo

```bash
python -m source.demos.random_demo --control-hz 2 --seed 0
```
效果：以给定频率（默认 2Hz）向环境喂随机动作，弹出渲染窗口观察机器人随机运动；退出窗口后在终端打印
observation keys、action shape、reward、terminated/truncated、info 等信息。

### 3. 末端 IK 圆形轨迹 Demo

```bash
python -m source.demos.ik_sine_demo --radius-x 0.03 --radius-y 0.03 --frequency 0.08
```
效果：切换到 IK 控制模式，让末端沿一个（可带 z 方向二次谐波的）椭圆轨迹运动，手指做周期性开合；
查看器中会叠加绘制：红色轨迹圆心、蓝色轨迹路径、绿色实时目标点，以及左下角的仿真频率/实时倍率统计文字。
可加 `--print-error` 在终端打印每秒的 IK 误差与迭代次数。

### 4. 触觉 taxel 站点预览

```bash
python -m source.demos.tactile_preview --patch skin_0_0_p --radius 0.0025
```
效果：编译带触觉传感器的手部模型，在查看器中用不同颜色球体标出每个 taxel 站点位置（青色=近节，
绿色=中节，黄色=指尖，红色=手掌），并随手指运动实时刷新。不加 `--patch` 时显示全部贴片。

### 5. 触觉采样网格离线绘图

```bash
python -m source.demos.tactile_sampling_plot --patches skin_0_0_p skin_palm_p --strategy compare-all --save out.png
```
效果：用 matplotlib 画出所选皮肤贴片的 STL 三角网格曲面、拟合曲面与采样点网格（3D 图），
`--strategy` 可选 `fit / mesh-uv / rbf-outer / fingertip-ellipsoid / compare-all / compare-fingertip`，
便于调试/对比不同曲面拟合策略的采样质量。

### 6. 代码合并小工具（辅助文档/审查用）

```bash
python -m source.code_summary.code_summary ./source/control/controllers.py ./source/environments/rl_env.py
# 或
python -m source.code_summary.code_summary --list source/code_summary/code_list.txt
```
效果：自动跳过二进制文件/不存在路径/目录，把有效文件合并为一份带编号分隔符的 `code_summary.txt`
（即本文档最初输入的那种"Code Merge Report"格式），方便整体喂给 AI 或做代码评审。

---

## 四、如何扩展新设备

1. **新机械臂**：仿照 `source/robots/arms/rm75b.py` 写一个新文件，构造 `ArmDescriptor` 并调用
   `register_arm(...)`，再在 `source/robots/arms/__init__.py` 中导入即可。
2. **新手爪/末端执行器**：仿照 `source/robots/hands/dex_hand.py`。若无触觉传感需求，
   `tactile_sensor_factory` 留空（默认 `None`），环境会自动回退到 `NullTactileSensor`；
   若需要触觉，参考 `dex_hand_tactile.py` 实现自己的 `TactileSensorBase` 子类
   （曲面拟合、接触力求和、学习模型皆可，框架不关心具体做法）。
3. **新底座**：仿照 `source/robots/bases/rethink_minimal_mount.py`。

新增设备后无需修改 `control/`、`environments/`、`registry.py` 中的任何既有逻辑。

---

## 五、可运行入口一览

| 命令 | 作用 |
|---|---|
| `python -m source.environments.robot_builder` | 静态预览拼装好的机器人模型 |
| `python -m source.demos.random_demo` | 随机动作 + 渲染 + 打印 env 信息 |
| `python -m source.demos.ik_sine_demo` | IK 圆形轨迹交互 demo（含可视化标记与统计） |
| `python -m source.demos.tactile_preview` | 触觉 taxel 站点 3D 预览 |
| `python -m source.demos.tactile_sampling_plot` | 触觉采样网格离线对比绘图 |
| `python -m source.code_summary.code_summary` | 多文件代码合并导出工具 |