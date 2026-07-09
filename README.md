# Descriptor-Driven MuJoCo Robot Framework

这是一个基于 MuJoCo 和 Gymnasium 的机器人仿真框架。当前架构的主线是：

`configs/current_robot.json -> robot profile -> registry descriptors -> robot_builder -> RobotGymEnv / demos`

机械臂、末端执行器、底座都由 descriptor 描述；配置文件决定当前装配，builder 负责统一拼装，环境和 demos 复用同一套装配逻辑。

## Project Layout

```text
configs/
  current_robot.json                 # 当前机器人入口，可指向 robot_profiles 中的 profile
  robot_profiles/
    rm75b_dex_hand.json
    rm75b_pika_gripper.json

source/
  control/
    arm.py                           # 机械臂位置控制 / IK 控制
    end_effectors.py                 # 手/夹爪控制器
    composite.py                     # 组合控制器与控制器工厂
  demos/
    common.py                        # demo 共享的配置加载和环境创建
    random_demo.py
    ik_sine_demo.py
    tactile_preview.py               # dex_hand 触觉专用
    tactile_sampling_plot.py         # dex_hand 触觉专用
  environments/
    robot_config.py                  # profile + overrides 配置解析
    robot_builder.py                 # MjSpec 拼装：底座 + 机械臂 + 末端执行器
    rl_env.py                        # Gymnasium 环境
    tactile_sensors.py               # 触觉传感器接口与 NullTactileSensor
    tasks.py
  robots/
    descriptors.py                   # ArmDescriptor / EndEffectorDescriptor / BaseDescriptor
    registry.py                      # descriptor 注册与查找，自动发现内置模块
    arms/
    bases/
    hands/
  sensors/
    tactile/
      dex_hand.py                    # dex_hand 触觉实现
      _surface_fitting.py
```

## Robot Config

默认入口是 `configs/current_robot.json`：

```json
{
  "profile": "robot_profiles/rm75b_pika_gripper.json",
  "overrides": {}
}
```

`profile` 指向一个可复用机器人配置，`overrides` 用于临时覆盖局部字段。常用字段包括：

- `arm_name`
- `hand_name`
- `base_name`
- `hand_attach_rot_xyz_deg`
- `attach_point_name`
- `base_mount_site_name`
- `hand_prefix`
- `control_mode`
- `enable_tactile_sensors`

推荐优先修改 profile 或 `current_robot.json`，而不是给每个 demo 塞很多命令行参数。

## Main Entrypoints

```bash
python -m source.environments.robot_builder
python -m source.demos.random_demo
python -m source.demos.ik_sine_demo
python -m source.demos.tactile_preview
python -m source.demos.tactile_sampling_plot
```

`robot_builder` 和通用 demos 都读取同一套 robot config。`random_demo`、`ik_sine_demo` 通过 `make_env()` 创建环境，因此会使用和 builder 一致的手爪安装旋转、安装点和底座挂载配置。

触觉相关 demo 目前是 dex_hand 专用；如果当前配置不是 `dex_hand`，会给出明确错误。

## Architecture Notes

- `robots/descriptors.py` 只描述设备是什么：XML 路径、执行器名、安装点、默认前缀、控制器工厂、触觉工厂。
- `robots/registry.py` 负责注册和查找设备。内置设备模块会被懒加载自动发现；新增设备时不需要修改 registry。
- `environments/robot_builder.py` 只操作 `mujoco.MjSpec`，不做 XML 字符串拼接。
- `control/composite.py` 根据 arm/hand descriptor 构建组合控制器。
- `environments/rl_env.py` 将 builder、controller、task、tactile sensor 组合成 Gymnasium 环境。

## Adding Devices

新增机械臂：

1. 新建 `source/robots/arms/<name>.py`。
2. 构造 `ArmDescriptor`。
3. 在模块顶层调用 `register_arm(...)`。

新增手或夹爪：

1. 新建 `source/robots/hands/<name>.py`。
2. 构造 `EndEffectorDescriptor`。
3. 设置 `position_actuator_names`、`default_prefix`。
4. 如需专用控制器，提供 `controller_factory`。
5. 如需触觉，提供 `tactile_sensor_factory`；否则留空，环境会使用 `NullTactileSensor`。
6. 在模块顶层调用 `register_hand(...)`。

新增底座：

1. 新建 `source/robots/bases/<name>.py`。
2. 构造 `BaseDescriptor`。
3. 在模块顶层调用 `register_base(...)`。

新文件放入对应目录后会被 registry 自动发现。之后只需要在 robot profile 里使用新的 `arm_name`、`hand_name` 或 `base_name`。

## Useful Commands

```bash
python -m source.environments.robot_builder --no-tactile
python -m source.demos.random_demo --control-hz 2 --seed 0
python -m source.demos.ik_sine_demo --radius-x 0.03 --radius-y 0.03 --frequency 0.08
python -m source.demos.tactile_preview --patch skin_0_0_p --radius 0.0025
python -m source.demos.tactile_sampling_plot --patches skin_0_0_p skin_palm_p --strategy compare-all --save out.png
```
