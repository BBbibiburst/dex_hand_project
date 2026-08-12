# 代码结构与语义约定

## 模块边界

```text
source/envs/                 MuJoCo 环境、任务和 task registry
source/robots/               机器人组件描述与 robot config
source/controllers/          机械臂与末端控制
source/grasping/             搜索、GraspQP 适配、进化与物理验证
source/workflows/            可恢复的目录级工作流
source/evaluation/           公共 benchmark schema 与状态常量
source/scripted/             分阶段 scripted strategy
source/sensors/              触觉与其他传感器接口
source/teleop/               Vive、手套与 LeRobot recorder
source/imitation/            数据 schema、训练和评测
source/viz/                  通用可视化
apps/                        面向用户的数据采集程序
tools/                       搜索、验证、诊断和资产命令
```

Task 与 Scripted Strategy 是不同概念。Task 定义环境目标和 `task_success`；strategy 定义一套
自动执行流程。Strategy registry 显式记录对应 task，task 可以存在而没有 scripted strategy。

## 抓取稳定性层级

### Direct hold

`direct_hold_stable`：跳过完整接近路径，直接在候选终态闭合/保持后的动力学结果。只能用于
进化候选筛选。

### Trajectory collision

`trajectory_collision_free`：实际 approach/closure waypoints 满足当前物体和桌面碰撞约束。

### Trajectory hold

`trajectory_hold_stable`：执行完整轨迹后，最终保持满足位移、旋转、掉落和接触阈值。

### Benchmark status

只有完整轨迹验证通过才写 `trajectory_stable`。当前公共状态集中定义在
`source/evaluation/grasp_schema.py`：

```text
trajectory_stable
direct_hold_only
unstable
validation_error
search_error
```

`legacy_stable` 只允许旧报告可视化读取。

## Success 层级

- `info["task_success"]`：当前 task 条件在该 simulation step 成立。
- `strategy_verified_success`：scripted strategy 已完成整个执行与最终持续验证。
- `robot_lift_verified`：benchmark 中完整机械臂 Lift 执行成功且无桌面碰撞。

它们不能互相替代。局部的 `lift_stable_steps` 等字段表示条件连续成立的控制步数，不是公共
业务状态。

## Validation result

- `DirectHoldValidationResult.direct_hold_stable`
- `TrajectoryValidationResult.trajectory_collision_free`
- `TrajectoryValidationResult.trajectory_hold_stable`

调用者不再使用依赖上下文解释的 `result.hold_stable`。

## Tactile API

正式接口收敛为：

```text
read()
read_raw()
read_patches()
read_images()
metadata()
```

历史别名 `diagnostic_values()`、`read_concat()` 和 `read_image()` 已移除。数值算法、surface
fitting 与传感器物理不因此改变。

## Robot config

CLI 环境创建统一通过 `source/cli/robot_config.py`。触觉覆盖规则：

```text
显式 --no-tactile  → False
未指定             → None，保留 robot config 原值
```

应用不应自行实现 `enable_tactile_sensors = not args.no_tactile`。

## 持久化与兼容性

- 抓取 benchmark 当前 schema 为 4，语义为 `trajectory-hold-v2`。
- 长任务通过临时文件写入后原子替换正式 JSON。
- 恢复时校验影响结果的参数，禁止静默混合不同实验。
- 配置可从另一台机器复制；mesh 路径会从其中的 `assets/...` 部分重定位。
- 新 writer 不产生歧义旧状态；旧兼容逻辑只存在于只读可视化边界。

## 防回归检查

`tests/test_semantic_contracts.py` 等测试约束：

- direct hold 不能自动等同 trajectory stable
- 当前 benchmark 不写 `stable` 或 `legacy_stable`
- task success 与 strategy success 分层
- robot config 保留触觉设置
- scripted strategy 对应真实 task
- GraspQP、轨迹与 Robot Lift 接口保持明确

提交前运行：

```bash
python -m ruff check source apps tests examples tools --exclude deps
python -m pytest -q
git diff --check
```
