# UltraDexGrasp 新抓取示教流水线

## 目标与边界

`source/ultradexgrasp` 是一套全新实现，用于替代 GraspQP 初始搜索与 DexEvolve 进化优化。
它不导入 `source.grasping`，不读取 `configs/grasps`，也不生成旧 grasp-config JSON。

上游 UltraDexGrasp 当前支持 UR5e + XHand/LEAP；项目中的 Dex Hand 是六驱动闭链机构，
无法把 XHand 的 12 个串联关节直接映射过来。因此本实现保留 UltraDexGrasp/BODex 的
批量可微优化思想，但以项目自身 MuJoCo 模型作为运动学真值。

## 数据流

```text
assets/grippers/dex_hand/dex_hand.xml
  → 六驱动曲面标定与缓存

ManiSkill 物体 mesh
  → 居中、缩放、凸碰撞壳、表面采样

手根 6D 位姿 + 六驱动
  → 接触/穿透/桌面/摩擦锥联合优化
  → 有效候选集合

具体 Lift 场景
  → RM75B IK 可达性排序
  → 张开/中转/预抓取/同步接近闭合/预载/抬升/验证
  → manifest.json + episode.npz
```

## 安装

推荐使用与已验证环境一致的 Python 3.10、Torch 2.4.1 + CUDA 11.8：

```bash
python -m pip install -e ".[dev,assets,ultradexgrasp]"
python -m tools.ultradexgrasp.bootstrap --clone-missing
python -m tools.ultradexgrasp.probe --strict
```

固定版本记录在 `deps/ultradexgrasp/versions.json`。`bootstrap` 默认只核对本地 checkout；
缺失时可显式执行：

```bash
python -m tools.ultradexgrasp.bootstrap --clone-missing
```

cuRobo/BODex 扩展需要使用 Conda 环境中的 CUDA 11.8 `nvcc`，不要使用系统 CUDA 11.5：

```bash
export PATH=/path/to/env/bin:$PATH
export CUDA_HOME=/path/to/env
export CUDACXX=/path/to/env/bin/nvcc
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=8
```

原生生成器本身只依赖 Torch、MuJoCo、SciPy 和 trimesh；上游 checkout 用于版本核验和
基线对照，不会在生产执行路径中伪装成 XHand 到 Dex Hand 的映射。

## 使用

完整生成与执行：

```bash
MUJOCO_GL=egl python -m tools.ultradexgrasp.generate \
  --object-id ycb:002_master_chef_can \
  --seed 11
```

快速调整规模：

```bash
python -m tools.ultradexgrasp.generate \
  --object-id ycb:003_cracker_box \
  --seed-count 32 \
  --optimization-steps 160
```

只运行抓取合成：

```bash
python -m tools.ultradexgrasp.generate \
  --object-id ycb:024_bowl \
  --synthesis-only
```

默认参数在 `configs/ultradexgrasp/default.json`。曲面代理缓存在
`configs/ultradexgrasp/cache/dex_hand_surrogate.npz`，该目录属于本地产物并已忽略。

## 坐标与控制约定

候选中的 `hand_translation` 和 `hand_rotation_matrix` 把 Dex Hand 根坐标映射到居中的
物体坐标。执行时：

```text
world_hand_R = object_R @ candidate_R
world_hand_t = object_t + object_R @ candidate_t
world_ee_R   = world_hand_R @ hand_attach_R.T
```

预抓取沿手坐标 `-Y` 后退。IK action 顺序是世界坐标位置 3 维、`wxyz` 四元数 4 维，
再接六个 Dex Hand 物理执行器目标。

## 输出 contract

成功目录：

```text
manifest.json       episode 元数据、候选和成功状态
episode.npz         每帧 qpos/qvel/ctrl/action/物体位姿/阶段/奖励/成功
candidates.json     几何候选及 RM75B 可达性排序
run.json            执行尝试摘要
```

`episode.npz` 还记录机器人—物体接触数量和法向力，便于区分 IK 失败、未接触、夹持不足与
抬升滑落。可使用 `DemonstrationEpisode.load(path)` 读取并验证。

## 已验证结果

- 上游 BODex + XHand baseline 已实际运行并返回有限候选。
- Dex Hand 曲面代理当前 216 个采样点，标定 RMS 约 0.68 mm。
- `002_master_chef_can`、`003_cracker_box`、`024_bowl` 均能生成五指有效几何候选，代表性
  最佳最大穿透小于约 1.1 mm。
- `002_master_chef_can` 已完成一次完整 RM75B Lift episode：最高抬升约 5.1 cm，验证
  30/30 帧成功，最终抬升约 4.5 cm。

这些结果证明新链路已端到端打通，但不等于所有 127 个对象都已完成批量验证。下一阶段
应在固定 seed 集合上统计每类物体的合成率、IK 可达率、Lift 成功率和平均尝试次数。
