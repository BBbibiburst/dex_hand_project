# P18.2-A：基于新增真实触觉映射图更新 Sim-to-Real 审计

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
Git HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`（`main`）  
执行性质：映射证据更新；未修改真实数据、物理参数、Strict、模型或 checkpoint，未训练、未运行实机、未 commit/push

## 1. 结论摘要

新增映射表解决了 P18.1 中“真实 patch 解剖名称全部未知”的主要缺口。现在可以把真实 ID 0–14 分别归入拇指、食指、中指、无名指、小指的三块阵列，并确认 ID 15 为手掌；这与全库 NPZ 中稳定的 16 块 shape 完全一致。

但该表不是缺失的 `config_mapping_zyhand.json`，也没有给出代码版本、hash、块内方向、传感器安装朝向或处理公式。因此当前结论分为三层：

- **图片直接支持**：真实 patch ID、中文名称和矩阵 shape。
- **代码与 shape 推导的候选**：真实各指 7×8 块对应仿真同指的 `_2` 指尖块；真实各指两个 4×8 块对应仿真同指 `_0/_1` 的某个排列；手掌对应 `skin_palm_p`。
- **仍未解决**：真实 `-2/-3` 到仿真 `_0/_1` 的一一顺序、每块 row/column 方向与翻转/旋转、原始 electrical channel order、raw→processed 算法及版本一致性。

712 维现在具备了**解剖 patch 级比较**的基础，但还不具备可靠的逐 taxel 空间对齐。可以开始受约束的 Raw Sim/Real 分布统计、patch 聚合标定和独立域判别诊断；不应启动以完整 712D 空间排列为输入的正式域迁移，更不应启动 GAN。

## 2. 证据等级

| 等级 | 含义 | 本轮实例 |
|---|---|---|
| `DIRECT_IMAGE` | 用户提供的映射图直接陈述 | Real ID 0=`拇指-1`、shape 7×8；ID 15=`手掌`、shape 7×16 |
| `CODE_CONFIRMED` | 当前仓库代码或 MJCF 直接确定 | 仿真 patch 名称/shape/flatten；finger 4 为 thumb；仿真行优先展平 |
| `DERIVED_CANDIDATE` | 综合解剖名、shape 和代码得到的候选，不是原 mapping 合同 | Real 0 → `skin_4_2_p`；Real 1/2 → `{skin_4_0_p, skin_4_1_p}` |
| `UNRESOLVED` | 当前证据不能确定 | Real 1 与 2 谁对应 `_0/_1`；块内镜像/旋转；processed 公式 |

新增图片表应在报告中保留来源标记。由于没有原图文件元数据、创建者、版本和 collector 引用，不能把它升级为“当前 79 个 session 使用的原始配置已找回”。

## 3. 仿真侧语义和展平合同复核

### 3.1 手指编号

仓库存在相互一致的代码证据：

- `source/teleop/devices.py` 把手套顺序 `thumb,index,middle,ring,pinky` 映射为手部顺序 `index,middle,ring,pinky,thumb`；
- `source/teleop/vive/hand_skeleton.py` 定义 `FINGER_NAMES = (index, middle, ring, pinky, thumb)`；
- `source/grasping/dexevolve_contacts.py` 明确说明 digit index 4 始终是 thumb；
- `assets/grippers/dex_hand/dex_hand.xml` 将 `skin_4_0_p`、`skin_4_1_p`、`skin_4_2_p` 安装在 `thumb_first/second/third` body 上。

因此本轮将仿真 finger 0–4 的解剖语义认定为：食指、中指、无名指、小指、拇指。这个结论来自仓库代码，不来自真实映射图。

### 3.2 segment 与形状

`source/sensors/tactile/fitting/layout.py::dex_hand_patch_layout` 定义每指：

- `_0`：4×8，`segment`；
- `_1`：4×8，`segment`；
- `_2`：7×8，`fingertip-ellipsoid`。

MJCF 还表明 `_0` 安装在第一节 body，`_1` 和 `_2` 位于更远端 body；拇指的 `_0/_1/_2` 分别位于 thumb first/second/third body。由此可以确认仿真 `_2` 是指尖类型，但不能反推真实名称后缀 `-1/-2/-3` 的近远端语义。

### 3.3 仿真块内方向

`source/sensors/tactile/dex_hand.py` 以 `for row ... for col ...` 生成传感器名，因此仿真 flatten 是 row-major。`_taxel_frame` 定义局部 X 为列增大方向、局部 Y 为行增大方向、局部 Z 为外法线。

这只确定仿真侧。真实 NPZ 的二维 shape 不能证明真实阵列在手上的行列物理方向。缺失 splitter/mapping 后，仍无法判断真实矩阵是否转置、上下/左右翻转或旋转。非方阵 shape 会限制某些无转置的 90°解释，但不能排除采集程序在保存前已做转置。

## 4. 真实—仿真候选对应表

下面的箭头均为 `DERIVED_CANDIDATE`，不是原始 mapping 文件证明。

| Real ID | 图片名称 | Shape | 仿真候选 | Sim flat interval | 尚未解决 |
|---:|---|---:|---|---:|---|
| 0 | 拇指-1 | 7×8 | `skin_4_2_p` | [544,600) | suffix语义；块内方向 |
| 1 | 拇指-2 | 4×8 | `skin_4_0_p` 或 `skin_4_1_p` | [480,512) / [512,544) | 二选一；块内方向 |
| 2 | 拇指-3 | 4×8 | `skin_4_0_p` 或 `skin_4_1_p` | [480,512) / [512,544) | 二选一；块内方向 |
| 3 | 食指-1 | 7×8 | `skin_0_2_p` | [64,120) | suffix语义；块内方向 |
| 4 | 食指-2 | 4×8 | `skin_0_0_p` 或 `skin_0_1_p` | [0,32) / [32,64) | 二选一；块内方向 |
| 5 | 食指-3 | 4×8 | `skin_0_0_p` 或 `skin_0_1_p` | [0,32) / [32,64) | 二选一；块内方向 |
| 6 | 中指-1 | 7×8 | `skin_1_2_p` | [184,240) | suffix语义；块内方向 |
| 7 | 中指-2 | 4×8 | `skin_1_0_p` 或 `skin_1_1_p` | [120,152) / [152,184) | 二选一；块内方向 |
| 8 | 中指-3 | 4×8 | `skin_1_0_p` 或 `skin_1_1_p` | [120,152) / [152,184) | 二选一；块内方向 |
| 9 | 无名指-1 | 7×8 | `skin_2_2_p` | [304,360) | suffix语义；块内方向 |
| 10 | 无名指-2 | 4×8 | `skin_2_0_p` 或 `skin_2_1_p` | [240,272) / [272,304) | 二选一；块内方向 |
| 11 | 无名指-3 | 4×8 | `skin_2_0_p` 或 `skin_2_1_p` | [240,272) / [272,304) | 二选一；块内方向 |
| 12 | 小指-1 | 7×8 | `skin_3_2_p` | [424,480) | suffix语义；块内方向 |
| 13 | 小指-2 | 4×8 | `skin_3_0_p` 或 `skin_3_1_p` | [360,392) / [392,424) | 二选一；块内方向 |
| 14 | 小指-3 | 4×8 | `skin_3_0_p` 或 `skin_3_1_p` | [360,392) / [392,424) | 二选一；块内方向 |
| 15 | 手掌 | 7×16 | `skin_palm_p` | [600,712) | 块内方向 |

结构化逐项结果见 `outputs/tactile_p18_2_2026-10-25/mapping_update.json`。

## 5. 712维可比性重新评估

### 5.1 已改善的部分

以前只能按 shape 类型比较；现在可以按真实解剖名称聚合：

- 五根手指和手掌的总响应、非零率、分位数、时间导数；
- 每根手指的 7×8 指尖候选与仿真 `_2` 比较；
- 每根手指两个 4×8 指节块作为无序集合，与仿真 `_0/_1` 聚合比较；
- 拇指、非拇指和手掌的接触阶段统计。

这些比较不需要先决定两个 4×8 块的排列，也可以使用对行列翻转不敏感的统计量。

### 5.2 仍不能做的部分

不能可靠构造一个逐元素的 `real[712] → sim[712]` 重排矩阵，原因包括：

1. 每指 `-2/-3` 到 `_0/_1` 有二元排列歧义；
2. 16块内部 row/column 的物理方向未知；
3. 原 mapping 是否对数组做过 transpose/flip 未知；
4. 79个 session 是否使用同一 mapping 版本没有 hash 证明；
5. Real raw 与 Sim touch 的单位和传递函数不同。

因此“维数相同”仍不等于 taxel 同位、数值同量纲或可直接监督。

## 6. processed数值超过1

新增映射表没有提供任何 raw→processed 处理信息，不能解释 P18.1 观察到的 processed 最大值 5.04（单 trial）和 9.90（全库抽样）。这些值可能来自校准系数、单位选择、基线扣除、门控/滤波或未归一化输出，但当前没有代码可证明其中任何一种。

因此：

- 不把 `processed > 1` 判为坏数据；
- 不把 processed 解释为 `[0,1]` 压力；
- 不以 clip 到1的方式“修复”；
- 分析时必须按 session 配置版本分层，并分别报告 raw 和 processed。

## 7. 当前可启动工作边界

| 工作 | 决策 | 当前允许范围 | 关键限制 |
|---|---|---|---|
| Raw Sim vs Real 分布统计 | **可开始，有限制** | patch总量、分位数、稀疏率、时序响应；指尖/手掌；指节无序集合 | 单位不同；需阶段/物体匹配及session版本分层 |
| patch级标定与归一化基线 | **可开始，有限制** | train-only robust scaling、分位数匹配、每指聚合；保持raw与processed两路 | 不得声称逐taxel标定；不可用测试集统计 |
| 独立域判别器 | **可开始，仅诊断** | 使用方向不敏感的patch聚合特征；按session划分；匹配任务阶段；报告混淆因素 | 不能把高准确率直接归因于“触觉域差”，也不能作为GAN就绪证明 |
| 完整712D空间域判别 | **暂缓** | 无 | segment排列和块内方向未解，会让分类器学习错误排列 |
| GAN/生成式域迁移 | **不具备条件** | 无 | mapping方向、processed算法/单位、版本漂移和下游保持指标均未解决 |

### 是否具备启动域判别基线的条件？

**有条件具备。** 可以启动一个明确标注为“诊断性”的 patch-aggregate 域判别器：输入使用每块或每指的总量、分位数、非零率和时间特征，采用 session-level split，只用训练集计算归一化，并匹配相同接触阶段。它可以量化域可分性和配置漂移。

**不具备**启动完整 712D spatial discriminator 作为正式域迁移基线的条件。否则分类器很可能利用 segment 排列、镜像、采样配置或 processed 版本差异，而不是学习有研究意义的接触域差异。

## 8. 最小实验与验收条件

下一项唯一最值得执行的实验仍是**逐 patch 受控接触映射验证**，但新增表让实验可以聚焦于尚未解决的部分：

1. 对每根手指的 `-2/-3` 位置分别单独按压，确定其近/远端语义和 `_0/_1` 对应；
2. 在每块四角及中心做已知位置按压，确定 row/column 正方向、镜像与旋转；
3. 同步保存 raw、processed、时间戳和视频，记录采集配置及 mapping hash；
4. 每点至少重复3次，检查主响应 patch、空间质心和重复性；
5. 冻结 `real ID → anatomy → sim patch → orientation transform → flat slice` 的版本化 JSON。

在硬件实验前，可以先完成低风险的 patch-aggregate 统计和诊断域判别器设计，但结果必须保持 permutation/orientation invariant，且不进入 GAN 训练。

## 9. 最终回答

### 新映射图改变了什么？

它可信地补充了真实 ID 0–15 的解剖名称，结束了“所有真实 patch 名称未知”的状态，并使五指/手掌级统计可开展。

### 是否已经得到可信的逐taxel真实—仿真映射？

**没有。** 指尖和手掌有唯一 shape/anatomy 候选；每指两个 4×8 指节块仍有排列歧义，所有块的内部方向仍未知。候选表不能替代原 mapping 文件或受控验证。

### processed算法是否得到解释？

**没有。** 映射图不包含采集、标定、滤波或归一化算法，数值超过1仍然未解。

### 现在能否启动域判别基线？

**可以启动受限的 patch-aggregate 诊断基线；不能启动完整712D空间基线或GAN。** 正式空间域迁移仍需先解决 segment排列、块内方向和处理算法/单位。

本轮只新增本报告和结构化 JSON，旧报告与数据均未覆盖。
