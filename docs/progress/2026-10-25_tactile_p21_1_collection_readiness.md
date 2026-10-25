# P21.1：真实触觉采集可行性与受控配对实验准备

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
Git HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`  
执行性质：只读软件审计；未连接或访问硬件，未运行任何机械臂、灵巧手或真实传感器程序

## 1. Executive Summary

当前软件侧真实触觉采集状态为 **BLOCKED**。

历史数据能够证明另一台Windows主机上的外部工程曾完成以下工作：`Usb32`读取、16块拆分、raw/processed双路保存、时间戳和多设备同步。但当前仓库和本机可访问路径中没有：

- `Usb32`实现及依赖；
- `config_mapping_zyhand.json`；
- splitter；
- raw→processed算法；
- calibration/auto-zero/gating实现；
- 独立、只读真实触觉采集入口。

因此，现在不能在本仓库中安全地“只读取触觉”并证明不会初始化机械臂或灵巧手控制。已有NPZ只能离线读取，不能替代在线硬件接口。

仿真侧状态更好：

- `apps/export_sim_tactile_raw.py`已经能从现有轨迹同步导出raw、processed、simulation time、stage和接触真值；
- `tools/tactile/validate_contacts.py`能够自动将球形probe放到指定taxel；
- `tools/tactile/interactive_probe.py`能够选择patch并交互移动probe。

但目前没有一个现成程序同时完成“指定patch/中心四角位置 → baseline/contact/hold/release → 双路raw/processed配对导出”。所以仿真指定位置采集为 **CONDITIONALLY READY**，需要一个隔离的诊断导出适配器；本阶段没有实现它。

下一步不是连接硬件，而是先取得并静态审查原始采集工程、mapping、processed算法和标定材料。只有确认触觉-only模式绝不会初始化arm/hand命令后，才允许执行最小无运动采集。

## 2. 真实采集软件证据

### 2.1 历史数据直接证明的合同

79份session配置一致引用：

`Usb_API_stable_Tujian_/spliter/config_mapping_zyhand.json`

较新配置还记录外部工程路径：

`D:\Robot\xxx_aaa\batch_test_v2\configs\collection_config.yaml`

真实配置中的关键字段：

| 字段 | 历史值/含义 |
|---|---|
| `sensor_class` | `Usb32` |
| `sensor_shape` | `[32,32]`，是provider层声明，不等于最终16块shape |
| `splitted` | `true` |
| `mapping_file` | 外部相对路径，当前缺失 |
| `fake` | `false` |
| `sync_group.sample_hz` | 新配置60 Hz；历史存在版本变化 |
| `tactile_timeout_s` | 0.03 s |
| `calibrate` | `true` |
| `auto_zero_calibration` | 新配置常为`true` |
| `calibrate_file` | `null`；不能推断为无需标定 |
| gated/consistency | 配置存在，但执行代码缺失 |

每个完整trial包含：

- `tactile_raw_split.npz`；
- `tactile_processed.npz`；
- `samples.jsonl`中的`sample_ts`、`tactile_ts`、`tactile_sample_ts`、`sync_id`、`tactile_sync_id`和stage；
- hand/arm状态字段。

这证明历史collector具有双路数据和同步机制，不证明当前仓库具有相同接口。

### 2.2 当前工作区缺失项

定向搜索当前仓库、项目父目录和`/home/zmy`可访问相关路径，没有发现：

- `Usb32` Python/C++类、DLL/SO或Python绑定；
- mapping原文件；
- `Usb_API_stable_Tujian_`或`spliter`目录；
- `batch_test_v2`采集工程；
- 可单独启动的real tactile reader/recorder。

当前`source/sensors/tactile/`全部是MuJoCo仿真实现，不是Usb32真实驱动。`source/teleop/session.py`中的`read_raw`也读取仿真env sensor，不能读取真实Usb32。

### 2.3 在线接口可用性判定

| 接口 | 历史数据证明 | 当前可执行 | 结论 |
|---|---|---|---|
| Real raw | 有16块raw NPZ | 否 | BLOCKED |
| Real processed | 有16块processed NPZ | 否 | BLOCKED |
| 设备时间戳 | `tactile_ts`等存在 | 否 | BLOCKED |
| 同步ID | `sync_id/tactile_sync_id`存在 | 否 | BLOCKED |
| Tactile-only模式 | 配置曾把devices组合为tactile/hand/arm | 未发现独立模式实现 | BLOCKED |

不能根据配置中存在`tactile.enabled`便假设把`arm.enabled=false`、`hand.enabled=false`就一定安全；需要实际collector代码证明初始化和关闭路径不会连接或下发命令。

## 3. 仿真配对采集能力

### 3.1 已经可用

`apps/export_sim_tactile_raw.py`对现有Strict replay轨迹已经验证：

- 同一env step后保存712D `mjSENS_TOUCH` raw与production processed；
- 保存simulation time、frame index、stage、qpos/qvel/ctrl、物体pose及逐指接触真值；
- 独立processor重建误差为0；
- raw读取不改变仿真状态；
- 24个episode完成双路导出。

该程序适合回放完整抓取，不提供指定patch和接触位置的受控压触。

### 3.2 已有probe工具

`source/sensors/tactile/probe.py`可向MjSpec注入自由球形probe并设置位置。

`tools/tactile/validate_contacts.py`可以：

- 按patch筛选；
- 获取每个taxel site的位置和法向；
- 自动执行接近、接触、hold和release；
- 读取raw并记录目标响应、crosstalk和release。

`tools/tactile/interactive_probe.py`可以选择patch、显示heatmap并手动移动probe。

### 3.3 尚未满足配对协议的部分

- 自动probe验证默认侧重raw和taxel pass/fail，没有同步保存production processed；
- 没有统一的baseline/contact/hold/release逐帧NPZ合同；
- 没有以patch中心和四角为显式实验单元的manifest；
- probe穿入深度不是可追溯的真实载荷，不能直接与真实牛顿值比较；
- 尚未记录Real mapping/hash，无法生成可审计的pair ID。

因此现有仿真组件足以支撑后续最小适配，但不能直接宣称受控Real/Sim配对采集已经打通。

## 4. 最小无运动真实试验设计

### 4.1 硬性前置条件

只有同时满足以下条件才允许开始：

1. 取得`Usb32` reader及依赖，并完成静态代码审查；
2. 取得实际mapping文件并记录SHA-256；
3. 取得raw→processed、auto-zero、gating、consistency和calibration实现；
4. collector提供明确的`tactile_only`模式；代码证明不会实例化arm/hand客户端或下发任何命令；
5. 机械臂与灵巧手保持固定，运动控制电源/命令路径按实验室安全规程禁用；触觉传感器供电和数据链路独立确认；
6. 输出使用全新目录，不写入`outputs/realdata/`；
7. 空载dry acquisition验证raw、processed、时间戳、sync ID、shape、finite及单调时间；
8. 操作者现场确认可以仅用外部手持接触头触碰传感器，不触发机器人动作。

### 4.2 首轮三个patch

建议选择：

- 一个7×8指尖：真实`食指-1`或有明确物理标记的任一指尖；
- 一个4×8指节：同一手指`-2`或`-3`之一，用于开始区分近/远端；
- 7×16掌面。

选择必须基于实物标签或提供者确认，不能只根据shape猜测。

### 4.3 每个位置的时序

每个patch先只测试中心：

1. BASELINE：不接触，至少2 s；
2. CONTACT：外部标准接触头缓慢接触，记录人工事件按钮/脚踏或独立触发时间；
3. HOLD：保持固定条件至少2 s；
4. RELEASE：完全释放并继续记录至少2 s；
5. 每个patch至少3次独立重复；每次重新记录baseline，不复制帧。

若没有载荷传感器，本轮只能验证patch身份、方向线索、响应重复性和时间合同，不能声称力—响应标定。接触头、直径、材料、方向和操作者必须记录。

### 4.4 最小成功条件

- 3个目标patch均为主要响应patch；
- raw与processed逐帧数量一致且可按sync ID关联；
- 时间戳单调，无不可解释的丢帧/重复帧；
- 三次重复的响应方向一致；
- release后信号能回到预注册的baseline容差；
- 全程无arm/hand命令、目标更新或状态变化；
- mapping、collector、配置和代码hash完整保存。

任一条件失败即停止扩展到16块。

## 5. 全16块扩展协议

最小试验通过后，对16块分别执行：

- 位置：中心、左上、右上、左下、右下；
- 方向：优先沿传感器表面法向；若评估斜向，单独作为后续条件，不与法向混合；
- 载荷：至少两个可重复等级；若没有校准载荷设备，只记录位移/接触事件并明确无力标定；
- 时序：baseline→contact→hold→release；
- 重复：每位置/载荷至少3次；
- 每次记录完整16块，不只保存目标块，以评估串扰和错误mapping；
- 同步视频必须看见patch标签、接触位置和触发事件；
- 所有trial按独立session和日期组织，不随机拆帧。

五个位置用于确定row/column方向、镜像和旋转。两个4×8块必须分别物理按压，确认`-2/-3`与仿真`_0/_1`的近远端对应。

## 6. Real/Sim配对字段与目录建议

建议使用新的独立root，例如：

```text
outputs/tactile_paired_collection_v1/
  contract.json
  split_manifest.json
  real/session_0001/trial_0001/{manifest.json,tactile.npz,video.mp4}
  sim/session_0001/trial_0001/{manifest.json,tactile.npz}
```

### 6.1 每帧字段

- `frame_index`；
- `source_timestamp`、`host_timestamp`、`sync_id`；
- `raw_patches`或可无损恢复16块的raw712；
- `processed_patches`或processed712；
- `phase`：BASELINE/CONTACT/HOLD/RELEASE；
- `contact_event`：独立触发真值，不由触觉阈值反推；
- 可用时记录`load`、`indenter_pose`和`contact_position`；
- Sim额外保存真实contact geom、normal force、probe pose和simulation time。

### 6.2 Trial/session元数据

- domain、session ID、trial ID、repeat ID；
- real patch ID、解剖名、sim patch候选、row/column transform；
- position ID、nominal row/column、中心/四角；
- 接触头尺寸/材料、方向、载荷或位移协议；
- sensor serial、USB port、firmware/driver版本；
- mapping、collector、processing、calibration和配置hash；
- sample rate、timeout、丢帧、overrun、温度；
- 操作者、日期和备注；
- 所有NPZ及视频hash。

同一真实session的全部重复只能进入同一split；对应Sim pair也随pair group进入同一split，防止配对派生数据泄漏。

## 7. Real/Sim当前对应边界

| 内容 | 当前状态 |
|---|---|
| 五根手指与掌面patch身份 | 图片表支持 |
| 7×8指尖→Sim `_2` | shape+代码候选，仍需实物验证 |
| 两个4×8→Sim `_0/_1` | 未确定顺序 |
| 掌面→`skin_palm_p` | 高置信候选，方向未知 |
| 块内row/column | Real未知、Sim确定 |
| Real raw vs Sim raw单位 | 不可比较 |
| Real processed vs Sim processed算法 | 不同且Real算法未知 |
| 接触事件 | Sim有，Real历史数据没有独立真值 |
| 载荷 | Sim有接触力诊断；Real没有已追溯标定 |

受控采集前不能计算逐taxelL1/L2，也不能把probe penetration当作真实力等价值。

## 8. 外部必须提供的材料

按优先级：

1. `config_mapping_zyhand.json`原文件、版本和hash；
2. `batch_test_v2`对应数据采集时期的完整源码/commit；
3. `Usb32`类、底层库、平台/驱动依赖和安全初始化说明；
4. splitter及raw/processed NPZ保存代码；
5. auto-zero、baseline、gating、filter、consistency和坏点处理算法；
6. calibration文件、方法和每session使用版本；
7. 16块安装图、接口编号、`-1/-2/-3`物理含义及row/column方向；
8. 传感器厂商型号、序列号、ADC/压力单位、量程、饱和和采样协议；
9. 能否只启用tactile而完全不连接arm/hand的官方操作说明。

## 9. 允许采集与停止标准

### 允许开始

必须满足第4.1节全部条件，并先由人工审核collector的设备初始化路径。首次运行只能是无接触、短时、全新输出目录的tactile-only dry acquisition。

### 立即停止

- 程序初始化arm或hand通信/命令对象；
- 任何关节目标、力、速度或位置命令被发送；
- mapping、processing或calibration版本未知；
- raw/processed shape或帧数不一致；
- 时间戳非单调、sync ID无法解释或持续timeout；
- 非目标patch成为主响应且无法由物理接触解释；
- 输出饱和、NaN/Inf、设备断连或温度/电气异常；
- collector试图覆盖历史`outputs/realdata/`。

## 10. 最终结论

### 软件侧采集就绪状态

**BLOCKED。** 当前没有可执行的真实Usb32采集栈，不能安全证明“只读触觉且不驱动机器人”。

### 仿真侧就绪状态

现有轨迹双路导出为READY；指定patch/位置的自动配对采集为CONDITIONALLY READY，需要将probe工具与双路导出合同整合。

### 下一项动作

**先向数据提供者取得外部采集工程、mapping、processing和calibration材料，并进行静态安全审查。** 在这些文件缺失时停止，不连接硬件、不猜测Usb32接口，也不伪造采集数据。

本阶段到此停止，等待人工审核。
