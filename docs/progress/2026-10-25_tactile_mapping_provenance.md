# P18.1：真实触觉通道映射与采集算法溯源

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
Git HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`（`main`）  
执行性质：只读溯源；未修改源代码、配置、原始数据或checkpoint，未训练、未运行实机、未commit/push

## 1. Executive Summary

本轮没有找到可信的 `config_mapping_zyhand.json`、`Usb32` 实现或真实 processed 算法。

- `CONFIRMED`：79份真实session配置都指向同一个相对路径：`Usb_API_stable_Tujian_/spliter/config_mapping_zyhand.json`。
- `CONFIRMED`：较新session还记录原采集工程路径为 `D:\Robot\xxx_aaa\batch_test_v2\configs\collection_config.yaml`，trial路径位于 `D:\Robot\xxx_aaa\output_v2\...`。这说明数据来自另一台Windows主机上的独立工程，不是当前Linux仓库直接生成。
- `CONFIRMED`：对项目工作区、父目录、`/home/zmy`常用可访问路径的定向搜索，没有找到 `xxx_aaa`、`batch_test_v2`、`Usb_API_stable_Tujian_`、`spliter`、映射文件或`Usb32`源码候选。
- `CONFIRMED`：当前所有本地/远程引用分支的提交历史中，对关键字符串和历史路径的检索均无命中；Git不可达blob的内容检索也无命中。没有证据表明这些文件曾进入本仓库。
- `CONFIRMED`：1645个processed NPZ中，patch ID→shape完全稳定：`00/03/06/09/12`为`7×8`，每组其后两个ID为`4×8`，`15`为`7×16`。
- `INFERRED`：该模式很像“五个手指各由1个指尖块+2个指节块，外加1个掌面块”，且shape集合与仿真完全一致；但缺少mapping文件，无法确认五组对应哪根手指、两个`4×8`块谁近端/中端，也无法确认每块行列方向、翻转或旋转。
- `CONFIRMED`：真实raw与processed并非可由已保存配置证明的简单统一缩放。一个近期完整trial的抽样相关系数为0.571（仅processed>0位置为0.737），processed存在大量置零且最大值5.04；全库抽样最大值9.90。
- `BLOCKED`：`auto_zero_calibration`、`processed_gated_filter`、`consistency`等配置只说明存在这些开关/系数，缺少执行代码，不能据此还原公式，也无法可靠解释processed为何超过1。

最终判定为 **D：本机和Git历史中均无足够证据，需要向数据提供者索取采集工程和映射文件，或开展受控接触验证。**

## 2. 工作区与本机搜索

### 2.1 搜索范围

执行了以下只读定向搜索：

- 当前仓库及父目录；
- `/home/zmy`，排除Conda包缓存、普通缓存及34 GB原数据payload的重复扫描；
- 文件名和目录名：`config_mapping_zyhand.json`、`*mapping*zyhand*`、`Usb_API_stable_Tujian_*`、`spliter`、`*usb32*.py`、`xxx_aaa`、`batch_test_v2`；
- 文本引用：`Usb32`、`tactile_raw_split`、`tactile_processed`、`processed_gated_filter`及校准/归一化关键词。

### 2.2 结果

`CONFIRMED`：除`outputs/realdata/*/session_config.yaml`和`source_config.yaml`中的引用外，没有找到映射文件、采集工程或处理源码。当前仓库中的`source/sensors/tactile/`是MuJoCo仿真实现，不是`Usb32`真实采集实现。

`CONFIRMED`：真实数据目录没有README、硬件说明、mapping、calibration文件或采集脚本。配置中的`calibrate_file: null`不等于“无需标定”；同时存在`calibrate: true`和部分session的`auto_zero_calibration: true`，实际算法仍依赖缺失代码。

## 3. Git历史与对象检查

### 3.1 分支与历史

检查了：

- `main`；
- 本地`experiment/grasp-v1`；
- `origin/main`、`origin/experiment/grasp-v1`；
- 所有可达提交的路径变化和字符串历史。

对以下字符串执行`git log --all -S`及全历史name-status检索，均无命中：

- `config_mapping_zyhand`；
- `Usb32`；
- `tactile_raw_split`；
- `processed_gated_filter`；
- `Usb_API_stable_Tujian`、`batch_test_v2`、`xxx_aaa`。

### 3.2 不可达对象

`CONFIRMED`：`git fsck --full --no-reflogs --unreachable`显示存在若干历史不可达blob/tree；对体积≤5 MB的不可达blob做上述关键词内容检索，无命中。没有执行恢复、checkout或写入Git对象。

因此没有可报告的历史commit ID或历史路径。结论不是“文件绝对不存在于任何备份”，而是：**当前仓库所有可达历史与可检查的本地Git对象都没有证据。**

## 4. 真实数据生成来源证据链

### 4.1 已确认链路

从 `outputs/realdata/20260817_105909_slip_3D_liulengzhui_30/` 的完整trial可确认：

1. `session_config.yaml`/`source_config.yaml`指定`sync_group.sample_hz: 60`，设备为tactile/hand/arm；
2. tactile配置为`fake: false`、`sensor_class: Usb32`、`sensor_shape: [32,32]`、`splitted: true`；
3. mapping引用为`Usb_API_stable_Tujian_/spliter/config_mapping_zyhand.json`；
4. `samples.jsonl`含同步ID、sample/tactile时间戳、stage、hand/arm状态；
5. `tactile_raw_split.npz`与`tactile_processed.npz`按相同帧编号及patch ID保存；
6. `trial_meta.json`和manifest指向原Windows采集目录；
7. stage/commands记录Close、Lift、Hold、Slip Induce等动作。

可证链路到此为止：

`batch_test_v2配置` → `Usb32（实现缺失）` → `mapping_file（缺失）` → `16块raw NPZ` → `processed步骤（实现缺失）` → `processed NPZ/samples`。

### 4.2 配置版本漂移

`CONFIRMED`：数据跨多个采集配置版本：

- 早期session常为100 Hz tactile、60 Hz camera/sampling配置，`zero_on_baseline: true`、`processed_gated_filter: true`；
- 后期主要为60 Hz同步、30 Hz相机，`auto_zero_calibration: true`、`zero_on_baseline: false`、`processed_gated_filter: false`；
- 16项`consistency`系数在配置中存在，后几项可为1.2/1.4/1.7。

`BLOCKED`：没有采集程序，无法确认这些字段在哪个处理阶段应用，也不能假设所有79个session共享同一raw→processed公式。后续数据兼容性验证必须按配置版本分层，不能直接合并。

### 4.3 raw与processed实证

近期完整trial `.../trial_0030`抽取30帧、每帧712值：

| 项目 | 结果 |
|---|---:|
| raw/processed shape | 30×712 / 30×712 |
| processed正值数量 | 3352 |
| 全部位置Pearson相关 | 0.571 |
| processed>0位置相关 | 0.737 |
| 正值位置raw范围 | 20–10729 |
| processed范围 | 0–5.0387 |

`CONFIRMED`：processed保留了部分raw变化关系，但包含门控/基线/校准类非线性或分段行为的迹象。

`MISSING`：不能仅靠输入输出样本唯一反演算法。特别是以下内容无法确认：

- raw是ADC计数、压力还是其他量；
- 零点计算和符号方向；
- gated filter公式与状态初始化；
- `consistency`的应用方向；
- 坏点补偿、裁剪、饱和和归一化；
- processed>1的确切原因与单位。

## 5. 真实patch稳定结构

对全部1645个可读processed NPZ检查首帧，所有文件均满足下表，没有shape异常：

| Real patch ID | Shape | 出现文件数 | 可确认的真实名称/位置 | 行列方向 |
|---|---:|---:|---|---|
| 00 | 7×8 | 1645 | 未知；形状符合某一指尖 | 未知 |
| 01 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 02 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 03 | 7×8 | 1645 | 未知；形状符合某一指尖 | 未知 |
| 04 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 05 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 06 | 7×8 | 1645 | 未知；形状符合某一指尖 | 未知 |
| 07 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 08 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 09 | 7×8 | 1645 | 未知；形状符合某一指尖 | 未知 |
| 10 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 11 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 12 | 7×8 | 1645 | 未知；形状符合某一指尖 | 未知 |
| 13 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 14 | 4×8 | 1645 | 未知；同组指节候选 | 未知 |
| 15 | 7×16 | 1645 | 唯一shape与仿真palm吻合；仍需mapping确认 | 未知 |

patch ID顺序稳定是重要证据，但不是解剖映射证据。

## 6. 与仿真布局逐块对照

仿真定义位于 `source/sensors/tactile/fitting/layout.py::dex_hand_patch_layout`，展平顺序位于 `source/sensors/tactile/dex_hand.py::DexHandTactileSensorBase`。仿真每指顺序是segment 0、segment 1、fingertip，和真实文件中每组三块的shape顺序`7×8,4×8,4×8`并不相同。

### 6.1 仿真确定布局

| Sim patch | Shape | Flat interval |
|---|---:|---:|
| `skin_0_0_p` | 4×8 | [0,32) |
| `skin_0_1_p` | 4×8 | [32,64) |
| `skin_0_2_p` | 7×8 | [64,120) |
| `skin_1_0_p` | 4×8 | [120,152) |
| `skin_1_1_p` | 4×8 | [152,184) |
| `skin_1_2_p` | 7×8 | [184,240) |
| `skin_2_0_p` | 4×8 | [240,272) |
| `skin_2_1_p` | 4×8 | [272,304) |
| `skin_2_2_p` | 7×8 | [304,360) |
| `skin_3_0_p` | 4×8 | [360,392) |
| `skin_3_1_p` | 4×8 | [392,424) |
| `skin_3_2_p` | 7×8 | [424,480) |
| `skin_4_0_p` | 4×8 | [480,512) |
| `skin_4_1_p` | 4×8 | [512,544) |
| `skin_4_2_p` | 7×8 | [544,600) |
| `skin_palm_p` | 7×16 | [600,712) |

### 6.2 当前能建立的映射表

| Real IDs | Real shape | Sim候选 | 证据 | 置信度 | 未解决歧义 |
|---|---:|---|---|---|---|
| 00,03,06,09,12 | 7×8 | 五个`skin_*_2_p` | shape集合唯一匹配 | 中（仅类型） | 哪根手指；row/col翻转/旋转 |
| 01,02,04,05,07,08,10,11,13,14 | 4×8 | 十个`skin_*_0_p/_1_p` | shape集合匹配 | 低 | 手指、近/中节、方向全部未知 |
| 15 | 7×16 | `skin_palm_p` | 两边唯一7×16 | 中高（仍非文件证明） | 行列方向、镜像/旋转 |

`INFERRED`：真实ID呈五组三块结构，每组`7×8 + 4×8 + 4×8`，很可能对应一根手指；但无法证明组0–4与仿真finger 0–4的排列相同。没有执行自动重排。

## 7. 决策与最小缺失项

### 7.1 分类

本轮属于 **D**，不是A/B/C：

- 没找到映射；
- 没找到采集程序；
- 没找到processed算法；
- 当前数据没有保存代码版本或mapping哈希；
- Git历史也不能补足。

### 7.2 应向数据提供者索取

按优先级：

1. `D:\Robot\xxx_aaa\Usb_API_stable_Tujian_\spliter\config_mapping_zyhand.json`原文件及其hash；
2. `batch_test_v2`采集工程对应2026-06和2026-08数据的commit/archive；
3. `Usb32`类、splitter和raw/processed保存代码；
4. gated filter、auto-zero、consistency和calibration实现；
5. 真实手上16块的安装图、patch名称、接口编号和row/column方向；
6. 传感器产品型号、ADC/压力单位、量程、采样协议和标定记录；
7. 每个session所用代码版本与mapping版本。若无法提供，应明确承认无法追溯版本。

## 8. 下一项唯一最值得执行的实验

在取得mapping/采集代码后，执行**受控单patch接触映射验证**：

- 分别只按压五个指尖、十个指节区域和掌面；
- 每个区域至少重复3次，记录raw、processed、同步时间戳和视频；
- 检查声明patch是否成为主响应块，并确定row/column增大方向；
- 对每块测试四种候选方向（原向、上下翻转、左右翻转、180°），用已知按压位置消除歧义；
- 冻结验证后的`real_patch_id → anatomical patch → orientation → sim flat slice`表及hash。

在这一实验通过前，不应训练GAN，也不应把真实712维直接重排到仿真顺序。

## 9. 最终回答

### 是否找到了可信通道映射？

**没有。** 找到了稳定的16块shape结构和强烈的布局线索，但缺失官方mapping，无法证明手指编号、指节顺序和块内方向。

### 是否找到了真实processed算法？

**没有。** 只找到配置字段和raw/processed输出。数据统计能证明不是简单统一缩放，但不能唯一恢复真实算法。

### 下一项唯一最值得执行的实验是什么？

**向数据提供者取得mapping与采集工程后，完成逐patch受控接触验证，冻结真实—仿真映射表。** 若文件无法取得，则直接开展同样的受控接触实验重建映射，但结果必须标为实验重建映射，而非原始采集合同。

本轮只新增本报告，未改动P18报告或任何其他文件。
