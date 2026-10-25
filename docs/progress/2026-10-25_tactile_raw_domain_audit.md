# P18.2-C：仿真原始触觉导出与严格匹配的 Sim/Real 域差异复核

日期：2026-10-25  
项目根目录：`/home/zmy/dex_hand_project`  
基线 HEAD：`6be6a346fc4fbdf5f3fb42c1ea7c34cae7b6376f`  
结果目录：`outputs/tactile_p18_2_2026-10-25/raw_domain_audit/`  
执行边界：未修改真实数据、物理参数、Strict、生产策略或checkpoint；未训练GAN/控制策略，未运行实机，未commit/push

## 1. Executive Summary

P18.2-C补齐了仿真侧处理前712D `mjSENS_TOUCH` 导出，并在冻结的session/root划分、20 Hz时序和CLOSE/HOLD/LIFT三个可比较阶段下复核了域差异。

主要结果：

- 24条P15轨迹全部完成双路导出，覆盖8个独立roots和3种grasp modes，每条280个Approach起点后的Strict replay帧；
- 仿真raw读取对qpos/qvel/ctrl/simulation time的最大影响为0；
- 独立克隆processor从raw逐帧重建production processed的最大误差为0；
- 所有raw/processed数组均为`[280,712]`、finite，并保存逐帧simulation time、stage、物体位姿和接触真值；
- 仿真触觉合同hash为`34855f80c0f57cf0c89554203b902305ccde9449380b73e320ed6765625b5751`；
- 三条代表性无缓存Strict复核保持原分类：xminus和pitch通过，roll因抬升不足失败；诊断读取没有改变Strict结果；
- 严格匹配阶段下，Real processed vs Sim processed的所有预处理、Logistic和MLP、3个seeds均为BA=1.000、ROC-AUC=1.000、`D_confusion=0`；
- Real raw vs Sim raw同样全部完全可分。该结果不表示raw物理单位等价，只说明即使去掉项目processed链路，两域的稀疏性、尺度、空间激活和时间统计仍显著不同；
- 仿真raw与processed对真实接触、多指接触和拇指对向标签均保留较高可分性，但这只在仿真接触真值上成立；真实侧没有相应逐帧真值，无法完成跨域任务保持验收；
- 当前证据不支持启动GAN。优先需要逐patch方向验证与严格配对的受控接触数据，而不是扩大判别器或生成器。

## 2. 仿真raw/processed信号合同

### 2.1 MuJoCo raw

代码证据：`source/sensors/tactile/dex_hand.py::read_raw`。

1. 每个taxel在MJCF中注册一个`mjtSensor.mjSENS_TOUCH`标量sensor；
2. `bind()`把712个sensor ID转换为`model.sensor_adr`地址；
3. `read_raw()`从`data.sensordata[addresses]`复制float32数组；
4. 顺序为仿真patch layout顺序，每patch内部row-major；
5. raw是MuJoCo touch sensor输出，不等价于真实Usb32 ADC计数或压力单位。

### 2.2 项目processed

代码证据：`source/sensors/tactile/signal_processing.py::TactileSignalProcessor.process`。

实际执行顺序：

1. `max(raw - deadzone, 0)`；
2. clip到`[0,saturation]`；
3. 幂律：`saturation * (value/saturation)^nonlinear_exponent`；
4. 有状态低通：`filtered = alpha*value + (1-alpha)*previous`；
5. 每patch执行8邻域crosstalk，随后可选Gaussian blur；
6. 可选Gaussian noise并再次clip；
7. 若`normalize=true`，除以saturation。

本轮配置：deadzone 0.002、saturation 1.0、exponent 0.7、lowpass alpha 0.45、crosstalk 0.12、Gaussian sigma 0、noise 0、normalize true。

### 2.3 策略输入与真实信号

- 仿真策略观测：`source/envs/rl_env.py::_get_observation`调用`tactile_sensor.read()`，即上述processed；
- 本轮仿真raw：同一`env.step()`结束后额外调用无状态`read_raw()`；
- 真实raw：Usb32 splitter输出，单位和电气含义未知；
- 真实processed：缺失的外部程序生成，算法和单位未知，数值可以超过1。

因此四者不能仅凭712维直接等同。

## 3. 双路导出实现与不干扰证明

新增独立程序：`apps/export_sim_tactile_raw.py`。

导出流程没有修改production sensor：

1. `env.step(control)`照常产生processed observation，并推进一次low-pass状态；
2. 保存该observation中的processed；
3. 调用无状态`read_raw()`保存同一仿真时刻的sensordata；
4. 独立processor clone接收同一raw，只用于校验，不影响env sensor；
5. raw读取前后逐帧比较qpos/qvel/ctrl/time。

### 3.1 完整导出结果

| 项目 | 结果 |
|---|---:|
| Episodes | 24 |
| Roots | 8 |
| Modes | xminus、pitch、roll |
| Frames/episode | 280 |
| Raw/processed shape | 280×712 |
| raw读取状态影响 | 0 |
| clone processed重建误差 | 0 |
| reset processed重建误差 | 0 |
| control time-step误差上限 | `2.77e-14 s` |
| finite检查 | 24/24通过 |

每episode还保存frame index、simulation time、stage、qpos/qvel/ctrl、物体pose、逐指object contact count/force、palm contact和task success。每个NPZ及数组均有SHA-256。

### 3.2 历史episode与Strict replay状态历史

当前Strict replay从Approach前一帧开始，原P15完整episode则经历settle/transit/pregrasp后到达Approach。低通processor具有历史状态，接触动力学也会放大微小状态差。因此：

- 新replay processed与原完整episode tactile的最大差异为0.889；
- replay qpos与原episode对应帧最大差异为0.188 rad；
- 这不是raw读取造成：同一新replay内部raw读取状态影响和processed重建误差均为0。

所以本轮把双路数据定义为“当前Strict replay语义下的同步raw/processed”，不把历史完整episode tactile冒充逐帧真值。

### 3.3 Strict抽查

无缓存重新回放：

| Mode/root | Strict | Final lift | Max lift |
|---|---|---:|---:|
| xminus/p00 | 通过 | 60.57 mm | 68.01 mm |
| pitch/p00 | 通过 | 61.23 mm | 67.21 mm |
| roll/p00 | 失败 | 47.70 mm | 52.12 mm |

与P15的模式级结论一致。结果见`strict_spot_check.json`。

## 4. 冻结数据集与stage匹配

沿用P18.2-B冻结划分：

- Real：16/4/4 sessions train/validation/test；
- Sim：5/1/2 roots train/validation/test；
- split SHA-256：`2e3d38eb055e9029f22fa75c0db2f0a81a95c413493b9f3c7dab9b996f70630d`。

相关模式始终随root进入同一split。真实每session最多2个trials。真实帧用`tactile_ts`降采样至目标20 Hz；仿真直接使用20 Hz控制帧。

Strict replay只包含Approach及之后，因此域分类仅比较：

- CLOSE；
- HOLD；
- LIFT。

BASELINE没有可靠的同一Strict初态双路记录，SLIP_INDUCE在仿真没有等价stage，二者均排除。没有为了增加样本而强行匹配。

真实降采样后的有效间隔仍受原采集抖动影响：

- 配置`2c295d254fdd`中位约57–66 ms，P05/P95约56–75 ms；
- 配置`d89a69f0904e`中位约57 ms，P05/P95约55–70 ms；
- 仿真稳定为50 ms。

这仍是可被判别器利用的时序差异，不能解释为传感器物理差异。

本轮形成310个域—信号—阶段单元。测试集每条轨道包含18个仿真单元和12个真实单元。

## 5. 配置版本与信号统计

冻结24个真实sessions来自3种配置版本，但经过“存在同步ID、双NPZ和目标stage”的筛选后，实际进入严格匹配分析的只有两个版本：`2c295d254fdd`和`d89a69f0904e`。第三种版本没有足够有效同步单元，因此不能报告其正式结果。

### 5.1 数值范围和稀疏性

| Domain/signal | Stage | 全部unit范围 | unit零值比例中位数 | unit P99中位数 |
|---|---|---:|---:|---:|
| Real processed，新配置 | CLOSE | 0–11.74 | 0.848 | 1.342 |
| Real processed，新配置 | HOLD | 0–9.65 | 0.850 | 1.596 |
| Real processed，旧配置 | CLOSE | 0–6.59 | 0.878 | 0.207 |
| Real raw，新配置 | CLOSE | 0–21940 | 0.429 | 3750 |
| Real raw，旧配置 | CLOSE | 0–14869 | 0.422 | 3157 |
| Sim processed | CLOSE | 0–0.909 | 0.934 | 0.021 |
| Sim processed | HOLD | 0–0.930 | 0.889 | 0.050 |
| Sim raw | CLOSE | 0–43.59 | 0.995 | 0 |
| Sim raw | HOLD | 0–41.93 | 0.992 | 0 |

Real processed超过1得到再次确认，没有clip。Sim raw非常稀疏，而crosstalk+low-pass使Sim processed出现更多小正值。Real raw则有约57%非零通道，明显不同于Sim raw约0.5–1.1%非零；但由于真实raw基线和单位未知，不能把该比例直接解释为接触覆盖。

完整逐配置、stage、signal的range、零值比例和P50/P90/P95/P99位于`analysis/signal_statistics.json`。

## 6. 严格匹配域判别结果

特征仍采用P18.2-B的127维方向不敏感region统计：每指指尖、两个指节合并、掌面、时间差分、自相关和相对激活量。不依赖4×8顺序或块内方向。

每种组合运行3个seeds，使用相同冻结split：

| Track | Preprocess | Probe | BA | ROC-AUC | D_confusion | Group bootstrap BA 95% CI |
|---|---|---|---:|---:|---:|---:|
| Processed/processed | common robust | Logistic | 1.000 | 1.000 | 0.000 | [1.000,1.000] |
| Processed/processed | domain robust | Logistic | 1.000 | 1.000 | 0.000 | [1.000,1.000] |
| Processed/processed | domain quantile | Logistic | 1.000 | 1.000 | 0.000 | [1.000,1.000] |
| Processed/processed | all three | MLP | 1.000 | 1.000 | 0.000 | [1.000,1.000] |
| Raw/raw | common robust | Logistic | 1.000 | 1.000 | 0.000 | [1.000,1.000] |
| Raw/raw | domain robust | Logistic | 1.000 | 1.000 | 0.000 | [1.000,1.000] |
| Raw/raw | domain quantile | Logistic | 1.000 | 1.000 | 0.000 | [1.000,1.000] |
| Raw/raw | all three | MLP | 1.000 | 1.000 | 0.000 | [1.000,1.000] |

所有seed范围也是1.000–1.000。CLOSE、HOLD、LIFT逐stage均为BA 1.000。

这排除了“P18.2-B只是因为Real raw对Sim processed路径不一致”这一单一解释。即便raw/raw和processed/processed分开，两域仍完全可分。但仍不能对六类潜在原因做唯一归因，因为以下因素同时存在：

1. raw单位和基线不同；
2. 真实processed算法未知；
3. 时序采样仍有抖动差异；
4. 真实物体/手部动作与Sugar Box仿真不配对；
5. patch内部方向和两个4×8块顺序未验证；
6. real stage不是物理接触真值。

所以结论是“差异在更严格控制后仍存在”，不是“已经证明某个单一传感器差异是根因”。

## 7. 任务信息保持

### 7.1 仿真接触真值probe

使用8帧窗口和当前回放逐指object contact真值，只在仿真侧训练线性probe：

| Signal | 标签 | BA（3 seeds范围） | ROC-AUC约值 |
|---|---|---:|---:|
| Raw | 任一指接触 | 0.966–0.971 | 0.993–0.996 |
| Processed | 任一指接触 | 0.968–0.989 | 0.998–0.999 |
| Raw | 多指接触 | 0.997 | 0.999 |
| Processed | 多指接触 | 0.997 | 1.000 |
| Raw | 拇指对向接触 | 0.964 | 0.993–0.995 |
| Processed | 拇指对向接触 | 0.973–0.976 | 0.999 |

当前processor没有明显抹掉这些仿真任务标签，且processed略有改善。但这不是跨域任务保持证据。

### 7.2 接触起始时间

以“任一taxel首次大于0”减去“首次digit-object contact”定义诊断偏差：raw与processed的中位数均为0 step，但P05为-34 steps、P95为0。

负偏差说明taxel会响应非目标物体接触、自接触或环境接触，因此“任一正值”不是可靠的object-contact onset检测器。该结果不能用于声称触觉提前34步预警。

### 7.3 真实侧不可验证项

真实数据只有人工stage，没有逐帧object-contact、多指接触、拇指对向或滑移真值。因此下列正式任务保持指标停止：

- Real接触/非接触；
- Real接触起点误差；
- Real多指与对向接触；
- SLIP_INDUCE跨域保持。

没有用stage替代这些真值。

## 8. 工程验证

新增：

- `apps/export_sim_tactile_raw.py`；
- `apps/run_tactile_raw_domain_audit.py`；
- `tests/test_sim_tactile_raw_export.py`；
- `tests/test_tactile_raw_domain_audit.py`；
- 本报告和独立结果目录。

验证结果：

- 定向pytest：4 passed；
- 相关Ruff：All checks passed；
- 24/24双路episode shape/finite/hash检查通过；
- 3个模式Strict无缓存抽查完成；
- 总产物约6 MB；
- 未修改`outputs/realdata/`。

主要索引：

- 总结：`raw_domain_audit/summary.json`；
- 仿真导出总结：`raw_domain_audit/sim_dual/summary.json`；
- 每episode双路NPZ和manifest：`raw_domain_audit/sim_dual/mode_*/...`；
- 严格复核：`raw_domain_audit/strict_spot_check.json`；
- 域结果：`raw_domain_audit/analysis/domain_results.json`；
- 任务保持：`raw_domain_audit/analysis/task_preservation.json`；
- 配置/stage统计：`raw_domain_audit/analysis/signal_statistics.json`；
- 数据清单：`raw_domain_audit/analysis/dataset_inventory.json`。

## 9. 是否进入GAN及下一项实验

### GAN结论

**不值得现在进入GAN训练。** 完全可分并不自动证明需要更强的生成模型。当前还存在明确的可控混杂：raw单位、真实processing、采样抖动、物体/动作分布和patch方向。GAN可能只消除这些数据集标识，同时损坏接触信息。

### 简单标定结论

Patch级robust/quantile是必要对照，但本轮全部仍为BA 1.000，不能单独解决。它们应保留为后续配对数据上的baseline，而不是宣布失败后直接跳到GAN。

### 下一阶段唯一最值得执行的实验

**逐patch受控配对接触采集。** 对真实手与仿真使用同一解剖patch、尽可能一致的接触位置/阶段和重复协议，同时记录：

- Real raw/processed、配置和时间戳；
- Sim raw/processed及真实接触真值；
- 四角/中心按压位置，用于确定块内方向；
- 两个4×8块的近/远端对应；
- 每点至少3次重复。

先在该小规模配对集上比较公共标定、patch robust和quantile mapping。只有简单方法后域probe仍显著可分、且双方都有可靠任务信息保持标签时，才有依据评估GAN。

本阶段到此停止，等待人工审核。
