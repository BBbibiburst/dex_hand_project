# 机器人仿真、专家示教与数据构建证据档案

版本：2026-11-19 V2。只读检索、代码审计和历史产物抽检，未执行物理仿真、训练、实机操作或服务器操作。

## 证据等级与统计规则

- L0：已定位文件或记录，尚未核验实现/产物，不可直接作为正文定量结论。
- L1：报告、配置、清单或汇总已核对，仍不能替代逐Episode实际文件验证。
- L2：本轮对指定函数或具体产物作交叉检查；只对列明范围有效，不代表整条路线或全量数据最终验收。

本档案新增机器可读核验记录见 `docs/writing_audit/v2/robotics_source_audit.json`，包含源路径、SHA256、读取字段及抽检结果；复现脚本为 `scripts/audit_robotics_writing_sources.py`。脚本只读取列出的本地JSON、五条开发Episode NPZ及两个LeRobot导出的Parquet，不读触觉研究Test/Holdout，不调用MuJoCo或训练。

R编号属于论文证据档案。DATASET-P32、DP-P3、TACTILE-P43等须带路线前缀，不能因为P编号相同而互相替代。尝试次数、候选数、有效对象案例、成功Episode、独立Root、资产Hash和小时数分别统计。

## R-01 机器人模型、环境与控制

**问题/目标**：核实RM75B与Dex Hand组合、执行器与任务接口。

**实现证据（L2函数级）**：

- `source/robots/arms/rm75b.py`：RM75B_ARM注册，7个position actuator，ee_site为right_hand_site，controller_kind为position_ik，手安装旋转(-90,-90,0)度。
- `source/robots/hands/dex_hand.py`：DEX_HAND_POSITION_ACTUATORS列出4个手指推杆及拇指旋转/抓握共6个驱动，注册position控制与触觉factory。
- `source/envs/manipulation/factory.py`、`source/envs/manipulation/base.py`、各任务文件已定位（L0；本轮未逐函数核查全部配置）。XML由上述descriptor通过asset_path加载，准确资产版本、闭链等式约束及控制器数值参数仍需在对应小节核验。

**可写**：7臂+6手执行命令的仿真组合与注册机制。**不可写**：13个独立物理关节；已完成真实机器人控制或动力学标定；仅凭descriptor证明与硬件完全一致。**正式使用**：可作实现事实，几何精度/真实性暂UNKNOWN。

## R-02 曲面阵列、712D通道与处理链

**实现证据（L2所列调用链）**：

- `source/sensors/tactile/dex_hand.py`：backend为simple_box；`_add_touch_sensor`创建mjSENS_TOUCH并关联site；`bind`取得sensor地址；`read_raw`取mjData.sensordata；`read`交给TactileSignalProcessor。不能描述为touch_grid插件输出。
- `source/sensors/tactile/fitting/surfaces.py`：segment、mesh-uv、fingertip-ellipsoid布局函数入口已核对；实际拟合参数、各Taxel面积和法线尚需对应版本核验。
- `source/sensors/tactile/signal_processing.py`：可见高斯滤波核、邻域串扰等处理实现。配置项存在不等于每个历史Episode均启用；必须读取对应manifest metadata。
- 已有物理语义审计见 `EXPERIMENT_EVIDENCE.md` E-P40及 `SYMBOLS_AND_TERMS.md`。16Patch布局及712通道需保留其固定顺序；Real逐Taxel物理对应仍未认证。

**实物输出交叉检查（L2抽样）**：R-05五个开发Episode都有float32合同来源的712维传感器数组，全部finite且非全零。本轮脚本保存实际shape/非零率；未重新生成数据。

**可写**：曲面适配的仿真site阵列、原生touch读取和版本化信号处理。**不可写**：刚体接触柔顺参数等于真实硅胶形变；processed具有已认证Pa/N单位；默认阵列与实机物理坐标完全一致。C8离线软触觉重建和原生产传感器不是同一个版本。

## R-03 分级候选生成与可达扩展

**实现证据（L2指定函数，非全量算法复验）**：

- `source/grasping/graspqp_adapter.py::refine_candidates_with_graspqp`调用官方graspqp.metrics的GraspSpanMetricFactory；实际优化手平移、6D旋转参数与六驱动fraction logits。应称官方指标的欠驱动适配，不声称GraspQP为本人原创。
- `source/grasping/dexevolve.py`明确为DexEvolve-style refinement；`DexEvolveConfig`定义population、offspring、mutation、接触及扰动目标，调用execute_grasp。不是仅凭名称认定完整复现第三方全部算法。
- `tools/grasping/batch_grasp_edit.py`：`_preflight_lattice`及lattice计数解析/分级调度可定位；`source/rl/grasp_edit/templates.py`是相关模板入口。腕部展开、各级候选去重和连续IK详细公式本轮只定位（L0），正文前需读具体函数。
- `source/grasping/executor.py`：候选rank、hand targets与位姿转换已定位。GraspQP几何候选不自动具有RM75B可达性。

**历史清单（L1）**：`outputs/dex_hand_top100_v2/summary.json`含100条不同object_id，状态98 LATTICE_SUCCESS、2 RL_SUCCESS。本轮按实际results重计；这是历史pipeline状态，不是100条FINAL Strict复验、不证明算法对比优势，独立Asset Hash数UNKNOWN。

**可写**：候选细化—进化搜索—机器人可达筛选/腕姿态扩展—必要时后续优化的分级实现。**不可写**：各级成功率为相同分母，或100物体均为新执行Strict通过。正式算法效果需逐候选清单、失败原因及版本一致的对照，缺失则不写优越性。

## R-04 连续轨迹、碰撞筛选与Strict Replay

**实现证据（L2）**：

- `source/verification/strict_replay.py`定义FINAL配置：lift 0.055m，speed/angular speed各0.10，penetration 0.003m，尾部contact/grasp至少0.80，table contact最多0；以实际版本为准，不将阈值伪称通用物理标准。
- `source/grasp_pipeline/replay.py::replay_grasp_trajectory`是另一回放接口，按尾部成功比例判定；不能和FINAL Strict等同。
- `source/grasping/executor.py`记录qpos/qvel/ctrl/action、对象姿态及接触统计；旧Recorder与新pre-action adapter不同，见R-07。

**历史实际结果（L1文件已读）**：`outputs/five_task_p32_b_2026-11-03/lift_sugar_strict.json`记录Lift开发模板Strict。与该目录五任务成功结果分开，不将Lift Strict传播到其他任务。

**缺口**：本轮不重跑Strict；全量不同对象的FINAL JSON、模型Hash及源轨迹绑定仍待按章查证。其他四任务有自己的成功判据，不默认具有相同Strict合同。

## R-05 五任务与专家示教案例

历史批次汇总按实际JSON重计，均为L1且**不能作为最终逐文件验收**：

|路径（outputs下）|汇总实际记录|重要限制|
|---|---|---|
|dex_hand_top100_v2/summary.json|100对象ID，98 LATTICE_SUCCESS、2 RL_SUCCESS|不是统一Strict100/100|
|pick_place_top100_final/summary.json|100条SUCCESS，100条cached|缓存汇总，不是本轮100次新执行|
|stack_task_valid_v24/summary.json|task-valid筛选48对象：43 SUCCESS、5 FAILED|不能写Top100全成功，筛选域要交代|
|push_top100_final/summary.json|100条SUCCESS，98条cached|历史成功状态需对应manifest与当前版本|
|nut_assembly_top100_final/summary.json|100条variant SUCCESS，无object_id字段|100参数variant不能认证100种独立资产；后续复现失败必须保留|

进一步L2抽查同一开发目录 `outputs/five_task_p32_b_2026-11-03/` 的下列真实NPZ（每行是 `case/source_episode/episode.npz`）：

|任务/用例|实际sensor_tactile shape|正值比例|时间戳0.05s连续|
|---|---|---:|---|
|Lift/lift_sugar_rgb_00|447×712|0.0840497|是|
|PickPlace/pick_tomato_rgb_00|638×712|0.1529644|是|
|Stack/stack_ring_rgb_01|376×712|0.1826328|是|
|Push/push_tomato_rgb_00|605×712|0.0049540|是|
|Nut/nut_retain_lower_rgb_00|599×712|0.1271759|是|

五条触觉均finite，非全零。`summary.json`给出对应success与合同、视频检查记录（L1），`docs/progress/2026-11-03_p32_b_online_recorder.md`记载失败及Nut控制对照（L1）。本轮没有解码全部视频、逐帧复算任务success或Root身份，不能据此认证完整联合验收。这些是5个指定开发案例，不是5个已确认独立Root，不累计为正式生产新增小时。

Nut原始及降低抓点仍失败、保持抓持命令后单例成功，是历史对照；不改写成跨seed稳定成功。Stack第二对象只对具备object_state及对象名的具体新合同成立，不回填旧NPZ。

## R-06 LeRobot实际构建规模与字段

本轮读取info及全部所列Parquet行/episode_index/tactile，结果为L2文件内容抽检：

|导出目录（outputs下）|info Episode/帧|实际唯一episode_index/Parquet行|触觉检查|
|---|---:|---:|---|
|dp_2026-10-11/lerobot_5|5 / 1550|5 / 1550|每行712，finite，正值比例0.1272970|
|dp_p1_2026-10-11/lerobot_formal30|30 / 14310|30 / 14310|每行712，finite，正值比例0.0940288|

原件：各目录`meta/info.json`及`data/chunk-000/file-*.parquet`；脚本记录每个文件Hash与行数。5条版本使用video feature，30条版本image feature，不能混为同一个视频编码合同。info声明state39、action13、tactile712、timestamp/frame_index/episode_index/task_index；这些字段声明不自动证明动作语义或采样因果关系。

**可写**：这些指定导出在本机实际存在，行数与metadata一致。**不可写**：30个独立物理Root、30条成功Strict示教、5+30可直接去重相加、所有历史RGB与触觉实机同步。实际物理源lineage/重复内容、全量图像与动作字段、PTS及成功率本轮未认证，标UNKNOWN。目录名formal30不是质量证书。

其他LeRobot目录仅定位L0，不汇总为更大数据量，不把重复导出算新增采集。真实触觉记录与仿真Episode没有已证明同步物理配对。

## R-07 关联机制、频率与版本

L2代码：`source/data/production_recorder.py::validate_online_arrays`要求state[T,39]、position_action[T,13]、ik_action[T,13]、sensor_tactile[T,712]及object_state[T,N,7]；检查float32、finite、非负触觉、时间增量0.05s、next_timestamp间隔，可选因果连续性，v4另检查exact replay状态/control float64。SCHEMA当前为five-task-online-v4，不能称所有历史文件都是v4。

`OnlineEnvironment`在动作前采集；旧 `executor._Recorder.append`记录env.step之后观测。`docs/progress/2026-11-03_p32_b_online_recorder.md`说明RGB在相同MuJoCo状态捕获及PTS偏移，属于指定开发配置的记录，不是全部产物本轮复验。速度配置可能降低RGB频率，不能沿用20Hz视频到每个生产版本；本机未找到服务器同名batch2配置原件，服务器实际生效参数UNKNOWN。

能证明的是特定仿真Episode内关联实现与指定样本时间连续，不是跨Real/Sim物理同步，更不是所有生产数据已逐帧验收。动态滤波状态与重复观测处理需保留配置版本。

## R-08 采集、恢复、计账与质量控制

L2代码：`source/data/production_queue.py`含SQLite WAL、事务、lease token及过期恢复，episodes表对lineage_hash/content_hash设唯一约束，区分EXPORTED、DUPLICATE、QUARANTINED。机制存在不等于任意外盘断连都能恢复且数据完全无损。

L1原件：`outputs/five_task_p38_exact_replay_2026-11-03/summary.json`登记12 unique_valid_development_episodes、315.55开发秒、9成功资产、new_formal_production_seconds=0；未核对所有源文件，数字只作为该历史summary的声明，不替代全量审计。

服务器当前数据库/挂载盘在本机不可访问，当前有效小时、各任务成功数、独立Asset Hash、磁盘容量、触觉覆盖率均UNKNOWN。用户历史看板快照只能以其明确时间和来源引用，不用旧快照推测2026-11-19状态。论文定稿需要从服务器实际数据库和最终文件报告读取；本轮不请求停止、重启、修改生产程序。

重复采样索引读取时长不计新增物理仿真小时。失败Episode是否计入有效时长依具体冻结配置，不混淆最初默认仅成功与后期合法失败计入的政策。

## 按章使用与缺口

第二章优先R-01/R-02/R-05；第三章R-03/R-04/R-05；第四章R-06/R-07/R-08；第五章只使用实际冻结迁移子集，不将生产全量替代其训练规模。

最先补足的材料核验（不是新增训练）：准确XML/控制器及拟合配置；候选级可达/失败清单；Strict与普通Replay的版本绑定；Parquet源lineage及动作因果关系；服务器最终DB/文件验收；第三方GraspQP/DexEvolve原文与依赖版本。若暂缺则限制表述，不编造质量提升或新增独立样本。

没有本轮新增实机验证、模型训练、物理采集。所有结果均按各自证据层级进入论文，不依据summary单独认定最终验收。
