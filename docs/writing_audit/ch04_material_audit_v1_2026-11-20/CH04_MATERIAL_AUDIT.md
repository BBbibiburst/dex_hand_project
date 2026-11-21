# 第四章整章材料审计

日期：2026-11-20；阶段2，未写正式正文、未编译、未修改生产程序/历史数据/划分/参考文献/前章。没有连接SQLite或访问服务器，没有重新执行轨迹、生成数据或训练。

## 就绪结论

**有条件具备撰写7000–8000 TeXcount正文的材料条件，建议进入阶段3。** 本章应定位为“版本化数据合同、语义/时序修复与有限数据质量验证”，不能定位为服务器全量数据集已完成验收。按本轮任务六节可组织约7200–7700统计量（规划判断，非已有正文计数）；4.1、4.2、4.4材料最充分，4.3有单样本精确转换证据，4.5能形成真实版本对照但缺全局Root及服务器最新规模。

无需新增实验即可写出初稿；若要声称正式1000小时/100资产验收或训练集无泄漏，现有材料不足，需另行读取对应正式数据库、文件与冻结分组，不应为了本章升级证据而自动补采或重建。

L0为定位，L1为报告/历史声明，L2为指定代码、原件、数组、Hash核对，L3为新执行验证。本轮没有L3。重新计算旧数组不等于重新执行机器人任务；历史回放报告的成功也不是本轮新增验证。

## 实际数据流程

旧路径：任务生成器在动作执行后记录qpos/qvel/ctrl/action/tactile → 候选与Strict记录 → LeRobot导出。动作是IK位姿还是低层位置命令，需要按版本核对。修复路径：从源ctrl选择可执行动作，使用前一条post-action源观测构成对应pre-action输入，或在重新执行导出时先读观测再执行控制。五任务在线路径通过OnlineEnvironment和记录适配器，在物理执行前采集状态与传感器、执行后补上实际控制和next_timestamp，只提交任务记录器接受的步骤，排除规划探测步骤。

打包路径将源state/position_action/sensor_tactile映射到modalities中的state/action/tactile，保留源manifest、完整模型及独立回放记录，核验文件Hash与合同后再登记队列。视频按显式采样配置与控制行索引关联，不要求所有版本RGB均与控制同频。

## 4.1 多模态数据表示

- 39D state来自受控执行器的actuator_length(13)、actuator_velocity(13)与上一位置命令ctrl(13)，不是39个关节位置，也不是包含物体真值的完整qpos。arm角度/角速度与hand腱长度/速度具有不同单位，不能全部标rad。
- position_action为arm7+hand6低层位置目标；ik_action的前7维为xyz+四元数、后6维为手部命令。两个13D不能互换，数字维数相同不证明语义相同。
- qpos/qvel是完整仿真状态，随自由物体数变化，含仿真特权信息；不能直接叫实机可观测state。末端目标可在IK动作中出现，但所查主数据合同没有证明所有版本都保存了实际测量末端位姿，不能把目标当观测。
- object_state为[T,N,7]，位置+四元数；N可能包含动态对象、固定Nut或target_region site，不应把所有条目都称独立运动物体。
- sensor_tactile为float32[T,712] processed观测，无已认证Pa单位；legacy tactile与其实际不同。RGB为独立MP4或LeRobot image/video特征，保存结构和内容验证分开。
- Episode是执行记录，Frame是序列行，seed是随机初始化参数，Root必须由来源与物理初态规则定义；不能由文件名、seed、job_id或lineage_hash数量直接推导独立Root。

可形成合同表和NPZ—Manifest—modalities—LeRobot关系图。机器人和触觉生成原理只引用第二章，任务/Strict定义只引用第三章。

## 4.2 采集时序与关联

production_recorder._before读取仿真时钟、pre-state、触觉和对象状态；step/step_physics推进后_after记录next_timestamp及实际position_action/float64控制。prepare_recorded_control供低层代码在覆盖控制缓冲区之前捕获状态，不能将当前版本的处理回填旧数据。传感器缓存以time和raw bytes为键，避免额外读观测再次推进IIR或抽新噪声。

当前validate_online_arrays检查state39、两种action13、sensor_tactile712的shape和float32；对象姿态、finite、非负、0.05s步进、next_timestamp时长及可选命令/物理连续性。require_causal开关决定是否实际审查连续性；返回的schema名称本身不能替代manifest版本和实际通过条件。缺少字段时不能填零制造通过。

控制/状态/触觉的20Hz与RGB采样分开。capture默认1920×1080/20Hz，speed配置可以使用640×360/5Hz，代码还支持控制频率整除的其他RGB频率。必须以实际online_contract及rgb_control_indices为依据，不能把默认参数写成服务器全部数据的实测值。

本轮发现旧PickPlace命令关联不满足当前检查，旧Stack缺pre_qpos/pre_qvel；详见问题清单。所选v4五任务样本均通过实际数组上的命令与qpos/qvel连续性比较，但这只是保存数据的局部一致性，不是完整物理重放。

## 4.3 转换与统一存储

collect_generated_lerobot的recorded_pre_action_indices按源ctrl尾部长度定位回放控制，验证其与目标controls一致，再返回start-1开始的前一观测行索引。重新执行导出分支在env.step之前写入状态/图像与即将施加的control；非重新执行分支恢复源前一行状态并使用对应触觉。两者都不能被描述为单纯把原action列复制出去。

P31 write_causal_source只负责把观测[:-1]与action[1:]配对、丢掉首动作并保存偏移元数据；其源校验主要是shape/finite，不自动把IK动作转换成关节动作。因此时序移位与动作语义修正是独立问题，“调用了causal函数”不够证明可训练。

当前five-task打包映射是position_action→action、sensor_tactile→tactile，并保存精确积分/控制辅助数组。LeRobot是另一导出形式，不能把v4 NPZ包自动称为已转换为LeRobot。现有三个LeRobot数据集都可读且维度合法，但两个早期版本仍受历史合同错误影响。

单物体修复版340行与源477行中后340条ctrl逐项比对，动作最大绝对差为0；state末13维与前一行ctrl、触觉与前一行源tactile的最大差也为0。这是指定数值字段转换的L2证明，不是全部state、RGB像素或所有导出目录已认证。

## 4.4 审计、清洗与恢复

可写成分层准入而不是删除失败数据：执行结果/Strict层、数据结构层、时序关联层、来源与版本层、训练角色层。一个Episode可以任务成功但关联错误，也可以是合法失败轨迹而不适合当成功专家标签。

production_queue使用状态机、lease token、事务及lineage/content唯一约束；register要求处于VALIDATING且verification.passed为True。重复源或语义内容进入DUPLICATE，重试超限进入QUARANTINED；只说明代码机制，不把模拟故障测试计为真实采集。部分成功/合法失败是否计入小时取决于冻结配置，不混为成功示教。

production_artifacts核对路径、文件Hash、源与规范数组逐项一致、模型/回放记录与源manifest绑定、job payload及formal admission。lineage绑定task/asset/source文件；content hash对时钟减首时间并舍入，排除integration_state/control_exact辅助字段，加入解码RGB摘要，避免只改绝对时钟或多存辅助字段产生“新数据”。Hash唯一不证明物理初态独立或无近似重复。

视频写入有有界队列、超时和失败传播，verify_video解码检查帧数、PTS、尺寸及解码摘要。该函数会运行已有视频解码，不是本轮已执行的全部工作；本轮仅核验所选文件Hash和历史验证记录，不声称重解码了所有视频。队列/文件恢复机制存在也不证明外盘故障后全量无损。

v4新增float64完整积分状态及控制，保存模型用于重放；已有Sink同条件对照支持旧精度状态合同不足，但未分离完整状态与控制精度各自贡献。它是数据可复现性证据，不能重复第三章抓取成功率或当作新增多样性。

## 4.5 数据统计与验证范围

本轮直接读取三个完整选定LeRobot目录的全部数值Parquet行，以及11条指定源NPZ（五条旧P32、五条v4五任务、一个v4 Sink对照），核对其中v4五包的登记文件Hash。数据集清单中的统计是本轮重算，不是服务器全库规模。

LeRobot的1550/14310/340行与info声明一致，分别有5/30/1个episode_index；state39/action13/tactile712均finite。task_index在这三组中分别映射2/3/1个“lift:物体”标签，均属Lift，不是2/3/1种操作任务。splits均仅声明train；没有认证的新验证/测试分组。数值全Episode摘要在这三组之间未发现完全重复，但不同长度、不同导出版本和同一源重用不受此结论排除。

五个v4包含2727个控制行，覆盖五任务，存在4个不同登记asset_hash、5个lineage及5个登记content值。Root数仍UNKNOWN。所有登记文件的SHA256匹配、缺失数0；动作/触觉/状态与源映射逐项一致。视频文件存在5/5不等于本轮认证了每帧RGB；admission明确为development_verified_not_formal_production，不是正式1000小时生产验收。

旧P32五条记录有2665行，不能与v4五条直接相加为独立新数据。统计与风险详见CH04_DATASET_INVENTORY.md和dataset_checks.json。历史P38 summary的“12条/315.55秒”不在本轮按全量重新验收范围，保持L1，不替换成当前全项目数量。

扩容读取索引也要单列：docs/fast_collection.md明确历史源复用和重复读取，索引时长不计新增物理小时。服务器当前可访问数据库、累计有效小时、全量Root及训练/验证泄漏均UNKNOWN，不能沿用旧看板推测当前状态。

## 4.6 小结范围

可总结合同定义、动作语义与时序修复、分层质量审计及有限原件验证，承接第五章对冻结触觉子集的使用。不能由格式合法、Hash匹配或任务成功单项推出“全部训练可用”。

## 最终决策

可以在不补做实验的情况下形成一版完整初稿；正式写作保留章节预算，4.5用版本对照、字段一致性和已发现风险解释，不扩写未经读取的服务器规模。需要缩减的是全局独立性/无泄漏/同步覆盖结论，而非靠工程描述填充。建议进入阶段3，等待用户与ChatGPT审核，不自行开始正文。

## 多源边界补充：Real独立触觉记录

已登记P44冻结触觉合同的Real/Natural Train与Validation角色（详情见库存补充节），未打开Real原始数组或Test/Holdout。其规模保持L1合同声明，不能与本轮实际读取的NPZ/LeRobot库存相加。Real窗口为历史不等间隔触觉，既不是完整动作示教，也不与Sim形成已认证同步物理配对。本章说明这些数据来源和使用范围，第五章再讨论伪配对与跨域表示。
