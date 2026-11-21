from pathlib import Path
import json,hashlib
O=Path('docs/writing_audit/ch04_material_audit_v1_2026-11-20');a=json.loads((O/'dataset_checks.json').read_text());idx=json.loads((O/'source_index.json').read_text())
def put(n,s):(O/n).write_text(s.strip()+'\n')
def tab(h,rs):return '|'+ '|'.join(h)+'|\n|'+'|'.join(['---']*len(h))+'|\n'+'\n'.join('|'+ '|'.join(map(str,r))+'|' for r in rs)+'\n'
v4=[r for r in a['npz_cases'] if r['kind']=='packaged_v4_development']
s='''# 第四章整章材料审计

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
'''
put('CH04_MATERIAL_AUDIT.md',s)
s='''# 已有数据集清单与实际统计

下列统计只覆盖明确列出的目录和字段。直接读取Parquet/NPZ、重算数值与Hash的证据为L2；历史语义错误、执行/回放成绩及视频PTS报告单独为L1。没有重新采集、物理重放或全量解码。

## 完整读取的三组LeRobot数值数据

'''
s+=tab(['目录','info Episode/Frame','实际Episode/Frame','state/action/tactile','触觉非零比例','划分'],[(d['path'],f"{d['declared_episodes']}/{d['declared_frames']}",f"{d['actual_episode_indices']}/{d['actual_frames']}",'39/13/712',f"{100*d['tactile_nonzero_fraction']:.6f}%",str(d['declared_splits'])) for d in a['lerobot']])
s+='''
每行数据集内state/action/tactile均finite，触觉非负。时间增量最小/最大、重复时间数量、frame_index连续性及每Episode数值摘要保存在dataset_checks.json，不以浮点timestamp与0.05绝对相等代替合理容差判断。RGB只核对特征声明/列存在，不读取全库图像内容；完整图像覆盖率UNKNOWN。

三个目录实际meta/tasks.parquet中的标签都是lift:物体名，不将task_index数量称为物理任务种类。独立Root、跨数据源物理初态关系与有效组划分UNKNOWN。旧5和旧30是历史合同问题版本；P2修复版是指定Sugar源的340帧导出，不是新增独立物体或轨迹来源。三组不能简单相加为36个独立Root。

## 修复版与源数值一致性（重新计算）

'''
s+=tab(['项','实际最大绝对差','统计范围'],[(k,a['fixed_export_numeric_check'][k],'340行，指定13D或712D字段') for k in ['action_vs_source_ctrl_max_abs','state_command_vs_previous_ctrl_max_abs','tactile_vs_previous_source_row_max_abs']])
s+='\n源manifest：`'+a['fixed_export_numeric_check']['source_manifest']+'`。源477行，尾部起点137，动作对应源ctrl[137:]，状态命令及触觉对应前一行。只是指定映射检查，不证明RGB像素或物理重新执行完全一致。\n\n## 十一条原始NPZ检查\n\n'
s+=tab(['来源','案例','帧数','触觉正值率','前状态命令最大差','qpos连续性','modalities映射'],[(r['kind'],r['path'].split('/')[-1],r['frames'],f"{r['tactile_nonzero_fraction']*100:.5f}%",f"{r['pre_action_command_max_error']:.9g}",r['physical_qpos_continuity'] if r['physical_qpos_continuity'] is not None else 'UNKNOWN：缺pre_qpos','一致' if r.get('export_action_matches_position_action') and r.get('export_tactile_matches_sensor_tactile') and r.get('export_state_matches_pre_state') else 'MISSING/未具备') for r in a['npz_cases']])
s+='''
全部11条读取的数值字段finite，触觉非负；timestamp为20Hz合同，next_timestamp时长最大误差见JSON。以上“连续”只比较保存的qpos/qvel数组，不是本轮MuJoCo重放。触觉tactile与sensor_tactile均保留，不能互换统计。

旧PickPlace有373行（按相邻转移比较）超过1e-6命令连续性容差，最大约0.100000024；原文件保持不动。旧Stack缺pre_qpos/pre_qvel，物理连续性不可核验。二者modalities.npz缺失，不能描述为全部旧记录已完成统一导出。

五个v4正式目录包的来源标记实际是开发准入：2727帧、4个登记asset_hash、5个lineage、5个登记content值；不是5个认证独立Root，也不是正式生产。登记的全部文件Hash匹配，视频文件存在5/5、源传感器字段存在5/5；文件存在和既有验证Hash不足以认证所有RGB时间对齐，本轮未解码。

## v4状态/精度对照

P38 comparison.json记录相同380帧Sink执行源在旧float32路径下最大qpos偏差1.628897且回放失败；完整积分状态+float64控制路径下偏差0且回放通过。比较原件已读取，数值为L1历史回放记录。本轮读到了380帧v4源，确认辅助数组存在、finite及dtype；未重放，不升级到L3。两个改动同时存在，不能做单因素因果归因。

## 重复、泄漏和规模边界

对三组LeRobot逐Episode的state/action/tactile完全相同数值摘要比较，未发现完全相同的全Episode摘要。这个检查不含RGB，也不排除截断子序列、轻微差异、同模板或同源重复。P2指定源与P1来源关联明确，不作为新独立Root；本轮没有获得完整跨版本Root lineage或组划分，因此全局泄漏结论UNKNOWN。

P44开发五任务与P32旧五任务有来源/初态关联，不累计为十个独立物理条件；P38摘要12条开发记录未经本轮全部读取，不作为当前全库规模。所有选定数据集仅按所列范围报告，不推出项目总Episode/Frame或完成小时。

服务器批次2目录及其数据库在本轮未访问，最新规模、容量和有效小时UNKNOWN。重复读取扩容索引只反映训练采样量，不能累计新增物理仿真时长。
'''
put('CH04_DATASET_INVENTORY.md',s)
