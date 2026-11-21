# 已有数据集清单与实际统计

下列统计只覆盖明确列出的目录和字段。直接读取Parquet/NPZ、重算数值与Hash的证据为L2；历史语义错误、执行/回放成绩及视频PTS报告单独为L1。没有重新采集、物理重放或全量解码。

## 完整读取的三组LeRobot数值数据

|目录|info Episode/Frame|实际Episode/Frame|state/action/tactile|触觉非零比例|划分|
|---|---|---|---|---|---|
|outputs/dp_2026-10-11/lerobot_5|5/1550|5/1550|39/13/712|12.729703%|{'train': '0:5'}|
|outputs/dp_p1_2026-10-11/lerobot_formal30|30/14310|30/14310|39/13/712|9.402879%|{'train': '0:30'}|
|outputs/dp_p2_2026-10-11/lerobot_single_sugar_fixed|1/340|1/340|39/13/712|11.097158%|{'train': '0:1'}|

每行数据集内state/action/tactile均finite，触觉非负。时间增量最小/最大、重复时间数量、frame_index连续性及每Episode数值摘要保存在dataset_checks.json，不以浮点timestamp与0.05绝对相等代替合理容差判断。RGB只核对特征声明/列存在，不读取全库图像内容；完整图像覆盖率UNKNOWN。

三个目录实际meta/tasks.parquet中的标签都是lift:物体名，不将task_index数量称为物理任务种类。独立Root、跨数据源物理初态关系与有效组划分UNKNOWN。旧5和旧30是历史合同问题版本；P2修复版是指定Sugar源的340帧导出，不是新增独立物体或轨迹来源。三组不能简单相加为36个独立Root。

## 修复版与源数值一致性（重新计算）

|项|实际最大绝对差|统计范围|
|---|---|---|
|action_vs_source_ctrl_max_abs|0.0|340行，指定13D或712D字段|
|state_command_vs_previous_ctrl_max_abs|0.0|340行，指定13D或712D字段|
|tactile_vs_previous_source_row_max_abs|0.0|340行，指定13D或712D字段|

源manifest：`outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p00/candidate_00/manifest.json`。源477行，尾部起点137，动作对应源ctrl[137:]，状态命令及触觉对应前一行。只是指定映射检查，不证明RGB像素或物理重新执行完全一致。

## 十一条原始NPZ检查

|来源|案例|帧数|触觉正值率|前状态命令最大差|qpos连续性|modalities映射|
|---|---|---|---|---|---|---|
|online_v2_development|lift_sugar_rgb_00|447|8.40497%|0|True|一致|
|online_v2_development|pick_tomato_rgb_00|638|15.29644%|0.100000024|True|MISSING/未具备|
|online_v2_development|stack_ring_rgb_01|376|18.26328%|0|UNKNOWN：缺pre_qpos|MISSING/未具备|
|online_v2_development|push_tomato_rgb_00|605|0.49540%|0|True|一致|
|online_v2_development|nut_retain_lower_rgb_00|599|12.71759%|0|True|一致|
|packaged_v4_development|lift_sugar_ac9b12bc48cbf5775e50|447|8.11119%|0|True|一致|
|packaged_v4_development|nut_000_3106428ada24fdc54dce|648|12.22799%|0|True|一致|
|packaged_v4_development|pick_tomato_10e460303770c865bf7e|638|15.33453%|0|True|一致|
|packaged_v4_development|push_tomato_84651ac09b7a0bc393d3|613|0.50269%|0|True|一致|
|packaged_v4_development|stack_ring_e0f34add79e08848e5b1|381|20.64275%|0|True|一致|
|v4_sink_precision_case|attempt_001|380|14.78600%|0|True|一致|

全部11条读取的数值字段finite，触觉非负；timestamp为20Hz合同，next_timestamp时长最大误差见JSON。以上“连续”只比较保存的qpos/qvel数组，不是本轮MuJoCo重放。触觉tactile与sensor_tactile均保留，不能互换统计。

旧PickPlace有373行（按相邻转移比较）超过1e-6命令连续性容差，最大约0.100000024；原文件保持不动。旧Stack缺pre_qpos/pre_qvel，物理连续性不可核验。二者modalities.npz缺失，不能描述为全部旧记录已完成统一导出。

五个v4已导出目录包的来源标记为开发准入：2727帧、4个登记asset_hash、5个lineage、5个登记content值；不是5个认证独立Root，也不是正式生产。登记的全部文件Hash匹配，视频文件存在5/5、源传感器字段存在5/5；文件存在和既有验证Hash不足以认证所有RGB时间对齐，本轮未解码。

## v4状态/精度对照

P38 comparison.json记录相同380帧Sink执行源在旧float32路径下最大qpos偏差1.628897且回放失败；完整积分状态+float64控制路径下偏差0且回放通过。比较原件已读取，数值为L1历史回放记录。本轮读到了380帧v4源，确认辅助数组存在、finite及dtype；未重放，不升级到L3。两个改动同时存在，不能做单因素因果归因。

## 重复、泄漏和规模边界

对三组LeRobot逐Episode的state/action/tactile完全相同数值摘要比较，未发现完全相同的全Episode摘要。这个检查不含RGB，也不排除截断子序列、轻微差异、同模板或同源重复。P2指定源与P1来源关联明确，不作为新独立Root；本轮没有获得完整跨版本Root lineage或组划分，因此全局泄漏结论UNKNOWN。

P44开发五任务与P32旧五任务有来源/初态关联，不累计为十个独立物理条件；P38摘要12条开发记录未经本轮全部读取，不作为当前全库规模。所有选定数据集仅按所列范围报告，不推出项目总Episode/Frame或完成小时。

服务器批次2目录及其数据库在本轮未访问，最新规模、容量和有效小时UNKNOWN。重复读取扩容索引只反映训练采样量，不能累计新增物理仿真时长。

## 实机触觉与迁移冻结子集（只登记已有合同）

补充来源：outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/data_contract.csv。该已有合同声明Real Train 5153窗口/32 Session、Validation 780窗口/4 Session；Natural Train 4049窗口/50 Root、Validation 905窗口/10 Root。这是L1冻结合同，不是本轮重新读取Real原始数组后的规模认证，不计入上方Episode/Frame库存或新增物理小时。

合同明确Real保留历史不等间隔帧，Natural来源为20Hz连续仿真；物理通道对应为UNKNOWN/index-space only。Real触觉窗口不是具有机器人动作、RGB和Sim同步物理配对的多模态Episode。迁移子集的开发暴露角色及具体分组应由第五、六章材料说明，不把Validation重新称新盲测。Real原始驱动/处理标定及完整多模态覆盖本轮UNKNOWN；本章按来源分开组织，不强行补齐。
