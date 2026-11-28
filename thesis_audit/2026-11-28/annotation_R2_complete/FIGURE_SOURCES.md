# 本轮图稿与来源

九幅替换、一幅新增；保留此前已核对的控制器、处理算子、热图、模型和伪代码。下列图号来自最终编译aux，页码为正文印刷页码。

|图号/页码|内容|图稿|重绘脚本 / 原始来源Hash|
|---|---|---|---|
|2-1 / 13|当前机器人模型、控制与712维历史观测|thesis/figures/annotation_R2/platform_actual_architecture.pdf|draw_architecture.py / platform_actual_architecture_sources.json|
|2-3 / 15|实际关节模型、固定腱系数和执行器|thesis/figures/annotation_R2/joint_tendon_geometry.pdf|draw_geometry.py / joint_tendon_geometry_sources.json|
|2-7 / 22|指节、指尖、手掌三类算法实际中间结果|thesis/figures/annotation_R2/sampling_processes.pdf|draw_sampling.py / sampling_processes_sources.json|
|3-1 / 32|Sugar Box候选几何、历史执行及物体运动|thesis/figures/annotation_R2/grasp_evidence_pipeline.pdf|draw_grasp_evidence.py / grasp_evidence_sources.json|
|3-2 / 34|候选接触、进化记录及另物体PPO保存动作|thesis/figures/annotation_R2/candidate_geometry_feedback.pdf|draw_grasp_evidence.py / grasp_evidence_sources.json|
|3-3 / 36|已保存末端目标、预检残差及执行状态|thesis/figures/annotation_R2/wrist_recorded_targets.pdf|draw_grasp_evidence.py / grasp_evidence_sources.json|
|3-4 / 38|保存状态的运动学重建与历史Strict结果|thesis/figures/annotation_R2/recorded_tracking_strict.pdf|draw_tracking.py / recorded_tracking_strict_sources.json|
|3-6 / 43|五任务原始视频的实际阶段帧|thesis/figures/annotation_R2/five_task_phase_records.pdf|draw_task_phases.py / five_task_phase_records_sources.json|
|3-7 / 45|装配成功失败记录的高度、接触与阶段|thesis/figures/annotation_R2/assembly_failure_records.pdf|draw_failure_cases.py / assembly_failure_records_sources.json|
|4-1 / 48|v4与Sugar导出两个真实来源关系|thesis/figures/annotation_R2/data_lineage_instances.pdf|draw_lineage.py / data_lineage_instances_sources.json|

实际几何和保存记录不等于本轮新物理验证。候选图的GraspQP与进化代表不是同一个候选的连续优化轨迹；PPO为另一物体的条件分支。运动学重建不重新执行控制器；447行来源与322行Strict回放不混为同一次记录。失败图仅用对应失败NPZ，不借用成功截图。所有代码可读；重画需要原始来源，本TeX包仅依赖已导出的图稿。
