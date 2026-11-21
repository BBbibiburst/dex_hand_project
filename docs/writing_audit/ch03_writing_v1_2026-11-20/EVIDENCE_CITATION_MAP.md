# 论点、文献与内部证据对应

本轮不新增或修改BibTeX；复用以下已核验键：

|键|正文用途|边界|
|---|---|---|
|zurbrugg2025graspqp|3.2官方可微抓取指标来源|本项目仅适配，不宣称原创GraspQP|
|zurbrugg2026dexevolve|3.2进化优化原方法来源|本项目为DexEvolve-style，不冒充完整Isaac复现|
|mandlekar2022demonstrations|3.5人类示教与质量背景|不是本项目任务性能依据|
|mandlekar2023mimicgen|3.5自动构造示教的背景|不声称复现MimicGen|

上述元数据沿用reference.bib与第一章已审核文献目录；未扩展检索、未补造DOI。

|正文论点|内部代码/结果|等级与范围|
|---|---|---|
|3.1候选坐标、六驱动及合同|contracts.py::GraspCandidate、executor.py::candidate_world_pose|L2指定函数|
|3.2几何指标及约束|graspqp_adapter.py::refine_candidates_with_graspqp、hand_surrogate.py|L2已核验适配函数，不代表完整算法消融|
|3.2进化细化|dexevolve.py的变异、交叉、评价及档案；budget.py|L2当前实现，历史预算不回填|
|3.2格点与条件后续编辑|templates.py、batch_grasp_edit.py；Top100 summary|L2实现与清单统计，RL2条为历史状态|
|3.3坐标与两目标评分|executor.py::candidate_world_pose/rank_candidates_for_scene|L2函数；位置残差系数实际20|
|3.4插值与阶段|executor.py::_run_pose_segment/execute_grasp|L2函数；不保证全局无碰撞或无分支跳变|
|3.4严格与普通回放|strict_replay.py、profiles.py、grasp_pipeline/replay.py|L2函数；当前tail穿透判据与全程记录分开|
|3.5五任务逻辑|task_transfer.py/stack_transfer.py/push_transfer.py/nut_transfer.py|L2所审计函数/片段；前缀复用与重执行分开|
|3.6批次结果|outputs下dex_hand_top100_v2、pick_place_top100_final、stack_task_valid_v24、push_top100_final、nut_assembly_top100_final各summary.json|L2条目重计；内含成功是历史声明，不是本轮L3|
|3.6 Sugar十条Strict表|ch03_material_audit的historical_results_check.json::source_a_ten_rows及其逐条Strict原件|L2指定文件，表中数值来自对应final_lift和tail_max_angular_speed，显示保留5位小数|
|3.6九十次汇总失败|outputs/dp_p1_2026-10-11/generation_formal_summary.json|L1汇总，未逐条认证90源轨迹|
|3.6 Lift单例、Nut retain局部对照|five_task_p32_b_2026-11-03/lift_sugar_strict.json及Nut结果；docs/progress/2026-11-03_p32_b_online_recorder.md|指定结果L2及报告解释L1，不称跨seed稳定提升|

源函数行号、完整Hash与历史原件路径见审核通过的ch03_material_audit_v1_2026-11-20/source_index.json及CH03_EVIDENCE_MATRIX.md。没有借用14/127未知批次，无L3。

## 任务书与代码差异

任务书3.3评分示例的位置残差系数为2，审核材料和实际executor.py为20。正文式eq:ch03_reachability_score采用代码中的20，未修改任务书、实现或历史结果。原始评分是20*maximum_position_error+0.6*maximum_orientation_error+0.08*travel。

## 图表和公式

6图：整体流程、候选模块、可达筛选、轨迹与Strict、多任务关系为5张实现示意框图；第6张典型成功/失败案例为待补真实画面图位。没有伪造实验曲线或图像。拟用Lift及Nut对应历史视频/manifest；失败视频若无对应原件，不制作虚假截图。

6表：候选参数、Strict阈值、任务阶段、历史批次、Sugar10条、90次失败汇总。4个公式环境：候选合同、目标变换、可达评分、插值。无新增未实现的控制理论。
