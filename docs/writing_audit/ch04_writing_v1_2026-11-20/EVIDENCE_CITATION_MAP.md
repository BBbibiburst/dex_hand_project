# 第四章论点、证据与图表对应

本章以已审实现和指定数据核验为依据，没有新增或强行插入外部文献，不修改reference.bib。内部来源均可由材料审计目录source_index.json、dataset_checks.json及integrity.json追溯函数、字段与SHA256。正文不以代码文件数量作为贡献。

|小节|论点/对象|代码或数据依据|证据范围|
|---|---|---|---|
|4.1|state39、两种action13、tactile712及对象状态|source/imitation/observation.py；production_recorder.py；三LeRobot和11NPZ字段|L2所列合同与字段，不等同实机标定|
|4.2|执行前采集、实际动作与缓存|production_recorder.py::OnlineEnvironment/recorder_type|L2当前实现，不回填旧数据|
|4.2|RGB与控制降频对应|production_capture.py及production_video.py|L2实现，默认/速度配置不冒充服务器实测|
|4.3|pre-action索引与低层ctrl|collect_generated_lerobot.py::recorded_pre_action_indices/_export_one；P2固定340行|L2指定数值字段，非全部模态认证|
|4.3|P31移位与语义修复的区别|five_task_dataset.py::write_causal_source|L2代码；移位不自动转换IK动作|
|4.4|结构、时序、来源与隔离|production_recorder.py、production_artifacts.py、production_queue.py|L2机制，非全库实际运行保证|
|4.4/4.5|旧PickPlace373转移及旧Stack缺字段|dataset_checks.json::npz_cases及所列源NPZ|L2本轮材料复核；不推断唯一原因，不声称已修复|
|4.5|三LeRobot的5/1550、30/14310、1/340及非零率|dataset_checks.json::lerobot，全部所选数值Parquet|L2限定目录，RGB未解码，独立Root未知|
|4.5|五v4包2727行、5 lineage、4 asset、5 content|dataset_checks.json::npz_cases，五包episode.json及登记文件|L2逐项字段/文件Hash检查，不认证独立Root或正式验收|
|4.5|三个指定340行映射最大差0|dataset_checks.json::fixed_export_numeric_check|L2；只比较action、state命令区和指定触觉|
|4.5|Sink两路径误差1.628897/0|outputs/five_task_p38_exact_replay_2026-11-03/comparison.json|L1历史回放成绩，读取源辅助字段为L2，无本轮L3|
|4.5|Real/Natural冻结角色与规模|outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/data_contract.csv|L1历史合同声明，未读Real全库或Test/Holdout数组|

## 图表来源

图4-1数据载体、图4-2采样时序、图4-3多速率采集、图4-4质量准入均为根据实际代码组织的流程示意，不是实验截图。图4-5由comparison.json中两条已存误差绘制：旧路径1.6288973093、完整状态+float64路径0，图中保留六位显示。没有补造曲线，零值直接标注，不以极小正数替代。

plot_sink_comparison.py仅生成统计图；输入Hash和原字段保存在figure_provenance.json。图像资产为thesis/figures/ch04/sink_historical_precision.pdf及PNG。所有5张图均已提供，无待补图位；未运行物理仿真。

六表依次为字段合同、记录版本、质量门槛、LeRobot库存、旧/v4数据对照及Sink历史结果。任务标签是lift:物体，不将task_index计作多任务数量；三个库不能加成36个独立Root。

## 四个公式的实现对应

- eq:ch04_causal_pairing：在线pre-action或连续旧post-action行的对应，明确成立条件，不能无条件修复隐藏重置。
- eq:ch04_time_interval：0.05s控制/转移时间合同，表示浮点容差而非逐位时间相等。
- eq:ch04_mapping_error：最大绝对误差，对应审计脚本np.max(abs(...))。
- eq:ch04_duplicate：production_queue.register查询lineage相等或content相等，逻辑为OR，不是AND。

没有引入不存在的同步估计、语义对齐或无泄漏保证。
