# 5.6论点与证据对应

本轮仅读取现有Validation标量CSV、审计和合同，不运行分类器或模型。validation_metrics.csv保存55个实例-版本行，validation_records.json保留完整精度、混淆矩阵及原始来源；TeX表只展示六位小数，判定使用原始精度。

|正文|论点|直接证据与范围|
|---|---|---|
|协议段与图|迁移前11C0、Scaler/log1p/阈值、905 Sim与780 Real|CH05_C0_PROTOCOL_AND_METRIC_AUDIT.md；既有冻结合同；不把Test分母765代入|
|指标段|BA、D、AUC门槛与统计单位|同报告及既有common.metrics实现审计；BA/D等价，AUC独立要求|
|C3分析、两表|Validation联合1/11、D11/11，MLP11全部905 Sim判Real|outputs/tactile_p43_c11_latent_support_audit_2026-11-17/probe_metrics.csv，严格role=validation、group=F；不是C3/final_evaluation历史Test|
|C13分析|跨域仍1/11、MSE局部改善|outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv；tactile_p43_c13_amplitude_context_repair_2026-11-18/metrics.csv原始与repaired validation行|
|C23|6/11仅Real读出诊断，逐实例角色|outputs/tactile_c23_real_reconstruction_c0_2026-11-18/metrics.csv；负类是重建Real而非Natural|
|C24|2/11仅读出；帧增量误差下降、尾部/多样性收缩|outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/metrics.csv与Real_C13_C24_structure.csv，780同源Real窗|
|C27|联合2/11、D11/11；只有kNN与GRU12通过|outputs/tactile_c27_c3_route_improvement_2026-11-18/metrics.csv；config.json明确原C3双VAE与新增损失；不混C13|
|结尾|共享状态、监督歧义、物理读出与替代路线限制|C4/paired_cross_split.json，已审核CH05_MATERIAL_AUDIT/OPEN_ISSUES/C0_PROTOCOL_AND_METRIC_AUDIT及5.5专项；只归纳现有受限结论，不新审计/实验|

数据角色：前三组跨域905 Natural→输出vs780 Real；C23/C24是780重建Real vs780原Real，不进同一迁移汇总表。C25未执行OT、C26未运行C0没有登记为0/11。历史报告的Test汇总只用于辨认版本，不读取其原始数组或预测。

没有新增文献需求；reference.bib不变。本节以实际项目结果及冻结合同为证据，不添加无出处外部引用。逐实例分类门槛核对只对已有BA/AUC/D标量作一致性检查，不重算预测/AUC、不改阈值。
