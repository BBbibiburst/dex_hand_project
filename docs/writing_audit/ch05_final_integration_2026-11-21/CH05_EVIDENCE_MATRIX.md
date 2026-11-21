# 第五章逐节证据矩阵

证据：L1历史报告/审计声明；L2静态代码、合同、既有CSV/metadata及来源Hash核验；本轮无L3。既有数字复核不等于重跑模型。C3根目录为outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16，简称C3。

|论点|章节/行|直接来源|等级/结果|限制|
|---|---|---|---|---|
|原生touch是非负法向力汇总；processed无共同力学单位|5.1:5|mujoco2026xml；官方XML reference sensor/touch；既有E-P40|L2文档与项目实现审计，匹配|感知体积与同body/射线条件；不证明真实标定|
|P37旧5153/1408、1824/406与P43当前5153/780、4049/905分开|5.1:7；5.2:18–23|5.1数字审计、5.2数据角色审计；C3 c0/group_coverage.csv|L2合同/既有统计|不是当前重新分析P43差异|
|17.60/13.31%、正值P50/P95、变化率1.7154/0.0958%|5.1:9–11、30|5.1 R1 NUMERIC_AUDIT.md、numeric_check.json及来源Hash|既有L2静态数组核验，本轮只读结论|Train正值池；Val仅帧0→1，不是导数|
|三个表示D最小0.0272/0.1494/0.0573|5.1:13–28|P39 probe_metrics.csv的既有审计；5.1 R1 NUMERIC_AUDIT|既有L2，匹配|旧转换后幅值校准诊断，后训Probe非冻结C0；Amplitude有效4×5|
|Natural完整processed→Real风格；三类数据及三层分析|5.2四段|C3/code/inference.py、models.py；5.2专项六份报告|L2，匹配|目标不是结果；三层不是均已验收|
|GMM二维中心化索引偏移、逆构造未中心化|5.3:21–40、82|冻结5.3及其R1 COORDINATE_AUDIT/TECHNICAL_AUDIT；C3/code/expand.py|已有L2，匹配|责任度/锚点索引启发式，不是物理概率|
|Top-5分量、最多3次候选、提前停止后仅比已执行者|5.3:60、76–84|C3/code/expand.py:37–82；既有GMM R1/R2复核|L2，匹配|可拒收已有通过候选；非全局最优；失败Episode/评分保存|
|480行、160指纹、1200/240配对|5.3:50–55、84|字典manifest及既有GMM审计；paired_manifest与pairing_recount|L2，匹配|不同统计范围；窗口/指纹不等于独立Root|
|独立VAE、4×64、AR Mask与条件Gamma|5.4:5–55|C3/code/models.py:7–33；5.4专项架构与损失审计|L2，匹配|域内表示不天然对齐；TF与自由推理分开|
|尺度、混采、45轮/选30轮/优化步|5.4:57|C3/{real_vae,sim_vae}/complete.json；pipeline_cpu.py|L2，本轮读取metadata匹配|尺度无物理单位；重复访问不新增样本|
|GRU映射181824参数、Huber+0.1 NLL、冻结Decoder可反传|5.5:13–31|C3/code/models.py、pipeline_cpu.py:54–77；5.5专项审计|L2，匹配|NLL用真实Mask条件；不是分布对齐损失|
|45×19=855、30×19=570；Huber及均值参照|5.5:45–47|translator/complete/history；mapping_evidence.json|L2，本轮读取既有标量|Paired Validation，不是Natural成功；有Session例外|
|C3/C13/C27联合1/1/2，BA/D门槛均11/11|5.6:20–59、63|C11 group F Validation、C13冻结Validation、C27 metrics.csv|本轮L2只读55行五版本CSV核验；22个C3/C27实例表四舍五入核对|未运行C0；AUC不反转|
|C23 6/11、C24 2/11为自重建诊断|5.6:61|各自metrics.csv及C24 Real_C13_C24_structure.csv|本轮L2既有记录匹配|负类为重建Real，不是Natural|
|19种放置指纹/97验证窗复用|5.6:65|5.5 pairing_recount、5.2 manifest核验；C4 pairs.py定义|已有L2，匹配|未含qvel等全状态；不等于完整物理初态相同|
|C25未OT、C26未C0；完整验收未完成|5.6:65|既有第五章协议审计、C25/C26 decision及停止记录|已有L1/L2，匹配|未执行不记0/11；不称任务收益|

全部CSV路径/字段/精确值见validation_numeric_recheck.json，关键量分项见NUMERIC_CROSSCHECK.md。无Test/Holdout信号或预测读取。各节源Hash及接入映射见SOURCE_MANIFEST.json。
