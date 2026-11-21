# 论点—证据对应

C3根：outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/；C4根：outputs/tactile_p43_c4_root_cause_audit_2026-11-16/。本轮按已审核专项审计写作，读取原代码复核，不运行模型。

|段落/公式|核心论点|直接来源|边界|
|---|---|---|---|
|第1段、映射公式|固定均值缓存、同索引监督、输出Real坐标|C3/code/pipeline_cpu.py:54–63；models.py:31|非同步物理真值|
|第2段|两层GRU64→128、Linear128→64、181824参数|models.py:34–36；专项checkpoint_metadata.json|无公平MLP优越性结论|
|第3段、损失公式|Huber delta1、全B×4×64均值，加0.1×Decoder BCE+正值NLL|pipeline_cpu.py:63；models.py:27–28|没有新KL或对齐损失|
|第4段与流程图|冻结参数、可微输入梯度、Teacher Forcing|pipeline_cpu.py:54–63与models.py:18–28|不对硬阈值求导，不称自由生成成绩|
|第5段|1200/240、Seed4303、45×19=855、best30×19=570、优化器与验证汇总|translator/complete.json/history.csv；pipeline_cpu.py:59–77|不以C0选模型，未重跑|
|第5段部署|Natural→Sim均值→T→Real Decoder|code/inference.py:14–19|不用Sim Decoder输出替代|
|第6段|信息压缩、独立性、均值参考和Session反例|expand.py:37–53,66–90；C4/code/models_audit.py:51–71；translator_paired_error.csv|总体改善非全部Session改善|

沿用冻结5.3的统计伪配对/Top-5定义、冻结5.4的Encoder均值与Decoder条件似然定义。本节未新增外部理论论断或文献条目，没有新增cite，也未修改reference.bib。原始模型及数据Hash沿用专项source_hashes.json，本轮保护检查另见integrity.json。
