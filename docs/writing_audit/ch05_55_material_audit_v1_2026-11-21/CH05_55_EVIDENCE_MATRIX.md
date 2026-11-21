# 论点—实现—数据—证据矩阵

路径别名同总体报告：C3=tactile_p43_c3_dictionary_inverse_c0_2026-11-16；C4=tactile_p43_c4_root_cause_audit_2026-11-16，均位于outputs/。原始绝对/相对来源与SHA256见source_hashes.json。

|论点|代码/函数/行号|配置/产物|等级与边界|
|---|---|---|---|
|两层GRU+投影，181824参数|C3/code/models.py:34–36 Translator|translator/checkpoint.pt张量形状|L2，不运行网络|
|输入输出均值语义|pipeline_cpu.py:57–59 means；models.py:31|zs/zr与simtr/ptr同索引|L2，非采样latent|
|仅更新T，VAE冻结|pipeline_cpu.py:54–63|AdamW(T.parameters)|L2，无新梯度数值实验|
|Huber delta1、全部元素mean|pipeline_cpu.py:63 F.huber_loss默认参数|既有PyTorch接口与C4显式none归约交叉核对|L2代码；无额外维度尺度|
|Decoder NLL=BCE+正值Gamma NLL|models.py:27–28|同VAE模型定义|L2，非KL或单独幅值项|
|真实Mask Teacher Forcing|models.py:18–28 run/nll|target传入、prev=current|L2，硬Mask读出不作求导路径|
|冻结但对T可反传|pipeline_cpu.py:63 loss.backward；models.py参数路径|未包no_grad、未detach预测z|L2计算图，不冒称L3|
|Seed/批次/更新步与选择|pipeline_cpu.py:59–77|registration、history、complete、checkpoint epoch/steps|L2，855总步不写成选定模型步数|
|权重已冻结且一致|pipeline_cpu.py:78|converter_frozen.json、所存Hash|L2，仅读登记，不执行后续Test代码|
|推理全链|code/inference.py:7–19 Converter.transform|输入检查、posterior_mean、deterministic_decode|L2，不使用Sim Decoder输出冒充Real|
|1200/240数组与manifest同序|code/verify_pairs.py:6–21；expand.py:85–90|paired_train/validation.npz与paired_manifest.csv|L2本轮逐行Hash核对|
|分组与状态复用|C4/code/pairs.py:6–13,27–38|paired_independence_counts.json、paired_cross_split.json|manifest交叉L2；reset/近构型既有L1读取|
|Top-5与最优候选压缩|C3/code/expand.py:37–53,66–90|冻结5.3正文及生成记录|L2；Top-5分量不等于5个完整候选|
|均值参考为PairedTrain Real均值模板|C4/code/models_audit.py:51–71|translator_paired_error.csv|L2，按窗口再时间/维度归约；非全RealTrain均值|
|30→45训练与验证分化|translator/history.csv|训练报告原始行|L2日志，不重算预测|
|近输入与异目标风险|C4/code/input_ambiguity.py|paired_mapping_ambiguity_summary.csv|代码/历史CSV L2，非因果证明|
|后续版本隔离|C13/C24 train/run、C25/C26 decision、C27 common/train|各自config/registration|L2代码+L1范围说明，不拼统一模型|
|GRU相对MLP优势|指定材料未提供公平消融|无同协议对照|UNKNOWN，不补造|

训练日志所有行保留在training_evidence.json；总体、Session反例及历史歧义表在mapping_evidence.json。检查点形状与参数量在checkpoint_metadata.json。未读取Test/Holdout样本或预测，未写正式LaTeX。
