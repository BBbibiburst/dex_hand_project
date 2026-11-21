# 训练合同、检查点与已有映射证据

C3根：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`；C4根：`outputs/tactile_p43_c4_root_cause_audit_2026-11-16/`。以下为L2已有日志/代码/张量元数据核验，不运行模型。

## 1. 实际合同

|项目|实际值|来源|
|---|---|---|
|Seed|4303|code/pipeline_cpu.py:59,61|
|优化器|AdamW，仅T.parameters|同:59|
|学习率/weight decay|0.0003 / 0.0001|同:59|
|Batch|64，末批48|同:62–63，1200窗|
|每Epoch|19步，全部1200配对各访问1次|同:61–63|
|训练Epoch/总Step|45 / 855|translator/complete.json与history.csv|
|访问总量|54000次，不是54000独立配对|1200×45|
|保存条件|epoch≥30，Validation Huber+0.1 NLL改善>1e−5|pipeline_cpu.py:71–76|
|上限/停止|max100，patience15；实际best30后15轮未改善，于45停止|同上|
|选定权重|epoch30，Checkpoint optimizer_steps=570|translator/checkpoint.pt|
|梯度裁剪|T参数范数上限5|pipeline_cpu.py:63|
|训练设备|CPU、2线程|pipeline_cpu.py:11|
|输入|Paired Train1200，Paired Validation240；各为两Encoder的posterior mean|同:57–59|

双域VAE事先冻结。Natural参与Sim VAE训练，但**不直接参与原C3 Translator梯度训练或模型选择**；Translator仅随机打乱1200个配对窗口，每轮permutation(1200)，不按Session均衡采样。验证按样本数加权归并各batch，避免末48窗与64窗等权造成偏差（:65–70）。

Checkpoint只有model/epoch/optimizer_steps，没有优化器状态；不能声称可以从该文件无缝恢复原AdamW状态。Checkpoint Hash：

`c98e1146ae2f8d17ce05ecffacbfeaf24e1cf07db4d51783c69355fcacfb7b6a`

该Hash与converter_frozen.json登记一致。最小Validation总目标在全45轮中也出现在30轮；与两VAE不同，不能笼统将其描述成“更早更优权重被最低30轮规则排除”。仍须说明存在资格限制。

## 2. 曲线实数（history.csv，不重新评价）

|Epoch|Train Huber|Validation Huber|Train Decoder NLL|Validation Decoder NLL|Val总目标|
|---|---:|---:|---:|---:|---:|
|1|0.7537227121988932|0.7098021944363911|-2.035900368690491|0.1569993793964386|0.725502132376035|
|30|0.2556511922677358|0.3259198307991028|-4.245328947703044|-1.7468248367309571|0.1512373471260071|
|45|0.22208811382452648|0.3491156041622162|-4.338185469309489|-1.7871159474054972|0.17040400942166647|

30→45轮Train Huber继续降低，Validation Huber从约0.32592升至0.34912；Val NLL稍更负，却不足抵消潜变量误差上升，总目标从0.15124升至0.17040。这说明两目标趋势并非完全一致，不能只选更好看的NLL或Train曲线。历史报告与C4汇总末位差异来自batch汇总/逐窗汇总精度，不能当新的实验改善。

## 3. 均值参照到底是什么

C4/code/models_audit.py:51–71：先对**Paired Train所对应的1200个Real目标**提取posterior mean，沿窗口维求mean(0)，得到4×64常量模板；不是对全部5153 Real Train求均值，也不是每个时间维全局同一个64向量。将该模板广播到Train与Validation各样本，分别与各自真实Real posterior mean比较。

Huber delta1、逐元素reduction=none后按时间4和维64求均值，再按窗口或Session平均；ALL指标等价全体N×4×64平均。Translator和均值参考使用同一批目标、同一Real Encoder坐标，均未额外标准化，属于同一评价空间内可比较参照。

|集合|Translator Huber|Train均值模板Huber|相对降低|RealDecoder NLL|真实Real latent读出NLL|
|---|---:|---:|---:|---:|---:|
|train|0.25565120577812195|0.6218109726905823|0.5888602435588837|-4.245328903198242|-4.675130367279053|
|validation|0.32591983675956726|0.4942297637462616|0.34054994583129883|-1.746824860572815|-2.109210729598999|

“相对降低”为1−模型Huber/参考Huber，不是触觉域混淆度或百分比准确率。真实Real latent读出也使用Teacher Forcing，仅作为已知目标编码的条件读出参考，不是可用于Natural部署的oracle方法。

总体回归误差相对常量参考下降，支持在指定伪配对上学到非平凡关系。但3个Train Session与1个Validation Session劣于均值参考，不能声称各组全面改善。Validation反例为20260816_172352_slip_3D_bottle_C_30，34窗，Huber约0.33742，均值参考约0.32563；完整行见mapping_evidence.json。

这些误差不提供Natural配对真值，不证明自由输出712D结构或C0联合验收通过。详细阶段/Session与输出质量分析留5.6。
