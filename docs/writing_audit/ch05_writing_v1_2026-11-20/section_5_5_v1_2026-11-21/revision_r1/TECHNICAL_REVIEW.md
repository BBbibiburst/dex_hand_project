# R1四项定点复核

本轮仅改四处文字及部署读出的行内尺度表达式；节结构、结构图、两组显示公式与其他段落保持原样。无训练、推理、仿真或指标重算。

|项|正文修订|直接依据|
|---|---|---|
|1. Mask三种角色|首帧BOS；后续真实前帧Mask输入循环状态；真实当前帧Mask输入幅值头；当前真实Mask同时作为预测logit BCE监督目标|C3/code/models.py:20–22,28|
|2. 检查点|第30至45轮范围、最终30轮、570次参数更新|C3/code/pipeline_cpu.py:71–77；translator/history.csv与已审元数据|
|3. 对照范围/数值|Paired Validation潜变量Huber，T=0.32591983675956726，模板=0.4942297637462616；全窗口×4帧×64维平均，保留Session例外|C4/translator_paired_error.csv的role=validation,group_type=ALL；code/models_audit.py:58–68|
|4. 幅值尺度|部署输出使用s_pos乘Gamma混合均值Σωa/b；沿用5.4的shape/rate和尺度定义|C3/code/models.py:22；5.4冻结源|

Mask头仍产生预测logit；真实当前Mask分别作为幅值条件与BCE目标，不是用真Mask代替预测logit。Decoder参数虽固定，BCE和幅值NLL对Translator输出的梯度保留，原文梯度说明未改变。

对照模板由Paired Train的Real posterior mean沿窗口维取均值，保留4×64结构，不使用验证目标拟合。此次只读取已有CSV，不重新算Huber或改善率；Paired误差不代表Natural迁移，原文最终限制仍保留。

Gamma正值条件均值=s_posΣωa/b；最终完整输出再由模型预测Mask控制，原段“Mask由模型预测”保留。未把s_pos写成真实力学单位或增加裁剪。

TeX静态检查通过，不编译、不冻结；PDF中行内精确小数的断行尚待后续授权编译检查。完整精度数值按本轮要求保留，没有凭空增加位数。
