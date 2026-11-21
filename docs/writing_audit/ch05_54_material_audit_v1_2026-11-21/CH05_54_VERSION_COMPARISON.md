# C3/C13/C23/C24版本对照

路径别名：C3=`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`；C13=`outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18/`；C23=`outputs/tactile_c23_real_reconstruction_c0_2026-11-18/`；C24=`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/`。

|版本|实际修改/用途|训练参数与合同|冻结部分|已有主要结果及范围|等级|
|---|---|---|---|---|---|
|C3|独立Real VAE和Sim VAE域内训练，后续固定两者训练Translator|每VAE1,094,232参数；Real Seed4301、Sim4302，各45轮最佳30|训练各自VAE时其全部参数更新；训练Translator时两VAE冻结|训练日志、选定权重、域内重建均存在；不证明latent天然对齐|L2|
|C13|修复Real幅值分支使用预测Mask条件的训练路径；结构不变|只训练decoder.current与decoder.amplitude，6个张量594,512参数；Seed4313，18轮最佳3；正值NLL|原Real Encoder、previous、GRUCell、Mask、BOS及尺度buffer；原Sim VAE/Translator不重训|同780个Real Val窗MSE从0.07786882699至0.02631960165，逐窗平均Mask IoU保持0.5117154775|L2|
|C23|读取C13已有Real Val输出做冻结C0与输出诊断|0步，无新训练；复用既有重建，不重新生成|全部模型/C0|Real自重建诊断，MSE0.02631960165；不是Sim迁移|L2（读历史评价，不是本轮重算）|
|C24|以C13初始化，仅增加相邻保留帧log1p幅值变化L1|同594,512参数，Seed4313；18轮固定预算最佳2；NLL+269.8845743604945×Ltemp|与C13相同；Sim VAE/Translator未改|Real Val时序误差降低但MSE约0.027157449，幅值尾部/多样性收缩，不能称整体成功|L2（代码、权重、历史诊断）|

## 1. 优化步和所存权重不能混淆

|模型|总训练Epoch/Step|选定Epoch/Checkpoint内Step|选择规则|
|---|---|---|---|
|C3 Real|45 / 3645|30 / 2430|epoch≥30的Validation NLL+0.001KL最低|
|C3 Sim|45 / 5715|30 / 3810|epoch≥30，Natural/Paired Validation目标各半|
|C13 Real幅值|18 / 1458|3 / 243|Validation正值NLL，patience15|
|C24 Real幅值|18 / 1458|2 / 162|固定18轮；Validation正值NLL最低，非含时序总损失最低|

C13/C24新建AdamW优化器，不能写成恢复C3原优化器续训。C24另存训练终点权重，不能将终点与best混用。

## 2. 张量级版本核对

checkpoint_metadata.json记录7个权重文件的来源Hash、键、形状和标量；verified_facts.json记录张量一致性比较。C3 Real→C13只6个current/amplitude张量不同；C13→C24初始权重完全相同；C13→C24选定权重只同6个张量变化。未调用模型推理。

C13的缓存包含固定posterior mean及自由Mask循环状态；这不是重训Encoder，也不代表整个Dual-VAE被联合更新。C24仅修同一路径，不能把C24幅值、C13 Decoder和其他版本的结果拼成新的完整模型。后续完整链路若替换Real Decoder，必须另行声明实际使用权重；本轮不构造此链路。

## 3. 历史指标的可比性

C4/code/models_audit.py的Mask IoU为全局intersection/union；C13/code/run.py:59–60为先逐窗IoU再平均。因此C4原Real Val 0.5151513900与C13原基线0.5117154775不是Mask变动证据。C4同时记录RMSE，C13为MSE，不能直接相减作改善率。

上表C13前后MSE取同一metrics.csv、同一780窗及同一实现，可直接对照；C23复用相同输出故不能算独立重复验证。C24相邻帧统计是保留序列顺序上的变化，采样间隔不应默认等间隔20Hz，也不是物理时间导数。

## 4. 论文位置

5.4：独立VAE结构/概率模型/损失/训练与自由运行差异，并一句说明后续幅值修订仅限目标域。

5.6：详细重建误差、时序与幅值退化、C23/C24逐实例C0对照。C0联合结果必须沿原冻结协议；不由本轮重算。C27完整链路评价不得回填为C3域内训练结果。

## 5. 已有域内重建的描述性参照（不新增评价）

下表逐行摘自C4/frozen_reconstruction_metrics.csv。RMSE与全局汇总Mask IoU不等同于C13的MSE与逐窗IoU。它们支持已形成域内读出，但不证明公共语义或跨域成功。

|数据|窗口|RMSE|全局Mask IoU|
|---|---:|---:|---:|
|Real_train|5153|0.11705617148736186|0.7059695607336159|
|Natural_train|4049|0.057424343497607555|0.6214225549881085|
|Paired_train|1200|0.0013715728545512303|0.650826550730499|
|Real_validation|780|0.2790498692835028|0.5151513900301005|
|Natural_validation|905|0.05562773455753662|0.39928341918702503|
|Paired_validation|240|0.001617713039414183|0.5775013545241106|

该CSV的Natural Validation NLL均值约160.944、而中位数约−14.545，提示汇总值受尾部影响；不能仅凭均值推断每个窗口都失败。训练与验证差距是泛化风险证据，不是容量不足或尺度异常的单因素证明。C0结果仍留给5.6，本轮不读取预测文件、不重新评价。
