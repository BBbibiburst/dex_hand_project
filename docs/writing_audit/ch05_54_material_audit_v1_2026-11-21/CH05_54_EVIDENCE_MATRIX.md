# 5.4论点—原始证据矩阵

本报告路径相对于项目根。L0=定位；L1=历史报告/合同声明；L2=直接核对已有代码、日志、CSV、Checkpoint；L3=新执行验证。本轮无L3。引用历史审计的群组统计不升级成重新读取原始数据的L2复算。

## 来源代号

- C3：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`
- C4：`outputs/tactile_p43_c4_root_cause_audit_2026-11-16/`
- C13：`outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18/`
- C23：`outputs/tactile_c23_real_reconstruction_c0_2026-11-18/`
- C24：`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/`
- B：`outputs/tactile_p43_latent_translator_integrated_2026-11-16/`
- A：`docs/writing_audit/ch05_material_audit_v1_2026-11-20/`

|ID|可写论点|代码位置/直接字段|结果或配置来源|证据与边界|建议位置|
|---|---|---|---|---|---|
|V01|两个VAE独立建立与优化|C3/code/pipeline_cpu.py:27–28,53–56|real_vae及sim_vae/checkpoint.pt独立state_dict|L2；类相同不等于共享参数|5.4开篇|
|V02|每帧1424→128→GRU128→mean/logvar64|C3/code/models.py:7–12|checkpoint_metadata.json各权重shape|L2；每窗4×64，不是每帧256|Encoder|
|V03|logvar[-8,4]与随机重参数化|models.py:12,32–33|pipeline_cpu.py:21–26,39的sample参数|L2；eval()本身不令sample=False|Encoder/训练差异|
|V04|时序AR Mask|models.py:15,18–25|previous/cell/mask/BOS张量|L2；帧内条件Bernoulli仍因子化|Decoder|
|V05|当前Mask条件的两分量Gamma|models.py:16–17,22,28|amplitude末层4272维|L2；分量不是物理接触类别|Decoder|
|V06|自由读出为hard Mask×混合均值|models.py:22–26；inference.py:15–19|原推理入口静态读取|L2；未运行推理；没有Gamma随机采样|Decoder|
|V07|BCE按4×712均值，正值NLL按每窗正值数|models.py:27–28|C3损失定义|L2；全零窗幅值项0，非正值替换只为避免log0|公式|
|V08|NLL有1/positive_scale Jacobian|models.py:28，logsumexp后减scale.log()|C4/scale_formula_audit.json|L2；负连续NLL不必异常|公式/尺度|
|V09|KL按4×64均值，系数0.001|models.py:33；pipeline_cpu.py:39|model_training_registration.json beta_KL|L2；不能遗漏均值后称同一权重|公式|
|V10|各域Train分别拟合两个单标量尺度|pipeline_cpu.py:19–20,28|两域complete.json与Checkpoint buffer|L2；不减均值，不是C0 Scaler|数值合同|
|V11|Sim positive_scale=1e−7触底|scale_fit；C4/amplitude_scale_audit.csv|C4/scale_formula_audit.json|L2核对既有数值表；未读原始数组重算中位数|尺度/限制|
|V12|Real/Sim各45轮、best30、Seed4301/4302|pipeline_cpu.py:27–53|两域complete.json/history.csv、Checkpoint epoch与optimizer_steps|L2；总Step3645/5715与所存2430/3810不同|训练表|
|V13|Sim每batch32Natural+32Paired|pipeline_cpu.py:31–39|history.csv source/paired visits；coverage_exact.npz|L2；重复访问不算新样本或Root|训练表|
|V14|最早可选epoch30|pipeline_cpu.py:45–50|history selection_loss，eligible与global最小轮次|L2；早期更低损失不授权换权重|方法局限|
|V15|原始数据读取角色与路径|C3/code/common.py:24–27；pipeline_cpu.py:16|B/real_train与real_validation/index.json；A/verified_observations.json|路径/合同L2；独立群组数量沿既有合同L1，不称本轮原数组复核|训练数据|
|V16|C13仅current/amplitude更新|C13/code/run.py:23–25,36–57|parameter_change_audit.json、checkpoint.pt与C3逐张量比对|L2；仅Real幅值修订|版本说明|
|V17|C13重建MSE降低、Mask不变|C13/code/run.py:59–65|metrics.csv原始/修复同780窗行|L2既有统计，不是新推理；IoU为逐窗平均|5.6|
|V18|C23零训练复用输出|C23/code/evaluate.py:15–86|data_contract.json/reconstruction.json/registration.json|L2；原始评估存在，本轮不重算C0|5.6|
|V19|C24增加单一时序幅值项、参数不扩大|C24/code/train.py:16–18,32–53|registration.json、parameter_change_audit.json；初始/最终Checkpoint比对|L2；λ由Train初始尺度，best仍按Val正值NLL|版本与5.6|
|V20|C24尾部/多样性问题|C24/Real_C13_C24_structure.csv及decision.json|reconstruction.json及历史Session诊断|CSV数值L2、解释L1；不是单因素根因证明|5.6|
|V21|IoU统计口径不同|C4/code/models_audit.py:16–19；C13/code/run.py:59–60|frozen_reconstruction_metrics.csv与metrics.csv|L2；全局IoU不与逐窗IoU直接比增益|5.6表注|
|V22|latent非天然共同语义、监督映射仍可学习|C3独立训练；models.py:34–36；pipeline_cpu.py:54–70|未使用共享对齐目标，后续Paired Huber+RealDecoder NLL|代码L2；坐标不确定性为数学解释，非新潜空间实验|5.4结尾/5.5承接|

## 既有权重与图表的可追溯性

全部权重SHA256、字段和Tensor形状见checkpoint_metadata.json；全部读取来源Hash见audited_inputs.json。C3/converter_frozen.json中的三模型登记Hash与所存文件核对记录在verified_facts.json，不据此重建新转换器。

三幅建议结构图均由models.py及训练调用链直接归纳，详见CH05_54_MODEL_ARCHITECTURE.md的Mermaid。它们不是新实验可视化，不包含拟造热图、曲线或推断的物理Taxel位置。训练参数、尺度及版本表分别来自配套损失/版本报告；完整性能表留到5.6。

群组数、尺度统计和已有Validation指标的原始计算程序保留上述路径。本轮只复核其既有表与实现，没有以读报告替代重新运行验证，也不将L1汇总声明升级为本轮L3。
