# 实验口径及结论审计

仅现有Validation标量CSV；不运行C0、模型或重新拟合Scaler，不读Test/Holdout数组/预测。完整55行保存在validation_evidence_check.json。对现有BA、AUC、D作代数/门槛一致性核对，不重算AUC。

## 主协议与数据角色

原Real Train+Natural Train训练后冻结11C0；Logistic、kNN、MLP/GRU/Transformer各Seed11/12/13。Scaler、log1p、Real=1/Sim=0及0.5阈值固定。BA区间[0.45,0.55]与D≥0.90等价，AUC[0.475,0.525]单独要求。所有11实例均需联合满足。

|版本|负类/生成来源|联合或诊断门槛满足数|BA/D等价门槛数|通过实例|
|---|---|---:|---:|---|
|C3|905 Translated Natural vs780 Real|1/11|11/11|knn|
|C13|905 Translated Natural vs780 Real|1/11|11/11|knn|
|C27|905 Translated Natural vs780 Real|2/11|11/11|gru_12, knn|
|C23|780重建Real vs780原Real（仅读出诊断）|6/11|11/11|mlp_11, mlp_12, mlp_13, gru_13, logistic, knn|
|C24|780重建Real vs780原Real（仅读出诊断）|2/11|11/11|transformer_13, knn|

C3来源为C11/probe_metrics.csv严格group=F、role=validation，不是C3/final_evaluation的历史开发Test。C13来源是tactile_c13_frozen_c0_validation/metrics.csv，不是C13自身重建统计或C14新Probe。C27用原C3 Real/Sim权重，无C13 Decoder。

## 逐实例核对

|实例|C3 AUC/通过|C13 AUC/通过|C27 AUC/通过|
|---|---|---|---|
|logistic|0.682604 / 否|0.658306 / 否|0.673062 / 否|
|knn|0.500536 / 是|0.497752 / 是|0.509993 / 是|
|mlp_11|0.591571 / 否|0.571536 / 否|0.600164 / 否|
|mlp_12|0.598765 / 否|0.589597 / 否|0.630157 / 否|
|mlp_13|0.610894 / 否|0.601097 / 否|0.617238 / 否|
|gru_11|0.239559 / 否|0.187208 / 否|0.571915 / 否|
|gru_12|0.237257 / 否|0.191583 / 否|0.487958 / 是|
|gru_13|0.383622 / 否|0.308620 / 否|0.548114 / 否|
|transformer_11|0.282288 / 否|0.266441 / 否|0.357940 / 否|
|transformer_12|0.217189 / 否|0.236963 / 否|0.307241 / 否|
|transformer_13|0.698591 / 否|0.675462 / 否|0.643598 / 否|

BA与D完整值见JSON；5.6表的C3/C27六位小数与原CSV可对应。C3/C13只有kNN通过，C27新增GRU12，其余九个仍未通过。三者BA/D全通过不能覆盖AUC差异。未发现将2/11改为全成功，或只报最好Seed。

## 目标侧及局限的数值含义

C13同780 Real窗MSE0.07786883→0.02631960，未被误称Natural性能。C23读出6/11，C24读出2/11；它们不进入Natural通过数汇总表。C24时序数值为沿相邻保留帧log1p差分的绝对误差均值：0.00281497→0.00161364，非每秒物理导数；MSE0.02715745、汇总正值P99 3.54426→3.28971、四帧总量跨窗std 67.58367→57.52224，定义与report.py一致。

19指纹/97窗仅Paired跨划分记录字段复用，完整qvel等状态未包含；未推断初态完全同一。C25直接跨独立latent OT未执行，C26因结构保持停止C0未运行，不写0/11。C10-R1稳定性改善但三项未共同优于GRU，不能扩大成所有扩散无效；C5–C9仅有限配对/物理读出归因，不是新迁移成功证据。

最终结论必须保留：**C27原冻结Validation联合通过2/11，尚未完成完整跨域迁移验收。** 发展阶段Validation不是新盲测，没有真实任务收益或物理Taxel映射认证。

## 仍需增强的证据组织（非新实验要求）

数据与主结论匹配，但5.6未成组呈现三类域内重建、伪配对准入质量与配对latent回归的最小量化链。已有C4/frozen_reconstruction_metrics.csv、translator_paired_error.csv及5.3原准入记录可支持一个简表；引用时区分RMSE/MSE、全局IoU/逐窗IoU，不能无条件混比。此项是论文论证完整性建议，不否定当前已如实报告的失败结论。
