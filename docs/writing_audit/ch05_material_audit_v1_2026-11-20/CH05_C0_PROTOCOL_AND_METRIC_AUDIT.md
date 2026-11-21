# 冻结C0协议与指标复核

## 1. PRIMARY与SECONDARY

PRIMARY是C3在**原Real Train5153窗/32Session及Natural Train4049窗/50Root**上训练后冻结的11个C0：Logistic、kNN；MLP/GRU/Transformer各Seed11/12/13。冻结前数据共9202训练窗；当前开发Validation为780Real/4Session、905Natural/10Root。所有C0用同一pooled Train的log1p均值/标准差（std下限0.05），坐标为4×712，不重排通道，不按域重拟合。

C3/code/c0.py第15–16行给出标签与Scaler，c0/training_registration.json登记架构、epoch与Hash；c0/frozen.json列11实例和权重、kNN参照及ScalerHash。本轮逐文件Hash核对通过。常规深度预算20–100Epoch、patience15、AdamW lr1e-3/wd1e-3；原CPU续训路径记录保留，不能假定所有模型都在同一设备完成。

Real=1、Sim=0，概率p为Real分数，判定p≥0.5。common.metrics来自P37/code/common.py第23–27行：BA=(RecallReal+RecallSim)/2；AUC用正类Real的平均秩统计，平分数使用average rank；D=1−2|BA−0.5|。原门槛BA∈[.45,.55]、AUC∈[.475,.525]、D≥.9。BA门槛与D门槛数学等价，不是两份独立证据。AUC<.5不能改为1−AUC参加正式验收。

点估计为窗口级的类平衡BA及窗口分数AUC，并非Session/Root等权宏指标；训练时分组平衡Loss不能代替评价端的组平均。Bootstrap才按既定Session/Root分组重采样。

混淆矩阵行为实际0/1、列为预测0/1：[[Sim判Sim,Sim判Real],[Real判Sim,Real判Real]]。C23/C24中负类实际是重建Real，名称是读出诊断标签，不能将其重新命名为Natural。

SECONDARY是在Real与迁移Sim上重新拟合的Probe。P37/P39的表示分解与C14的freshly_trained结果属于这种问题，不能替代PRIMARY。特别核对C14/probe_metrics.csv，frozen_C0为False；C13原冻结结果另在`outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv`，已与C23历史对照逐值交叉核对。

## 2. 结果角色与可比性

|结果|负类来源/分母|联合通过|可支持结论|
|---|---|---:|---|
|C0原始Validation|905 Natural原始 vs780 Real|0/11|原域分类能力强，不是转换成绩|
|C3原GRU在C11-F复现|同905 Natural迁移 vs780 Real|1/11|初始转换未完成联合验收|
|C13修幅值后Natural|同905 Natural迁移 vs780 Real|1/11|此幅值修复没有带来全联合通过|
|C23|780重建Real vs同780原Real|6/11|目标侧读出能力改善但仍不完整；非迁移6/11|
|C24|同780重建Real vs同780原Real|2/11|目标侧最小时序修复不足，不能当Natural下降|
|C27|同905 Natural迁移 vs780 Real|2/11；D11/11|D专项达标，原联合未达标|

C3原final_evaluation及报告中的“1/11、765个迁移窗口”等属于历史开发Test角色（Real628/3Session、Natural765/10Root），不是上表Validation。本轮不读这些Test数组、预测文件或重新计算其成绩。只从历史报告辨认其角色和已公开结论；不能因通过数巧合相同而混为同一试验。

C11-F是原C3冻结链路的Validation复现；本轮仅核验已有标量CSV，不新运行模型。C27仍用原C3 Real/Sim VAE，未换成C13修复版。C27验收包三个模型与源权重Hash匹配，scope文件明确AUC不纳入专项、原联合2/11、非新盲测。

## 3. 全部实例实际Validation结果

以下由原CSV读取并重算D及联合判定；BA同时用保存混淆矩阵复核，未重新拟合概率或AUC。每条原路径/Selector/Seed/分母见claim_cards.csv；完整混淆矩阵见validation_metric_recheck.csv。

### C13_Natural_Validation

来源：`outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv`。筛选Validation。

|实例|BA|AUC|D|联合通过|
|---|---:|---:|---:|---|
|mlp_11|0.497436|0.571536|0.994872|否|
|mlp_12|0.498718|0.589597|0.997436|否|
|mlp_13|0.498077|0.601097|0.996154|否|
|gru_11|0.498718|0.187208|0.997436|否|
|gru_12|0.498718|0.191583|0.997436|否|
|gru_13|0.498718|0.308620|0.997436|否|
|transformer_11|0.497436|0.266441|0.994872|否|
|transformer_12|0.496154|0.236963|0.992308|否|
|transformer_13|0.498077|0.675462|0.996154|否|
|logistic|0.498718|0.658306|0.997436|否|
|knn|0.498629|0.497752|0.997259|是|

### C0_Original_Validation

来源：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/c0/metrics.csv`。筛选Validation。

|实例|BA|AUC|D|联合通过|
|---|---:|---:|---:|---|
|mlp_11|0.986939|0.989295|0.026123|否|
|mlp_12|0.988221|0.988649|0.023559|否|
|mlp_13|0.987027|0.988466|0.025946|否|
|gru_11|0.985458|0.995523|0.029083|否|
|gru_12|0.985458|0.990963|0.029083|否|
|gru_13|0.986011|0.992613|0.027978|否|
|transformer_11|0.986939|0.996029|0.026123|否|
|transformer_12|0.986762|0.986218|0.026477|否|
|transformer_13|0.988685|0.981678|0.022631|否|
|logistic|0.982696|0.994087|0.034608|否|
|knn|0.994210|0.998078|0.011581|否|

### C3_Natural_Validation_C11F

来源：`outputs/tactile_p43_c11_latent_support_audit_2026-11-17/probe_metrics.csv`。筛选Validation且group=F。

|实例|BA|AUC|D|联合通过|
|---|---:|---:|---:|---|
|mlp_11|0.497436|0.591571|0.994872|否|
|mlp_12|0.498718|0.598765|0.997436|否|
|mlp_13|0.498077|0.610894|0.996154|否|
|gru_11|0.498718|0.239559|0.997436|否|
|gru_12|0.498718|0.237257|0.997436|否|
|gru_13|0.498718|0.383622|0.997436|否|
|transformer_11|0.497436|0.282288|0.994872|否|
|transformer_12|0.496154|0.217189|0.992308|否|
|transformer_13|0.498077|0.698591|0.996154|否|
|logistic|0.498718|0.682604|0.997436|否|
|knn|0.501944|0.500536|0.996111|是|

### C23_Real_reconstruction

来源：`outputs/tactile_c23_real_reconstruction_c0_2026-11-18/metrics.csv`。筛选Validation。

|实例|BA|AUC|D|联合通过|
|---|---:|---:|---:|---|
|mlp_11|0.498077|0.508804|0.996154|是|
|mlp_12|0.498718|0.493263|0.997436|是|
|mlp_13|0.498718|0.495975|0.997436|是|
|gru_11|0.498718|0.467295|0.997436|否|
|gru_12|0.500641|0.446688|0.998718|否|
|gru_13|0.500000|0.482736|1.000000|是|
|transformer_11|0.497436|0.466189|0.994872|否|
|transformer_12|0.497436|0.433461|0.994872|否|
|transformer_13|0.501282|0.344831|0.997436|否|
|logistic|0.499359|0.491561|0.998718|是|
|knn|0.499359|0.494876|0.998718|是|

### C24_Real_reconstruction

来源：`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/metrics.csv`。筛选Validation。

|实例|BA|AUC|D|联合通过|
|---|---:|---:|---:|---|
|mlp_11|0.500641|0.541169|0.998718|否|
|mlp_12|0.500000|0.536326|1.000000|否|
|mlp_13|0.500641|0.535856|0.998718|否|
|gru_11|0.500000|0.322286|1.000000|否|
|gru_12|0.507051|0.334713|0.985897|否|
|gru_13|0.506410|0.339905|0.987179|否|
|transformer_11|0.498077|0.370702|0.996154|否|
|transformer_12|0.500641|0.330611|0.998718|否|
|transformer_13|0.503205|0.509030|0.993590|是|
|logistic|0.501923|0.558041|0.996154|否|
|knn|0.503205|0.501338|0.993590|是|

### C27_Natural_Validation

来源：`outputs/tactile_c27_c3_route_improvement_2026-11-18/metrics.csv`。筛选Validation。

|实例|BA|AUC|D|联合通过|
|---|---:|---:|---:|---|
|mlp_11|0.498541|0.600164|0.997082|否|
|mlp_12|0.498718|0.630157|0.997436|否|
|mlp_13|0.499734|0.617238|0.999469|否|
|gru_11|0.498718|0.571915|0.997436|否|
|gru_12|0.511978|0.487958|0.976045|是|
|gru_13|0.511425|0.548114|0.977150|否|
|transformer_11|0.497988|0.357940|0.995977|否|
|transformer_12|0.499469|0.307241|0.998938|否|
|transformer_13|0.503602|0.643598|0.992796|否|
|logistic|0.501480|0.673062|0.997039|否|
|knn|0.506364|0.509993|0.987272|是|

## 4. Bootstrap和解释边界

旧C0开发指标与分组Bootstrap不构成全新盲测；C3的既有Test结果也已公开。C23/C24原与重建对应同一4个Session，Bootstrap必须同源成对重采样，不能当8个独立Session；仅4组的推断能力有限。本轮未新运行Bootstrap。

D高时若两域大多被判Real，BA会靠近0.5，但AUC仍可保留显著排序差异；这不是阈值需要被放宽的理由。高D、输出finite、输出数值不同和Patch总量接近均不足以单独证明完整迁移。P28 Real-only grouped-CV反映Session漂移，不是当前冻结C0的理论可达上界。


## 路径简称（各表中的相对文件名以此为根）

- **C3**：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`。
- **C4**：`outputs/tactile_p43_c4_root_cause_audit_2026-11-16/`。
- **C13**：`outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18/`。
- **C23**：`outputs/tactile_c23_real_reconstruction_c0_2026-11-18/`。
- **C24**：`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/`。
- **C25**：`outputs/tactile_c25_latent_gmm_ot_feasibility_2026-11-18/`。
- **C26**：`outputs/tactile_c26_shared_frozen_encoder_2026-11-18/`。
- **C27**：`outputs/tactile_c27_c3_route_improvement_2026-11-18/`。
- **P39**：`outputs/tactile_p39_domain_decomposition_2026-11-15/`。
- **GMM**：`outputs/tactile_gmm_dual_vae_2026-11-16/`。

证据：L0仅定位；L1历史报告/配置声明；L2本轮直接核对源代码、指定Train/Validation数组、CSV及Hash；不执行L3。全部实际读取文件及SHA256见`audited_sources.json`；函数起止行见`source_code_index.json`。

## 5. 目标读出和替代路线的限定机制结果

C24/Real_C13_C24_structure.csv在同780 Real窗上：C13配对L_temporal=0.0028149674，C24=0.0016136414，按1−C24/C13计算约42.68%降低；MSE从0.026319602升至0.027157449（约+3.18%）。Mask逐位不变的登记与IoU0.51171548一致，但不能仅由IoU相等反推逐位相同。平均绝对帧变化0.0055292053→0.0025255096，Real为0.00063119946，约8.76×→4.00×；P95和P99下降伴样本差异收缩，联合通过6→2。C24/stage0.json确认四帧按原保留完整帧顺序、未跨Trial/Stage，Real并非均匀20Hz；这是索引增量一致性，不是物理速度误差。

C25/coordinate_gate.json、decision.json：两个独立4×64表示不能直接共坐标比较，cross-native距离与OT门槛未通过。文件gmm_model_selection.csv、ot_plan_audit.csv存在不代表已实际拟合/求解；状态为NOT_EVALUATED。其限制不否定C3受监督Translator的坐标映射。

C26/decision.json、C0_complete.json：共享C13 Encoder建立共同数值坐标，不提供共同物理语义；输出输入保持条件失败后停止，C0新推理0次、joint_pass_count=null，不写0/11。

C10-R1/decision.json与paired_metrics.csv：Seed11，PairedVal1200/240中的240验证窗，GRU latent MSE0.7525827、decoded MSE0.08118970、IoU0.4966360；v-prediction八个单样本误差均值为1.1580479、0.05348208、0.4879714。latent误差更大、解码MSE变小而IoU略低，预注册的三项共同改善门槛未通过；不选择best-of-K。八样本平均输出仅诊断，不代表单次MAP生成。原epsilon实验爆炸与v采样稳定性改善分开解释；该固定配置停止，不泛化为所有扩散模型无效。本轮不加载任何Natural Test，R1历史亦无Natural/C0评价。

C7–C9的连续场、Taxel积分与处理诊断只解释空间可分辨性，不是新Translator成绩。C9已有29多接触执行、330 Patch帧归因至25命令配置/16放置状态/1reset，Root独立性UNKNOWN；连续支持与最终processed的四连通指标不同，不把多帧当独立样本。具体采用哪张图须另核实其物理量单位和来源，不混入原C3触觉分布主表。
