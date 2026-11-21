# 第五章材料盘点与写作就绪性评估

日期：2026-11-20。任务：CH05。**最终决策：可以直接进入正文写作。** 本句是就绪性判断，不是正文写作授权；本轮结束后等待用户与ChatGPT审核。

## 十项结论

1. **已有材料足以形成完整、可信的第五章初稿**。证据主线为“观测差异→无同步配对的监督缺口→Real统计GMM引导实际球形探针→独立双域VAE→GRU Translator→完整712D输出→冻结评价及失败归因”。不能把“链路实现”写成“迁移目标完成”。
2. **不需要新增实验才能写可信初稿**。现存代码、有效配对数组、45轮训练日志、冻结权重、逐实例CSV和后续失败诊断足够支持限定结论。缺少物理真值等事项可直接作为局限，不以本任务新增研究补齐。
3. **5.3、5.4、5.5材料最充分**；5.6有完整指标与失败证据但最易混用版本。5.1的历史差异数据充分，受任务构成/采样合同影响；5.2缺少认证共同物理语义，恰是问题定义的限制，不需虚构条件补足。
4. **不能说统计GMM潜力已充分挖掘**。责任度确实已用于统计分解，而分阶段先验、分量信息、保留的候选多解尚未作为Translator概率监督利用。这里只确认可用信息与实施缺口；没有证据证明软监督必然更好。C25否决独立潜坐标间直接OT，不能否决统计GMM探针构造。
5. **Dual-VAE完成各域独立表示和域内重建；Translator完成有界伪配对上的候选跨域映射**。二者不提供天然公共潜坐标，不保证Natural支持覆盖，也没有物理配对真值。C13仅修Real幅值条件路径，C27仍使用原C3双VAE，不可合并成同一升级模型。
6. **初始跨域重建未通过原冻结C0联合验收**。本轮由已有Validation CSV复核：原C3映射（C11-F复现）联合1/11，C27联合2/11；两者D均11/11。C3最终报告的1/11来自另一历史开发Test角色，不能以它替代Validation。C23的6/11、C24的2/11都是Real自重建读出诊断。
7. **应正文保留**：C3主链、C4覆盖/监督歧义、C13/C23目标侧读出限制、C24时序改善与质量退化并存、C27专项与联合口径区别。C5/C6用于解释配对限制；C7–C9压缩为物理读出归因，不能另写一条已成功迁移路线；C10/R1、C25/C26及P28宜短述或汇入限制表。
8. **建议8000–9000 TeXcount，中心8500**。按本轮七节预算非平均分配，见结构报告。不是本轮已写字数，也不靠罗列全部实验凑篇幅。
9. **第五章到初始潜映射及其局限为止**。P44共享向量场、P45梯度路径、P46–P48质量评价、P49/P50对抗及退化、P51可信度NO-GO留第六章。K为Euler积分步数，不是对抗训练轮数。P52/P53未执行。
10. **核心证据缺口仍存在，但不阻断诚实写作**：Real硬件映射与同步配对、统一共同条件、Natural的正确Real目标、独立新盲验、真实任务收益均未获认证；因此不能写“实现物理精确迁移”“全部11联合通过”“对机器人成功率有提升”。

## 本轮实际复核范围

只读代码、已有Train/Validation配对数组及字典、CSV、注册表和模型Hash。未加载Test/Holdout样本或预测文件，不访问服务器，不运行模型推理、训练或物理仿真。历史报告中的Test汇总只用于辨认结果角色，不用来挑模型/配置，不另行复算Test。

直接读取paired_train/validation各sim与real数组，确认1200/240个完整4×712窗口、finite、非负及各自数值唯一；字典480行、16 Patch、160种登记放置后初态指纹。来源CSV重计：Train1200源/566Trial/30Session，Validation240源/106Trial/4Session；两组机器人reset各仅一种，19种放置状态和17种拓扑跨组复用。**1440数值窗口不等于1440独立Root。**

核对原C0全部11实例、Scaler及C3三个转换权重实际SHA256与冻结登记相符；C27验收包的Real/Sim VAE和GRU权重亦与对应源一致。逐实例重算既有CSV中BA、D与联合判据，并由混淆矩阵核对BA；没有重新推理或拟合AUC。数字选择器、Seed、数据角色、分母及允许表述见claim_cards.csv。

## 写作时保持的三条区分

统计GMM的索引空间不是VAE潜空间；独立Encoder latent维数相同不是共同语义坐标；被冻分类器判为Real不是分布质量与输入结构同时通过。全局稀疏率接近Real也不意味着Taxel激活位置一致。

审计全过程与代码起止行、参数来源、Hash见附属JSON。L1报告记载、L2直接复核和UNKNOWN分别保留；本轮不产生L3证据。


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

## 双域表示和Translator的实际训练合同

C3/models.py的Encoder接收[B,4,712]，拼接精确正值Mask与log1p(x)/encoder_scale形成每帧1424维，经Linear1424→128、GELU、LayerNorm和GRU128，分别输出4×64的mean、logvar（后者限[-8,4]）。训练以mu+exp(logvar/2)·epsilon重参数，推理取posterior mean。两个域模型分别创建、分别拟合尺度和训练，不共享参数。

Decoder按帧自回归：前帧Mask的32维嵌入（首帧BOS）与当前64维latent一起进入GRUCell96→128；Mask头输出712logits，推理以logit≥0作MAP。Amplitude以128维隐状态与当前Mask的32维嵌入为条件，为每Taxel输出两个Gamma分量的混合logits/shape/rate；推理取混合均值乘Mask。当前C3已经是Mask自回归机制，不能照搬更早阶段“每帧独立Bernoulli”诊断。

VAE损失为Mask BCE与正值Gamma-mixture NLL（每窗按正值数归一化）加0.001 KL。NLL包含positive_scale的Jacobian，似然数值可为负，不能用其符号判断训练失败。训练NLL使用目标Mask上下文，部署自由运行使用预测Mask，C13专门修复其中的幅值条件差异；Teacher Forcing损失不能当自由输出质量。

|模块|Seed|实际Epoch/最佳Epoch|Steps|参数量|训练来源|
|---|---:|---|---:|---:|---|
|C3 real_vae|4301|45/30|3645|1094232|5153 RealTrain|
|C3 sim_vae|4302|45/30|5715|1094232|4049 NaturalTrain+1200 PairedTrain|
|C3 translator|4303|45/30|855|181824|1200 PairedTrain→对应Real目标|

Real每Epoch81step；Sim每batch32Natural+32Paired、每Epoch127step；Translator batch64、每Epoch19step。Sim混合采样重复遍历Paired并补足Natural批，完整覆盖不等于每个样本恰好访问一次；原高级索引计数纠正不改变训练权重或模型。C3三个模块均45轮、按额外ep≥30限制只允许后期checkpoint，均选30；Real历史最低Validation NLL出现在第4轮，早期权重未保存，不能回溯声称其分类成绩会更好。

Translator是两层GRU64→128加Linear128→64，181824参数。两个VAE全部冻结后，以Paired Sim的Sim Encoder posterior mean监督到对应Real Encoder的mean，L=Huber(T(z_s),z_r)+0.1·NLL(D_r(T(z_s)),x_r)。冻结Decoder不等于切断对输入latent的梯度；只有Translator更新。部署链是Natural processed→Sim Encoder mean→GRU→Real Decoder→4×712；不使用Real窗口、GMM、探针、域标签或Sim Decoder生成最终输出。

**独立潜坐标并不使配对监督Translator在数学上无效**：它恰是在学习两套坐标之间的函数。被C25阻断的是未证明公共坐标就直接对两套latent算欧氏/W2并做OT，不是Huber监督本身。映射可靠性仍依赖伪配对是否携带可辨识信息。

C4/translator_paired_error.csv显示Paired Train/Validation Huber约0.25565/0.32592，优于均值参考，说明学到了一定配对关系；Natural没有正确Real目标，不能给其latent RMSE。C4的近输入—远Real目标现象支持监督歧义风险，不是网络完全没训练的证据，也不是更大网络必然有效的证据。

C13不重建新架构，只训练decoder.current及decoder.amplitude共594512参数；Seed4313，实际18轮、最佳3轮、1458step，Real5153/780。Encoder、Mask、循环骨干及Sim VAE/Translator仍冻结。C23仅对这一版本已有Real重建输出做冻结C0读出。C24延续同一最小参数集合加保留帧增量损失，未得到无损改进。

C4的Sim positive_scale=1e−7来自NaturalTrain+PairedTrain正值中位数8.78499e−9触发下限；Encoder实际采用log1p(x)/0.04088746，不能把似然缩放当Encoder输入直接放大。NaturalTrain与PairedTrain正值中位相差约七数量级，存在数值条件风险，但没有受控证据证明它是唯一或主要根因。完整量纲、Jacobian和极端值记录见C4/scale_formula_audit.json、amplitude_scale_audit.csv。

补核：已直接以CPU、weights_only=True读取三个C3与C13 checkpoint元数据及张量形状，C3所存Epoch均30；C13仅6个current/amplitude张量改变。未构造或运行模型，见checkpoint_metadata.json。
