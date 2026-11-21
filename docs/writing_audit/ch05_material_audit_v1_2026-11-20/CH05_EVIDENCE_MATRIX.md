# 第五章逐节证据矩阵

本表只登记可写论点，不把某行代码存在当作有效研究结果。附属claim_cards.csv记录具体数值选择器。

|小节/论点|实际代码位置或模型合同|已有结果/原始产物|本轮级别|允许表述与限制|
|---|---|---|---|---|
|5.1 Mask/幅值/Full均残留差异|P39/code/run.py::features/pool及表示构造；P37/common.metrics|P39/mask_statistics.csv、amplitude_statistics.csv、probe_metrics.csv、temporal_statistics.csv|L2 CSV与代码|旧P37开发差异；不是新C0主验收，非完整因果分解|
|5.1 幅值诊断表示实际是全帧正值分位数|P39/code/run.py 18–23行，registration|4×5正值P10/P25/P50/P75/P90，padding接口；representation_counts.csv|L2|不是每Patch固定分位数；排除含全零帧窗口，因此其分母不同，不能直接作信息量减法|
|5.1 原生touch与processed不同|source/sensors/tactile/dex_hand.py::augment_spec/bind/read_raw/read；signal_processing.py::process|C2/sensor_metadata与taxel_geometry_mapping；第二章已核验机制|L2代码，历史执行L1|仿真内部712通道与16Patch可追溯，Real物理方向仍未认证；不重写第二章|
|5.2 三类数据与无物理配对|C3/common.py::real/natural；pipeline_cpu.py|Real5153/780、Natural4049/905、Paired1200/240，注册/来源CSV|L2角色、配对数组|Natural参与Sim VAE且为部署输入；Paired用于Translator，不是同步物理接触|
|5.3 GMM仅Train拟合、K/BIC/稳定性|GMM/code/fit_gmm.py::samples及主流程；gmm.py::fit_em|gmm_bic.csv、gmm_parameters.npz哈希链|L2|二维索引偏移先验；不冒称力场/压力重建|
|5.3 软分量到单一物理候选|C3/code/expand.py、inverse.py、matching.py::inherited_metrics|inverse_gmm_*、pair_expansion_manifest.csv、paired_train/validation.npz|L2代码与指定数组|责任度已用；top5和单一最佳候选压缩多解，无后验加权Translator监督|
|5.3 480字典不等于480独立Root|C3/code/dictionary.py及physics.py::reset/run|dictionary_manifest.csv、dictionary_responses.npz|L2|16×5×3×2，放置后指纹160；仿真内部映射|
|5.3 来源交叉/加载饱和|C4/code/pairs.py、input_ambiguity.py；C3/expand.py深度段|paired_source_inventory.csv、inverse_control_saturation.csv|L2 CSV重计|1200/240数值唯一，但机器人reset复用、近拓扑交叉；不写全局无泄漏|
|5.3 C5/C6未解决全部覆盖|对应code/与summary|C5完整与同目标分母；C6实际S1只有1次，S2未执行|L1 summary，L2实现|局部改善、支持不足；不证明五球理论无效，不宣称扩展成功|
|5.4 两套独立VAE|C3/models.py::Encoder/Decoder/DomainVAE；pipeline_cpu.py::train_vae|各history.csv、complete.json、checkpoint.pt及converter_frozen.json|L2|4×64同维、无共享权重/天然语义；域内重建不等于迁移|
|5.4 C13只修幅值上下文|C13/code/run.py、registration|training_summary、checkpoint、C23/reconstruction.json|L2|594512可训练参数，Encoder/Mask冻结；不改称新联合共享VAE|
|5.5 GRU潜变量映射及冻结|C3/models.py::Translator、pipeline_cpu.py；inference.py::Converter.transform|translator/history.csv、complete、冻结权重|L2|Huber+0.1 RealDecoder NLL；两个Encoder及Decoder冻结，梯度可经冻结Decoder回传到Translator|
|5.5 Paired支持不足、监督歧义|C4/code/latent.py、input_ambiguity.py|distance_summary.csv、input_mapping_ambiguity_summary、paired_same_targets_oracle_vs_translator.csv|L2 CSV/代码|同一Sim Encoder内可比较Natural/Paired，不能跨两个独立Encoder裸欧氏匹配|
|5.6 原11冻结C0协议|C3/c0.py、P37/common.py::metrics；各后续评价器脚本|c0/frozen.json、scaler、11checkpoint、Validation指标|L2 Hash与CSV|PRIMARY/SECONDARY严格分开，D专项/原联合分开|
|5.6 C23/C24机制对照|C23/evaluate.py；C24/code/train.py及registration|C23/metrics、reconstruction；C24/metrics、Real_C13_C24_structure.csv|L2|6→2仅同源Real自重建读出，不是Natural通过数变化|
|5.6 C25/C26停止|C25/code/audit.py；C26/code/run.py/evaluate.py|coordinate_gate、decision、C0_complete|L2元数据与代码|OT未执行；C26 C0未运行，不能写0/11|
|5.6 C27专项边界|C27/config、candidate checkpoint与release models|metrics.csv、acceptance_scope.json、实际模型Hash|L2|D11/11、联合2/11；固定原双VAE，不混C13|
|5.7 引向第六章|P44–P51各报告仅边界核对|P51 NO-GO及P52/P53未执行|L1|提出后续研究问题，不预告对抗已解决|

## 可以复用的定量差异材料

|角色|域|窗口/组|精确正值比例|
|---|---|---:|---:|
|train|Real|5153/32|0.175996|
|train|raw|1824/36|0.133069|
|validation|Real|1408/7|0.171908|
|validation|raw|406/8|0.126286|

同一P39统计口径Train正值中位数：Real0.00500250，旧原Sim0.0000103974；P95分别1.5010055、0.4594847。数值只支持该旧数据版本的描述差异。P39校准前后Mask比例完全相同是已有一致性检查结论，不能由总体比例相同单独推导逐位一致。

P39各表示最差D（11实例）由原CSV复算：

|表示|最差D|BA范围|
|---|---:|---|
|mask_raw|0.000000|0.994318–1.000000|
|mask_before|0.027185|0.750170–0.986408|
|amplitude_raw|0.006433|0.763211–0.996784|
|amplitude_before|0.183886|0.611766–0.908057|
|amplitude_after|0.149358|0.739630–0.925321|
|full_raw|0.024435|0.975467–0.987783|
|full_before|0.027562|0.755969–0.986219|
|full_after|0.057256|0.781098–0.971372|

这些表示有相同五架构与深度固定Seed，但分母/特征内容/预处理各自拟合；Amplitude剔除含全零帧窗口。不能将不同表示D相减解释成Mask贡献百分比。P40是共享C特征4×64，没有新712D Decoder；所选候选最差D0.2427仅历史报告结果，不能当本章完整712D重建成绩。

## 图表来源建议

1. 差异图优先从P39/mask_statistics、amplitude_statistics、temporal_statistics按Train/Validation单独绘制；用16个Patch方阵展示，不把712通道全拉平替代空间图。
2. 物理链图可用C3字典manifest/plan与C2仿真内部映射；取指定Train/Validation实例，非自然实机对应图。
3. 模型图按C3/models.py画训练/推理分离图，标C13与C27支线版本；不得拼合参数形成不存在的最终模型。
4. 覆盖图从C4/distance_summary中仅Train/Validation重绘。原PCA/示例PNG可能混有历史Test，未经逐图来源核实不得直接用于正文。
5. 评价表使用本目录Validation逐实例CSV；C3 historical Test不取样本或预测，不用旧Test示例冒充Validation。

所有图本轮仅规划，不生成图、不编译、不写正文。既有可视化路径是否可以直接使用需以元数据核实，不能只看图名。


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

## Session、Task与时间合同补核

C23/Session_reconstruction.csv的四个Session并非完全相同行为：

|Session|窗口|同源MSE|Mask IoU|
|---|---:|---:|---:|
|20260814_150537_stable_maidong_30|240|0.023287|0.589238|
|20260814_151336_stable_maidong_30|240|0.028871|0.538483|
|20260816_172352_slip_3D_bottle_C_30|150|0.024078|0.476124|
|20260816_173235_slip_3D_bottle_A_30|150|0.029331|0.380443|

C4/reconstruction_by_task_stage_group.csv的Natural_validation、group_type=tasks记录的是**Sim VAE自身重建**，不是Sim→Real的物理对应：

|任务|验证窗|Sim同域Mask IoU|
|---|---:|---:|
|lift|111|0.402012|
|nut_assembly|302|0.600166|
|pick_place|243|0.304093|
|push|84|0.001120|
|stack|165|0.313884|

这说明任务构成与同域重建困难不同，不能把全局指标变化全部归为域迁移。Root/Session逐组原始行在同CSV group_type=g，阶段为s；未获确认的不同任务阶段不硬合并为共同物理状态。P39/stage_metrics.csv按CLOSE/HOLD/LIFT提供旧协议分阶段BA/AUC/D，不能用来补齐新Natural任务的共同条件认证。

P37/time_contract_diagnostics.json历史统计中，Real帧间隔中位约0.06525秒、四帧跨度约0.18798秒；Sim约0.05/0.15秒。其分位数组和历史生成定义可追溯，但不是本轮重读原始时间戳得到的新统计；C24又明确Real是顺序保留帧而非每一硬件采样。因此所有变化量只按保留帧索引解释，不能作同物理时间动态强弱的直接比较。
