# 全文图表与排版清单

当前实际接入正文：32图、32表、32个equation环境。11处includegraphics全部存在，格式均为PDF；标签无重名、ref/eqref/autoref无缺失目标。详细资产Hash、页尺寸、内嵌栅格尺寸见figure_inventory.json。编号由现有LaTeX自动生成，独立复编无未定义/重复标签警告。

11项外部图片没有相同字节Hash重复；均以width指定尺寸，未同时强制width/height，未发现拉伸指令。10个外部PDF为纯矢量；组合机器人PDF含4个栅格部件，不能将整个PDF称为纯矢量或为这些局部模型绘制统一声明300DPI。局部低像素片段应在后续排版阶段放大检查。其余图为TeX原生框图/流程图或占位。没有本轮新增实验图混入正文。

3张图无显式正文ref：fig:ch02_validation、fig:ch04_timing、fig:ch05_c0_validation_protocol。建议后续补正文图号指引；不是缺失图资源。图3-6仍为明确占位，失败影像缺失，不能称已补齐。

逐图逐表清单（图注采用源首行，完整内容见原TeX；行号以去注释后的读取结果定位，最终以label搜索为准）：

|类型|源文件:行|标签|正文引用数|图注首行|
|---|---|---|---:|---|
|table|thesis/body/ch01_introduction.tex:59|tab:related_work_routes|1|相关技术路线及其监督与适用条件}|
|figure|thesis/body/ch02_simulation_platform.tex:13|fig:ch02_architecture|1|仿真平台模块及控制、观测数据流}|
|table|thesis/body/ch02_simulation_platform.tex:29|tab:ch02_modules|1|仿真平台主要模块与功能}|
|figure|thesis/body/ch02_simulation_platform.tex:57|fig:ch02_robot_model|1|由当前XML与STL离线组合的机械臂和手部模型（静态关节零位，右侧为手部放大图；未显示底座，非任务执行截图）}|
|figure|thesis/body/ch02_simulation_platform.tex:73|fig:ch02_drive|1|当前模型的关节与驱动关系}|
|table|thesis/body/ch02_simulation_platform.tex:85|tab:ch02_actuators|1|机械臂及灵巧手执行器配置}|
|figure|thesis/body/ch02_simulation_platform.tex:135|fig:ch02_control|1|指定配置下的控制层次与多速率执行流程}|
|table|thesis/body/ch02_simulation_platform.tex:150|tab:ch02_control_parameters|1|主要控制与物理参数及其适用范围}|
|figure|thesis/body/ch02_simulation_platform.tex:183|fig:ch02_patch_order|1|五指与手掌触觉阵列的维度及索引组织}|
|figure|thesis/body/ch02_simulation_platform.tex:213|fig:ch02_fingertip|1|当前指尖STL及外壳拟合后的阵列采样点（网格局部坐标）}|
|table|thesis/body/ch02_simulation_platform.tex:222|tab:ch02_layout_parameters|1|不同曲面布局方法的关键实现参数}|
|figure|thesis/body/ch02_simulation_platform.tex:239|fig:ch02_frame|1|由相邻采样点构造的局部切线和法向（外向修正前）}|
|table|thesis/body/ch02_simulation_platform.tex:254|tab:ch02_patch_layout|2|16个Patch的阵列规格与712维索引区间}|
|figure|thesis/body/ch02_simulation_platform.tex:291|fig:ch02_processing|1|原生触觉读数到processed观测的处理链}|
|table|thesis/body/ch02_simulation_platform.tex:319|tab:ch02_signal_stages|1|触觉观测各阶段的输出与语义}|
|figure|thesis/body/ch02_simulation_platform.tex:345|fig:ch02_validation|0|平台实现与历史观测记录的分层核查结构}|
|table|thesis/body/ch02_simulation_platform.tex:362|tab:ch02_episode_validation|1|五条历史开发Episode的触觉观测核查结果}|
|figure|thesis/body/ch03_grasp_demonstrations.tex:19|fig:ch03_overall_pipeline|1|抓取候选到专家示教的分级构建流程}|
|table|thesis/body/ch03_grasp_demonstrations.tex:45|tab:ch03_candidate_parameters|1|候选优化参数及约束的实际作用}|
|figure|thesis/body/ch03_grasp_demonstrations.tex:67|fig:ch03_candidate_sources|1|已有候选模块在分级生成流程中的职责}|
|figure|thesis/body/ch03_grasp_demonstrations.tex:115|fig:ch03_reachability|1|局部腕姿态扩展与两个目标的可达性筛选}|
|figure|thesis/body/ch03_grasp_demonstrations.tex:156|fig:ch03_strict_replay|1|分段轨迹执行与Strict Replay质量验收}|
|table|thesis/body/ch03_grasp_demonstrations.tex:173|tab:ch03_strict_thresholds|1|当前final配置的Strict Replay验收条件}|
|table|thesis/body/ch03_grasp_demonstrations.tex:209|tab:ch03_task_stages|1|五类任务的示教阶段与主要执行约束}|
|figure|thesis/body/ch03_grasp_demonstrations.tex:228|fig:ch03_task_transfer|1|多任务示教的共用来源与任务专属控制关系}|
|table|thesis/body/ch03_grasp_demonstrations.tex:254|tab:ch03_historical_batches|1|历史批量路由与任务记录及其统计限制}|
|table|thesis/body/ch03_grasp_demonstrations.tex:275|tab:ch03_sugar_strict|1|Sugar Box局部扰动示教的历史Strict结果}|
|table|thesis/body/ch03_grasp_demonstrations.tex:299|tab:ch03_generation_failures|1|历史90次扰动尝试的汇总分类}|
|figure|thesis/body/ch03_grasp_demonstrations.tex:318|fig:ch03_examples|1|典型成功与失败案例的待补原始画面}|
|table|thesis/body/ch04_multimodal_data.tex:15|tab:ch04_fields|1|主要数据字段的维度与实际语义}|
|figure|thesis/body/ch04_multimodal_data.tex:37|fig:ch04_carriers|1|数据载体及其转换关系示意}|
|figure|thesis/body/ch04_multimodal_data.tex:68|fig:ch04_timing|0|执行前采集与历史执行后记录的索引关系}|
|figure|thesis/body/ch04_multimodal_data.tex:91|fig:ch04_multirate|1|控制记录与RGB采样频率分离的合同示意}|
|table|thesis/body/ch04_multimodal_data.tex:106|tab:ch04_recording_versions|1|不同记录路径的时序与版本边界}|
|table|thesis/body/ch04_multimodal_data.tex:149|tab:ch04_quality_gates|1|分层数据质量检查及其判定范围}|
|figure|thesis/body/ch04_multimodal_data.tex:173|fig:ch04_admission|1|分层质量审查与数据准入关系}|
|table|thesis/body/ch04_multimodal_data.tex:212|tab:ch04_lerobot_inventory|1|三组LeRobot数据的实际数值库存}|
|table|thesis/body/ch04_multimodal_data.tex:231|tab:ch04_version_quality|1|旧在线记录与v4开发包的质量对照}|
|table|thesis/body/ch04_multimodal_data.tex:253|tab:ch04_sink_precision|1|Sink历史回放状态与精度对照}|
|figure|thesis/body/ch04_multimodal_data.tex:267|fig:ch04_sink_precision|1|由已有报告绘制的Sink回放偏差对照（历史记录，非本轮新实验）}|
|table|thesis/body/ch05_sections/section_5_1.tex:15|tab:ch05_domain_feature_difference|1|旧P37数据校准后触觉的表示分解诊断}|
|figure|thesis/body/ch05_sections/section_5_3.tex:7|fig:ch05_gmm_physical_pipeline|1|GMM统计先验引导仿真接触与伪配对构建流程}|
|table|thesis/body/ch05_sections/section_5_3.tex:42|tab:ch05_probe_dictionary|1|球形探针响应字典的组织}|
|figure|thesis/body/ch05_sections/section_5_3.tex:62|fig:ch05_gmm_candidate_selection|1|软分量分解、候选压缩与单一伪配对选择示意}|
|figure|thesis/body/ch05_sections/section_5_4.tex:7|fig:ch05_dual_vae|1|双域VAE独立表示学习与条件解码结构}|
|figure|thesis/body/ch05_sections/section_5_5.tex:29|fig:ch05_translator_paths|1|伪配对监督训练与自由推理的数据流}|
|figure|thesis/body/ch05_sections/section_5_6.tex:7|fig:ch05_c0_validation_protocol|0|迁移前冻结C0的Validation评价协议}|
|table|thesis/body/ch05_sections/section_5_6.tex:24|tab:ch05_transfer_summary|1|同划分Natural Validation的跨域评价汇总}|
|table|thesis/body/ch05_sections/section_5_6.tex:36|tab:ch05_validation_instances|1|C3与C27的冻结Validation逐实例结果}|
|figure|thesis/body/ch06_sections/section_6_1_shared_vector_field.tex:41|fig:ch06_shared_vector_field|1|共享参数向量场迁移框架。灰色模块冻结，蓝色模块可训练；积分展开复用同一组参数，四帧序列在各次向量场调用内部处理。}|
|table|thesis/body/ch06_sections/section_6_1_shared_vector_field.tex:54|tab:ch06_mapping_comparison|1|直接潜变量映射与共享向量场方法的关系}|
|figure|thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex:31|fig:ch06_mask_gradient_paths|1|Decoder与判别器输入端的两处Mask梯度路径。实线为前向，虚线为生成器更新时的反向；B组的判别Mask支路中断，C组以代理导数传递，连续支路保留。}|
|table|thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex:38|tab:ch06_discriminator_comparison|1|判别器输入配置及原冻结C0联合结果}|
|figure|thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex:7|fig:ch06_f0_swd_baseline|1|统一评价中Original与初始生成器的SWD对照。相对下降及区间均取自既有组Bootstrap记录，不代表完整迁移通过。}|
|figure|thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex:22|fig:ch06_round_acceptance_states|1|各判别器状态下生成器更新前后的接受率。每组仅在同一冻结D内比较；第0轮为原域判别器诊断，Real接受率按Session宏平均。}|
|figure|thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex:31|fig:ch06_multi_round_distribution|1|固定分布评价合同下各生成器状态的SWD与MMD。两项均越低越好，生成结果未随优化轮次单调改善。}|
|figure|thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex:17|fig:ch06_multiview_refiner_architecture|1|残差生成与多视角判别结构。虚线表示冻结的双域VAE；Patch仅按既定通道索引分组。}|
|figure|thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex:28|fig:ch06_multiview_gradient_training|1|生成阶段各Epoch首步的分支梯度范数及逐Epoch平均对抗项。曲线来自历史训练日志，梯度非零不代表质量改善。}|
|table|thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex:35|tab:ch06_multiview_results|1|固定分布评价下的生成结果与终点诊断}|
|table|thesis/body/ch06_sections/section_6_5_distribution_quality_and_input_preservation.tex:16|tab:ch06_quality_dependency_comparison|1|共同评价合同下的分布、邻域与依赖指标}|
|table|thesis/body/ch06_sections/section_6_5_distribution_quality_and_input_preservation.tex:45|tab:ch06_quality_negative_controls|1|同源Validation派生负对照的互补诊断}|
|table|thesis/body/ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex:10|tab:ch06_real_session_reliability|1|开发Real Session的Joint接受率}|
|table|thesis/body/ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex:32|tab:ch06_loso_envelope_coverage|1|各判别分支的LOSO包络覆盖率}|
|table|thesis/body/ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex:47|tab:ch06_score_envelope_diagnostic|1|Real校准分数包络与生成样本上界超出比例}|

## 页面及警告

全93页文本边界扫描无超出MediaBox；无完全空白页。与B1逐页文本一致；已有32图/32表标题可解析，未见图注与图片跨页分离的新增问题。渲染抽查物理页17、29、43、55、66、70、78、80、85、86，见page_checks。页面55章末留白较多，属原有现象；图6-1、6-6/6-7及质量表可读，无可见重叠。第一章两处Overfull为5.79674pt及2.0989pt，编译日志原行78指向其表格内容，不能因页面边界扫描通过而忽略。后续可局部调整列宽，不改全局模板。

字体rsfs小字号替代、未启用glossary警告和42条BibTeX字段警告维持原状态；文献citedate/address/year缺项应逐条核实，不能猜补。图表浮动、章末留白不在本轮擅自修改。目录、公式及交叉引用独立复编已解析；全面检查为静态+日志+全页边界扫描，视觉检查是抽页，不宣称93页均人工放大审核。
