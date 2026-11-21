# 5.5跨域潜变量映射专项材料审计

日期：2026-11-21。**材料就绪，可提交用户与ChatGPT审核后撰写5.5；无需新增实验。** 本轮不写正文，不修改5.3/5.4冻结版本，不执行模型forward、训练、物理仿真、服务器任务或C0重算。

## 1. 七项最终决策

1. **完整初稿是否可写**：可以。结构、参数、均值语义、梯度路径、目标、训练日志、配对索引及推理入口均有直接证据。缺失公平GRU消融不妨碍真实描述设计。
2. **是否新增实验**：不需要。未知项作为局限，不为论文补训练。
3. **贡献表述**：构建并实现“统计伪配对监督—双域独立表示—时序latent回归—冻结目标Decoder读出约束”的工程方法链，分析有限对应与泛化边界。两层GRU、Dual-VAE和Huber组合本身不宣称首次、最优或已证实原创优势。
4. **留给5.6**：配对误差曲线与均值参考、分Session反例、Natural覆盖/监督歧义、完整712D空间幅值时序、C0 BA/AUC/D逐实例联合结果。5.5以方法机制为主。
5. **结论限制**：无天然公共坐标；有监督函数可学，但伪配对非物理真值；低Huber不等于Natural转换成功；冻结Decoder NLL含真实Mask条件，不是自由生成质量。
6. **预算**：1100–1250 TeXcount，中心1175；5.6建议中心1605，连同5.3实际1999和5.4实际1496维持全章8500中心。未改规划。
7. **最值得强调的问题**：有限、单一选择的统计监督，能否支持从Natural输入到正确Real latent的泛化，而不是把“两个独立空间维度相同”误作天然对应。

## 2. 核心实现结论

Translator为2层单向GRU(input64,hidden128,batch_first)，Linear128→64，181824参数。训练和推理都输入Sim posterior mean，训练目标为同一伪配对Real posterior mean；训练不再次随机采latent。

训练目标为平均Huber+0.1×冻结Real Decoder的Mask BCE与正值Gamma混合NLL。Encoder和所有Decoder参数不更新，但Decoder对Translator输出的输入梯度没有被detach。只有T被AdamW更新。真实Mask条件训练与预测Mask自由部署的差异仍存在。

Seed4303，45轮×19步=855步；最终所存第30轮权重只经历570步。验证总目标第30轮最低，不重新选择。权重Hash见训练报告。

1200/240窗口逐行内容Hash与manifest一致。Real来源无跨划分重叠，但登记放置状态共享19种，reset复用；窗口数不等于独立Root。GMM软责任度和多候选不会作为Translator概率监督输入，现有映射是确定性的均值回归。

## 3. 版本归属

|实验|在本节的位置与证据依据|
|---|---|
|早期tactile_gmm_dual_vae|历史GMM/表示探索入口，读取其训练代码作路线区分，不据名称替代C3 models.py|
|C3|原始独立VAE与两层GRU主链；pipeline_cpu.py/inference.py及权重直接证据|
|C4|既有配对独立性、潜变量回归和歧义诊断；已核代码与CSV|
|C13|只Real幅值current/amplitude修订，原C3 Translator不被自动更新|
|C23|复用C13自重建输出诊断，不是新Translator|
|C24|同一幅值参数范围增加时序约束，不是C3整体重训|
|C25|否决未经验证公共坐标的原生latent直接GMM-OT，不是否决监督函数|
|C26|共享冻结Real Encoder的新数值坐标路线，语义未认证、输入保持未通过；非本节T|
|C27|原C3双VAE和Translator初始化，增加Natural覆盖及Real坐标矩/源响应约束；未使用C13/C26作为新的统一权重|

C27定位直接核对code/common.py:25–31及code/train.py:1–35/config.json。C25、C26状态按各自decision与代码范围记录，不以它们的否决扩大到所有迁移方法。C27后续性能仅供5.6，不回填原始C3。C0分类分数/预测未在本轮读取或重新计算。

## 4. 证据说明与交付

L2：原始代码、训练日志、Checkpoint形状/标量、Train/Val数组Hash与manifest；L1：既有诊断汇总中本轮未重算的物理reset/近构型统计。没有L3。

配套报告提供具体路径、函数与行号、公式、梯度图、索引/独立性、曲线与写作预算。source_hashes.json、checkpoint_metadata.json、pairing_recount.json和integrity.json支持追溯。完成即停止，等待审核，不自动开始5.5正文。
