# 5.2逐段论点与证据

本轮只写正文及静态检查；未重算数据指标、未执行模型。L2指已有直接代码/合同核对及本轮只读代码材料，不提升为新实验L3。无新增外部引文，无BibTeX变更。阅读稿由section_5_2.tex机械转换：删除注释与label、转换section与数学环境；不另行改写文字。

|段落/源行|核心论点|直接证据与等级|与相邻章节的关系|不能推出|
|---|---|---|---|---|
|1，第5行|processed Natural输入、完整非负4×712、帧序和索引；Real风格不等于实测输出|5.2审计PROBLEM_FORMULATION §1–2；C3/code/inference.py:1–19、models.py:18–26，已有L2；C24/stage0.json关于保留帧及非均匀20Hz为既有登记L1|接5.1，不复述其统计数字|同shape等于同物理Taxel；帧差等于物理导数；所有Real分布特征已匹配|
|2，第7行|Real/Natural/Paired用途；1200/240代理监督而非独立1440 Root|DATA_ROLE_MATRIX；C3/code/pipeline_cpu.py:16–59、61–77；paired_manifest.csv及已审计配对来源，已有L2|引5.3构造依据，不展开GMM/字典/探针设计|统计伪配对等于同步物理真值；Natural直接参与原Translator监督|
|3，第9–16行|F定义、双重目标、物理及潜坐标限制；训练误差不同于自由输出评价|PROBLEM_FORMULATION §2–4；C3/code/inference.py:15–19、models.py:18–33、pipeline_cpu.py:54–77；冻结5.4:55、59及5.5 R1监督路径，已有L2|为5.4/5.5留出表示/映射职责，不新增联合损失|同维潜变量已对齐；Huber/NLL改善证明Natural输出正确；真实任务收益已实现|
|4，第18–23行|输出/域混淆/物理任务三层分析；原11C0门槛与BA/D等价；目标未完成|EVALUATION_CONTRACT §1–3；C3/code/c0.py:10–16、37–43，c0/frozen.json；P37/code/common.py:23–27；冻结5.6:5–20、65；已有L2与历史结果L1，不新执行评价|定义5.6评价目标，不提前列C3/C13/C27成绩|三层均已认证通过；D与BA是独立证据；有限模型不可辨识等于所有域信息消失|

路径简称C3=outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16；P37=outputs/tactile_p37_full_data_revalidation_2026-11-15。材料报告根目录为docs/writing_audit/ch05_52_material_audit_v1_2026-11-21。冻结章节位于thesis/releases/对应5.1/5.3/5.4/5.6目录；5.5仅R1及临时编译状态，未称冻结。

本文未新增cite或ref；通过内容衔接引出后续章节，避免独立片段出现未解析外部编号。公式只定义映射与既有D目标，不表示新增优化目标或算法贡献。
