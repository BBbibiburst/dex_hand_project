# TH-CH01-1.1-V2：1.1正文写作交付与自审

日期：2026-11-19。阶段3写作完成，等待用户审核；未编译、未冻结，不自动进入下一节。

## 1. 完整正文位置与修改范围

完整LaTeX正文见 [body/ch01_introduction.tex](../../body/ch01_introduction.tex)，从 `\section{研究背景与意义}` 到文件末尾，六个自然段。本章此前只有chapter标题和label，没有既有1.1需要替换；本轮保留章标题与label，在其后增加授权小节及 `sec:research_background` 标签。

约2100字符（不含引用命令和空白），汉字约1900，符合1800–2500字的篇幅要求；详细计数由静态检查保存。没有新增小标题、公式、图表、算法或其他小节。

修改的研究源文件仅两项：

|文件|变更|
|---|---|
|body/ch01_introduction.tex|新增1.1标题、标签和六段正文；移除已被本次正文替代的“正文待写”注释|
|reference.bib|原来只有占位注释，无可复用研究条目；保留原注释，新增本节实际引用的10项，未加入未引用W11|

原文件备份见 `ch01_1_1_draft_v2/ch01_before.tex.txt` 与 `reference_before.bib.txt`。另新增本交付记录、引用映射及静态核查记录，均不属于论文正文。没有调整七章规划、当前六章include关系、其他章节、封面、题目或模板。

## 2. 六段论证结构

1. 多指协调、动态接触与精细操作需求，落到观测、动作选择和执行反馈。
2. 视觉提供场景信息，触觉补充接触后局部状态；以已读视触觉研究说明动作调整价值，不声称触觉唯一。
3. 真实试验成本和安全约束引出仿真；将抓取候选、机器人可达性与执行验证连接到本平台。
4. 从专家示教转到数据内容、来源、采样及观测—动作关联；交代五任务开发记录和LeRobot构建，不列未经完整验收规模。
5. 从同维度不等价引出传感结构、接触读出和处理差异；说明本项目缺同步物理配对、统计伪配对监督有限。
6. 从平台、可执行示教、数据审计和重建探索说明工程及方法意义，不展开1.4方法清单或宣称成功。

## 3. 实际引用与BibTeX键

下列均来自已审核材料的W01–W10；题录及全文位置详见 [材料核验报告](ch01_1_1_material_review.md) 和 [literature_catalog.json](ch01_1_1_sources/literature_catalog.json)。未重新检索新论文或采用未核实来源。

|材料ID|正文所用文献|BibTeX键|引用段落|
|---|---|---|---|
|W01|OpenAI等，Learning dexterous in-hand manipulation，IJRR 39(1)，2020|openai2020dexterous|1、3|
|W02|Bohg等，Data-Driven Grasp Synthesis—A Survey，TRO，2014|bohg2014grasp|3|
|W03|Luo等，Robotic tactile perception of object properties: A review，Mechatronics，2017|luo2017tactile|2|
|W04|Calandra等，More Than a Feeling，RA-L，2018|calandra2018feeling|2|
|W05|Tobin等，Domain Randomization…，IROS，2017|tobin2017randomization|3|
|W06|Mandlekar等，What Matters in Learning…，PMLR 164，2022|mandlekar2022demonstrations|4|
|W07|Walke等，BridgeData V2，PMLR 229，2023|walke2023bridgedata|4|
|W08|Fang等，RH20T，ICRA，2024|fang2024rh20t|3、4|
|W09|Wang等，TACTO，RA-L，2022|wang2022tacto|5|
|W10|Si和Yuan，Taxim，RA-L，2022|si2022taxim|5|

BibTeX具体处理：

- W01使用2020年正式卷期，不采用2019在线年份；第一作者按登记的“OpenAI: Marcin Andrychowicz”保留为整体字段，随后列出其余作者，未将OpenAI与Marcin重复计为两个作者。
- W06、W07的作者字段及出版题名从已保存的PMLR正式BibTeX提取；W06使用164卷/2022，W07使用PMLR的作者顺序，不与PDF另一顺序拼接。PMLR的Fei-Fei姓名导出形式保留，若作者希望统一姓名显示，应在文献核验后单独处理，不默改元数据。
- W08使用已核验ICRA2024题录；阅读范围仍是此前说明的2023预印本相关正文，未谎称本轮通读最终出版版本。
- W06、W07未填猜测DOI；其余8项DOI均使用已核验值。对TACTO、RH20T、BridgeData V2、GelSight的大小写作BibTeX保护。
- W11本节未引用，未新增条目。保留其页码差异说明：IEEE出版登记2242–2251；所取得CVF PDF内页编号为2223–2232。以后如使用，按任务要求采用IEEE出版页码并继续保留来源差异，不能混成两篇论文。

## 4. 正文论点—证据对应表

|段落与关键论点|外部证据及实际阅读位置|内部证据与等级|边界自审|
|---|---|---|---|
|第1段：多指协调与接触变化；感知—动作反馈|W01 PDF2–3任务与系统概述|无需将本项目实验当通用背景证据|未把W01说成使用触觉的实验，未引用其性能数字|
|第2段：接近时遮挡、触觉信息与重抓取|W03 PDF2–3、9；W04 PDF3–5动作条件模型、数据及消融|本项目未宣称实现滑移检测|只述互补价值，不称触觉替代视觉或唯一途径|
|第3段：实机成本、安全、仿真条件及现实差距|W01 PDF2–3；W08 PDF4采集风险；W05 PDF2–3|source/robots/arms/rm75b.py；source/robots/hands/dex_hand.py；source/envs/rl_env.py，L2|无量化降本、零风险或仿真完全真实的结论|
|第3段：抓取候选不等于可执行专家示教|W02 PDF3–4，可达性过滤|source/grasping/executor.py::rank_candidates_for_scene；source/rl/grasp_edit/templates.py::build_grasp_edit_templates；GraspQP/DexEvolve相关实现，L2指定函数|未将开源方法称原创；未把IK预检等同全轨迹碰撞自由；未引入未核实算法对比|
|第4段：示教内容、观测和历史影响学习|W06 PDF3、5、7；W07 PDF3–4；W08 PDF3–5|source/data/production_recorder.py::_before/_after及旧executor._Recorder，L2|未将外部研究扩展成已证明本项目同步的收益|
|第4段：五任务开发记录与LeRobot构建存在|不是外部文献能替代的项目事实|五个source_episode/episode.npz，两个历史LeRobot目录实际Parquet；L2选定数值字段；success历史记录L1|没有成功率、全量小时数、独立Root规模；没有“全同步真实多模态数据集”|
|第5段：不同传感模型存在读出与校准假设|W09 PDF3–4、7；W10 PDF2–3|source/sensors/tactile/dex_hand.py::_add_touch_sensor/read_raw/read；signal_processing.py::process，L2|明确光学触觉文献不能认证本阵列物理等价；没有Pa/N标定断言|
|第5段：现有Real–Sim缺认证同步物理配对|外部文献不用于证明本项目缺口|P34 common_support_matrix.csv、P41 source_audit及712行mapping，L1；C3 matching.py，L2；paired_manifest.csv，L1|没有把same-frame raw/processed或统计伪配对说成同步接触真值|
|第6段：工程流程及方法研究价值|由前五段论证收束|R-01至R-08及E-C3等有范围的证据|价值表述不等于全部创新成立，不声称重建解决或实机提升|

原件全路径、源Hash与L2覆盖范围继承 [internal_evidence.json](ch01_1_1_sources/internal_evidence.json)。本轮已重新检查33项来源Hash一致；未以规划或旧聊天替代原件，亦未新运行物理或模型实验。

## 5. 自审结果

|检查项|结果与说明|
|---|---|
|论证逻辑|六段依次回答操作需求、触觉作用、可执行示教、数据使用条件、跨域困难及研究意义，未按P编号写进度|
|表达|未采用首先/其次/最后的固定列举，未使用显著提升/充分证明/全面优于；不以方法名堆叠代替背景|
|引用|10个不同条目全部实际引用，引用键可定位，引用紧随其支持的论点；未引用W11或未读候选|
|实现|RM75B-Dex Hand、原生touch后处理、五任务记录和LeRobot构建均限定为本地核实范围|
|事实边界|无全量同步、同步物理配对、全部Strict成功、VLA/实机性能增益或0.90已通过的表述|
|研究停止范围|未改P51结论，不将其结论扩展到P49；未写任何P52/P53结果或启动新实验|
|篇幅与结构|六自然段；无新公式、实验图或小标题；没有写1.2及其他章节|
|语法与文件范围|仅静态检查括号、引用键及文件Hash；这不是LaTeX编译或PDF排版验收|

自审未发现阻止提交审核的事实性冲突。自然表达与章节取舍仍须用户审核，不能以自审标记已获批准或已冻结。

## 6. 尚存问题和待确认事项

1. `thesis.tex`的`\bibliography{reference}`仍为注释状态，且其“尚未收录研究文献”说明在本轮新增条目后已过时。用户限制本轮不改主文件，所以完整保留；后续经授权编译前需启用入口并更新说明，本轮没有修改。
2. 当前主文件仍include六章，V2规划为七章；本轮不处理迁移，不影响只写1.1。题目、封面与备案状态不作推断。
3. 尚未检查PDF版面、最终引文显示和书目排版。第一作者整体字段及PMLR姓名形式的最终显示也要在获准编译后核对；静态完整性不等于排版已通过。
4. W11未使用，其IEEE/CVF页码差异记录保留在本交付中。正文无需为凑文献数量加入跨领域方法。
5. 用户需审核段落逻辑、语言自然程度、项目事实表述及研究意义定位。未经确认不编译、不冻结、不写1.2。

正文保留本项目数据关联和同步配对限制，不将其描述为普适不可达结论，也未以尚未验证的收益收束。
