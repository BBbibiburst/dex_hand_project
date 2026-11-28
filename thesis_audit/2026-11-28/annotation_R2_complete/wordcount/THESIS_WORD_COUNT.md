# 论文自动字数统计

2026-11-28T23:28:51.078072+08:00

TeXcount version 3.1.1, 2018 Oct 28.

TeXcount -utf8 -ch; sum=1,1,1,0,0,0,0; include text+headings+captions/footnotes/table cells; no math weight

**规划目标，不是学校官方要求。MISSING不等于实测0字；完成率是篇幅进度，不是研究完成度。**

## 当前真实结果

- 当前六章编译源正文合计：56440（text 54407，标题 555，图表题/脚注 1478）。
- 七章规划可匹配已测小计：47373；缺章：[5]。其余缺失不作伪测量；按已匹配文字计，中心目标进度 89.38%，距49500尚差 2127。
- 正文行内公式 451，独立公式 32；代码 0块/0物理行，均不计目标词数。
- 当前活动前置源计数：1560（摘要占位；宏生成封面未展开）；完整主文件静态计数 58311，不与正文目标混用。
- 当前PDF页数：UNKNOWN；已有PDF：STALE，旧产物页数 71。未重新编译，不能将旧PDF页数配给本轮正文。
- 参考文献列表：UNKNOWN_REQUIRES_CURRENT_BBL；已收录BibTeX条目 34，不作为正文词数。

## 七章进度

|章|题名|最低–最高|中心|当前|完成率（中心）|距最低缺口|状态|
|---|---|---:|---:|---:|---:|---:|---|
|1|绪论|7500–8500|8000|8928|111.60%|0|PRESENT|
|2|RM75B-Dex Hand多任务仿真平台与触觉感知建模|8000–9000|8500|9556|112.42%|0|PRESENT|
|3|面向欠驱动灵巧手的分级抓取生成与专家示教构建|8000–9000|8500|9233|108.62%|0|PRESENT|
|4|多源示教数据构建与质量控制|7000–8000|7500|7537|100.49%|0|PRESENT|
|5|基于统计伪配对与双域表示的触觉跨域重建方法|8000–9000|8500|MISSING|UNKNOWN|UNKNOWN|MISSING|
|6|基于迭代对抗优化的触觉迁移与综合评价|9000–10500|9750|9706|99.55%|0|PRESENT|
|7|总结与展望|2000–2500|2250|2413|107.24%|0|PRESENT|

## 42节预算与实际

|节|标题|最低–最高|当前|中心完成率|距最低缺口|状态|
|---|---|---:|---:|---:|---:|---|
|1.1|研究背景与意义|1800–2000|1941|102.16%|0|PRESENT|
|1.2|国内外研究现状|3500–4000|4825|128.67%|0|PRESENT|
|1.3|研究问题与技术难点|1000–1100|1015|96.67%|0|PRESENT|
|1.4|主要研究内容与创新点|800–950|918|104.91%|0|PRESENT|
|1.5|论文组织结构|400–450|227|53.41%|173|PRESENT|
|2.1|系统需求与总体架构|700–800|874|116.53%|0|PRESENT|
|2.2|RM75B机械臂与Dex Hand灵巧手模型|1300–1450|MISSING|UNKNOWN|UNKNOWN|MISSING_SECTION|
|2.3|灵巧手接触建模与控制方法|1300–1450|1565|113.82%|0|PRESENT|
|2.4|复杂曲面触觉阵列建模|1700–1950|2523|138.25%|0|PRESENT|
|2.5|712维触觉观测及处理流程|1000–1100|1109|105.62%|0|PRESENT|
|2.6|仿真平台功能验证|1600–1800|1700|100.00%|0|PRESENT|
|2.7|本章小结|400–450|416|97.88%|0|PRESENT|
|3.1|抓取任务定义与总体流程|600–700|717|110.31%|0|PRESENT|
|3.2|多来源抓取候选生成|1600–1800|1798|105.76%|0|PRESENT|
|3.3|基于可达性约束的抓取筛选|1300–1450|1423|103.49%|0|PRESENT|
|3.4|连续逆运动学与轨迹验证|1500–1650|2053|130.35%|0|PRESENT|
|3.5|多任务专家示教生成|1400–1550|1472|99.80%|0|PRESENT|
|3.6|抓取及示教质量实验|1200–1400|1344|103.38%|0|PRESENT|
|3.7|本章小结|400–450|405|95.29%|0|PRESENT|
|4.1|多模态数据表示与组织方式|1000–1100|1098|104.57%|0|PRESENT|
|4.2|示教数据采集与关联机制|1300–1500|1411|100.79%|0|PRESENT|
|4.3|数据格式转换与统一存储|900–1000|881|92.74%|19|PRESENT|
|4.4|轨迹质量审计与数据清洗|1500–1700|1717|107.31%|0|PRESENT|
|4.5|数据集统计与质量验证|1900–2250|1960|94.46%|0|PRESENT|
|4.6|本章小结|400–450|457|107.53%|0|PRESENT|
|5.1|真实与仿真触觉差异分析|1100–1250|MISSING|UNKNOWN|UNKNOWN|MISSING_FILE|
|5.2|非配对条件下的迁移问题定义|800–900|MISSING|UNKNOWN|UNKNOWN|MISSING_FILE|
|5.3|Real统计先验引导的仿真触觉探针与伪配对|1700–1900|MISSING|UNKNOWN|UNKNOWN|MISSING_FILE|
|5.4|基于Dual-VAE的双域触觉表示|1600–1750|MISSING|UNKNOWN|UNKNOWN|MISSING_FILE|
|5.5|基于潜变量Translator的初始跨域重建|1200–1350|MISSING|UNKNOWN|UNKNOWN|MISSING_FILE|
|5.6|初始迁移实验及局限分析|1200–1400|MISSING|UNKNOWN|UNKNOWN|MISSING_FILE|
|5.7|本章小结|400–450|MISSING|UNKNOWN|UNKNOWN|MISSING_FILE|
|6.1|共享参数向量场迁移方法|1500–1750|1893|116.49%|0|PRESENT|
|6.2|判别器设计与梯度优化机制|1400–1600|1508|100.53%|0|PRESENT|
|6.3|多轮对抗优化实验|1500–1750|1917|117.97%|0|PRESENT|
|6.4|多视角判别与残差重建探索|1200–1400|1441|110.85%|0|PRESENT|
|6.5|触觉分布质量与输入保持评价|1400–1650|1723|112.98%|0|PRESENT|
|6.6|判别器可靠性及跨Session泛化分析|1500–1750|1206|74.22%|294|PRESENT|
|6.7|综合讨论与本章小结|500–600|MISSING|UNKNOWN|UNKNOWN|MISSING_SECTION|
|7.1|主要研究工作总结|900–1100|990|99.00%|0|PRESENT|
|7.2|主要研究结论|600–750|671|99.41%|0|PRESENT|
|7.3|不足与未来展望|500–650|747|129.91%|0|PRESENT|

## 六章实际include文件（不套用七章预算）

|文件|实际章标题|统计量|正文/标题/其他|内容状态|
|---|---|---:|---|---|
|thesis/body/ch01_introduction.tex|绪论|8928|8827/86/15|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch02_simulation_platform.tex|机器人多任务仿真平台与触觉感知建模|9494|9028/87/379|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch03_grasp_demonstrations.tex|面向灵巧手操作的抓取候选生成与专家示教构建|9190|8836/86/268|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch04_multimodal_data.tex|多源示教数据构建与质量控制|7510|7258/72/180|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch05_dual_domain_reconstruction.tex|基于统计伪配对与双域表示的触觉跨域重建方法|21|0/21/0|SKELETON_ONLY|
|thesis/body/ch05_sections/section_5_1.tex|None|1261|1234/11/16|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch05_sections/section_5_2.tex|None|1114|1097/17/0|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch05_sections/section_5_3.tex|None|2129|2046/17/66|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch05_sections/section_5_4.tex|None|1437|1404/17/16|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch05_sections/section_5_5.tex|None|1424|1377/15/32|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch05_sections/section_5_6.tex|None|1648|1599/13/36|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch06_adversarial_evaluation.tex|基于迭代对抗优化的触觉迁移与综合评价|18|0/18/0|SKELETON_ONLY|
|thesis/body/ch06_sections/section_6_1_shared_vector_field.tex|None|1884|1784/11/89|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex|None|1503|1415/12/76|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex|None|1902|1757/8/137|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex|None|1431|1321/12/98|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch06_sections/section_6_5_distribution_quality_and_input_preservation.tex|None|1723|1679/13/31|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex|None|1206|1154/13/39|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch07_conclusion.tex|总结与展望|5|0/5/0|SKELETON_ONLY|
|thesis/body/ch07_sections/section_7_1_research_work_summary.tex|None|990|982/8/0|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch07_sections/section_7_2_main_research_conclusions.tex|None|671|665/6/0|DRAFT_OR_PLACEHOLDER|
|thesis/body/ch07_sections/section_7_3_limitations_and_future_work.tex|None|747|740/7/0|DRAFT_OR_PLACEHOLDER|

## 第一章细分（TeXcount实际subcounts）

|层级与标题|正文|标题|图表题/脚注|合计|
|---|---:|---:|---:|---:|
|Chapter: 绪论|0|2|0|2|
|Section: 研究背景与意义|1934|7|0|1941|
|Section: 国内外研究现状|0|7|0|7|
|Subsection: 机器人仿真与触觉感知建模|786|12|0|798|
|Subsection: 灵巧手抓取生成与可执行操作|863|13|0|876|
|Subsection: 专家示教与多模态操作数据|1288|12|0|1300|
|Subsection: 触觉Sim-to-Real与跨域重建|1821|8|15|1844|
|Section: 研究问题与技术难点|1006|9|0|1015|
|Section: 主要研究内容与创新点|908|10|0|918|
|Section: 论文组织结构|221|6|0|227|

## 前后置库存（与七章正文分开）

|文件|是否启用|统计量|
|---|---|---:|
|thesis/front/abstract.tex|ACTIVE|1381|
|thesis/front/cover.tex|ACTIVE|179|
|thesis/front/denotation.tex|INACTIVE|0|
|thesis/back/acknowledgements.tex|ACTIVE|311|
|thesis/back/appendix.tex|INACTIVE|7|
|thesis/back/publications.tex|INACTIVE|33|

封面hitsetup是元数据而非直接排版文字，按规则忽略，0不表示最终封面无文字。摘要/致谢等当前为占位符，记录其真实计数但不当成已完成内容。未启用的附录、符号表和成果目录只作库存。

## 结构差异与待写内容

V2逐章路径和状态由JSON自动读取。当前缺失的规划章节为：[5]。旧骨架各章实际标题与计数见上表，不自动改名映射为新章节。脚本没有创建章节或修改include。

第四章有条件保留：先补来源、lineage、采样/动作关联、审计与统计证据；其他缺口按THESIS_LENGTH_BUDGET.md和原规划论证卡处理，不自动填正文。

## 核验与限制

- tests.json保存已知答案的中文、英文、标题、表格、脚注、公式、注释、引用、代码及input/include测试。
- wordcount_audit.html为真实TeXcount着色解析；raw/为原输出，generated/为统计副本。
- 官方规则文档：https://tug.ctan.org/support/texcount/doc/TeXcount.pdf
- 静态文件名去重不等价任意TeX程序执行；动态输入/自定义宏需规则与样例。
- 只计数，不编译、不训练、不修改正式正文、文献或模板。
- include解析警告：['thesis/body/ch02_simulation_platform.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.96\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.98\\textwidth,height=0.80\\textheight,keepaspectratio]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.66\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.96\\textwidth]', 'thesis/body/ch02_simulation_platform.tex: dynamic input graphics[width=0.92\\textwidth,height=0.83\\textheight,keepaspectratio]', 'thesis/body/ch03_grasp_demonstrations.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch03_grasp_demonstrations.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch03_grasp_demonstrations.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch03_grasp_demonstrations.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch03_grasp_demonstrations.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch03_grasp_demonstrations.tex: dynamic input graphics[width=0.96\\textwidth]', 'thesis/body/ch03_grasp_demonstrations.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch03_grasp_demonstrations.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch04_multimodal_data.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch04_multimodal_data.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch04_multimodal_data.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch04_multimodal_data.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch04_multimodal_data.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch04_multimodal_data.tex: dynamic input graphics[width=0.78\\textwidth]', 'thesis/body/ch05_sections/section_5_3.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch05_sections/section_5_3.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch05_sections/section_5_3.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch05_sections/section_5_4.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch05_sections/section_5_4.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch05_sections/section_5_5.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch05_sections/section_5_5.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch05_sections/section_5_6.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch05_sections/section_5_6.tex: dynamic input graphics[width=0.97\\textwidth]', 'thesis/body/ch06_sections/section_6_1_shared_vector_field.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch06_sections/section_6_1_shared_vector_field.tex: dynamic input graphics[width=0.98\\textwidth,trim=0', 'thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex: dynamic input graphics[width=\\textwidth]', 'thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex: dynamic input graphics[width=0.90\\textwidth]', 'thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex: dynamic input graphics[width=\\textwidth]', 'thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex: dynamic input graphics[width=\\textwidth]', 'thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex: unsupported/dynamic include or includeonly: inspect audit', 'thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex: dynamic input graphics[width=\\textwidth]', 'thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex: dynamic input graphics[width=\\textwidth]']

操作：项目根目录执行 `./scripts/count_thesis_words.sh --all`。按V2规划章：`--chapter 1`；按节：`--sections`；文件：`--files`；解析：`--audit`。
