# 5.3 R2实际TeXcount

R1：1998；R2：**1999**，不超过2000。复用项目原TeXcount 3.1.1中文模式、Counter及原规则。正文/流程框/表格文字1932，标题17，图表题50；行内数学42、独立公式3另报、不折算正文。

统计对象是section_5_3.tex，不包含临时预览用的第五章标题。未删减公式、表格/图示内容或指标定义；仅在局部补明审核要求，并压缩同段重复的来源/独立性表述。

命令：`PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r2/count_section.py`。

原始输出、解析HTML和分项见wordcount/。正式主文件include未改，不将本节统计作为第五章整章统计。
