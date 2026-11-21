# 5.3 R3最终TeXcount

R2：1999；R3：**1999**。正文/流程框/表格文字1932，节标题17，图表题50；行内数学42及独立公式3另报。仍使用项目原Counter、TeXcount中文模式及原规则。

正文及所有技术数据与R2逐字一致，只加入局部框内底部2.5mm留白、将框内额外行距4pt调为2pt，以及跨页位置的mbox。没有改全局字号、行距、页边距或统计规则。

命令：`PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r3/count_section.py`。原始输出及解析HTML在wordcount/。统计不包含临时第五章标题，不是第五章整章结果。
