# 5.3 R1实际TeXcount

原稿：1830；修订稿：**1998**，符合1800–2000目标。增加内容限于坐标原点、混合统计、误差分母/空帧、指纹及提前停止的技术澄清，没有新增实验结果或额外小节。

|项目|实测|
|---|---:|
|正文/流程框/表格文字|1931|
|标题|17|
|图表题|50|
|合计|1998|
|行内数学（不折算正文）|42|
|独立公式（不折算正文）|3|

使用现有scripts/count_thesis_words.py::Counter、项目原TeXcount中文模式和原texcount_rules.tex，没有更改统计规则。节片段不在正式include树，故调用同一Counter统计独立文件，而非伪造第五章整章结果。

命令：`PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/revision_r1/count_section.py`。

原始输出在wordcount/raw/section_5_3.txt；完整解析在wordcount/wordcount_audit.html；分项在wordcount/result.json及float_counts.json。未编译，PDF页数UNKNOWN。
