# TeXcount实测

正文规划统计量：**1167**，目标1100–1250，中心1175。距最低+67，距最高余83。

|分项|数值|
|---|---:|
|正文及框图自然语言|1136|
|节标题|15|
|图题|16|
|合计|1167|
|行内数学（不计入文字合计）|30|
|显示公式环境（不计入文字合计）|2|
|图|1|

复用项目scripts/count_thesis_words.py的Counter和scripts/texcount_rules.tex，TeXcount3.1.1，中文UTF-8模式；独立片段未接入正式主文件，使用绝对output路径将全部输出限定在本目录。没有修改统计规则。

原始输出wordcount/raw/section_5_5.txt；实际命令wordcount/commands.json；逐字解析wordcount/wordcount_audit.html。图内文字已经计入text，不重复增加；公式符号、label/ref键及注释不计文字。无表格、脚注、程序清单或参考文献列表。

本节预算内，无需机械补至1175。
