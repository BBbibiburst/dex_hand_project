# 实际TeXcount统计

工具：项目`scripts/count_thesis_words.py`的Counter，TeXcount 3.1.1（2018-10-28），`-utf8 -ch`，同一`scripts/texcount_rules.tex`；不改全局规则。因本节独立且未接入主文件，直接调用既有Counter统计此片段，输出绝对路径限定在本交付目录。

|分项|实测|
|---|---:|
|正文及图内可见文字|1359|
|节标题|17|
|图题|16|
|正文规划总统计量|**1392**|
|行内数学数量（不加文字总数）|29|
|显示数学环境数量（不加文字总数）|3|
|图数量|1|

预算1350–1550，实测1392，距最低+42、距最高余158。无需为中心1450机械扩写。公式内变量、cite/ref/label键、注释不计普通正文。框图中的自然语言在详细HTML中按word标记计入正文，图题另计；没有重复把图内文字加一次。无表格、脚注或程序清单。

原始输出：wordcount/raw/section_5_4.txt；详细解析：wordcount/wordcount_audit.html；实际命令：wordcount/commands.json。

首次调用Counter使用相对输出路径，因Counter以thesis为cwd导致找不到生成文件；随后改用绝对路径，最终两种输出均成功。未改统计规则。
