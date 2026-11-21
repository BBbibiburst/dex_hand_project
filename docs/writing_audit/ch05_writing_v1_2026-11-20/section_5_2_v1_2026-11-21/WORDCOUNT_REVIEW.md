# 5.2 V1实际TeXcount

工具：项目scripts/count_thesis_words.py的Counter直接调用既有TeXcount与scripts/texcount_rules.tex，未运行会改写篇幅预算的全工程main入口。版本：TeXcount version 3.1.1, 2018 Oct 28.。

|统计项|实际|
|---|---:|
|正文|964|
|小节标题|17|
|图表题/脚注|0|
|验收合计|**981**|
|目标|900–1100，中心1000|
|距中心|−19|
|行内数学片段|6|
|独立公式环境|2|
|图表/代码块|0/0|

符合预算。公式数量单列、不按符号凑字数。无表格、脚注、参考文献正文。四个自然段不含标题；公式嵌在第3、4段中，公式后无空行另起段。

实际命令参数：perl thesis/.tools/texcount/texcount.pl -utf8 -ch -sum=1,1,1,0,0,0,0 -sub=subsection <独立generated片段>。生成片段仅前置现有统计规则，不接入主文件。完整原始输出见wordcount/raw/section_5_2.txt；结果见wordcount/metrics.json；着色解析见wordcount/wordcount_audit.html。无解析警告。TeXcount是规则解析，不代表完成LaTeX编译或PDF排版检查。
