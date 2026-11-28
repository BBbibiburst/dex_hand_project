# 实际TeXcount

运行命令：`./scripts/count_thesis_words.sh --files --output-dir thesis_audit/2026-11-28/annotation_R2_complete/wordcount`。真实原始输出、规则Hash与源文件Hash位于wordcount目录。

|章|实际字数|
|---|---:|
|1|8928|
|2|9494|
|3|9190|
|4|7510|
|5|9034|
|6|9667|
|7|2413|

实际接入正文合计：**56440**；包含活动前后置文本的源计数：58311。

标题、图表标题、脚注、表格正文计入；数学权重为0。文件汇总按当前include图；工具中旧V2规划标题匹配缺失第五章、摘要占位提示和71页旧PDF项属于继承的索引限制，不代表本次缺章。所有第五章片段均包含在actual_include_graph和实际合计中。当前PDF页数单独用PyMuPDF读取，不能用旧PDF字段。未修改计数规则。
