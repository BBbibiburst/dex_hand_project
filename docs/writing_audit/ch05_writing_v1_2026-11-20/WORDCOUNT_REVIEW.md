# 5.3实际TeXcount

**实测1830，符合1800–2000目标。** 统计工具为项目原有TeXcount 3.1.1中文模式；复用`scripts/count_thesis_words.py::Counter`及原规则，没有用正则估字数，没有改统计规则。

- 正文与表格、流程框文字：1763。
- 节标题：17。
- 图表标题：50。
- 合计：1830。
- 行内数学：38；独立公式：3，均不折为正文统计量。

两图一表的文字已由原引擎独立检查，分项见wordcount/float_counts.json；完整解析着色见wordcount/wordcount_audit.html，原始输出见wordcount/raw/section_5_3.txt。

命令：`PYTHONDONTWRITEBYTECODE=1 python3 docs/writing_audit/ch05_writing_v1_2026-11-20/count_section.py`。

现有全章CLI仅接受七章规划路径，不支持指定节片段；本次只调用其同一个Counter计算独立section_5_3.tex，没有创建虚假5.1/5.2或修改主文件。文件未接入正式编译树，不将该值称为第五章全章字数；PDF页数UNKNOWN，未编译。
