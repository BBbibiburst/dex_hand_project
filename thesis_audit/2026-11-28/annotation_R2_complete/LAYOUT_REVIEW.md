# PDF与版式核对

最终PDF **106页**；对照批注第二批98页，增加8页。新增内容主要为实际采样中间图、候选几何/历史任务阶段及必要论述，不是新增实验。

全106页自动检查页面文字边界：无文字坐标超出版面；无低于40字符的空页候选。逐张查看7张全页缩略图，放大查看新增图所在PDF页22、24、31、41、43、45、47、52、54、57。检查范围是布局、图形和主要标注，不等于106页逐字人工校对。page_review保存缩略图与放大图；pdf_page_checks.json保留逐页数据。

图2-3为整页浮动图，固定腱系数和名称可辨；采样图三行分别展示指节、指尖、手掌。图3-6为20帧历史阶段画面，每行任务独立。图4-1两个来源路径分开，索引137/136偏移保持。候选图坐标刻度稀疏化、提高实体对比度，未改变几何数据。全页概览未见图文重叠、孤立标题或整页空白。

中文摘要仍有续页（PDF第5页约数行），英文摘要亦为两页；不是本轮通过改字号压缩处理，列为后续排版优化。部分复杂图的小字号仍需用户在实际阅读尺寸确认。本轮不更换全局字体，不将人工体验等同于自动验收。

最终XeLaTeX日志未出现Overfull、未定义引用、缺字或浮动体过大。保留的模板警告如下：

- LaTeX Font Warning: Font shape `U/rsfs/m/n' in size <10.53937> not available
- LaTeX Font Warning: Font shape `U/rsfs/m/n' in size <7.52812> not available
- Package glossaries Warning: No \printglossary or \printglossaries found. 
- LaTeX Font Warning: Size substitutions with differences

完整构建日志见build.log，最后一遍TeX详细日志见build/thesis.log。
