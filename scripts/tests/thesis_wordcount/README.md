# 统计规则测试（不编译论文）

运行：`./scripts/count_thesis_words.sh --test`。这是TeXcount集成验证，不是机器人/训练实验。

- `basic.tex`：已知答案18。正文“中文 Alpha beta”为4；章题2、节题2；脚注“脚注 Footnote”为3；表题2、表格“甲乙 Gamma”为3；图题2。合计18。另有2个图表、1个行内公式、1个独立公式、1个代码块。注释、cite/label/ref键、图片路径、公式内容、代码和参考文献正文不计。
- `includes.tex`：主文2、child的“中文 Alpha”3、second的“触觉”2，唯一源文件计7。child重复引用只计一次；原生TeXcount `-merge`得10，保留对照日志，说明项目为何额外进行规范路径去重。
- `custom.tex`：局部TC规则使`tactileterm`参数计入，预期3，不模拟任意宏展开。
- `extended.tex`：longtable题2、multicolumn文字2、multirow文字2、Epsilon 1，共7；代码2块、表格1。hitsetup与TC忽略区不计入。
- 第五项是原生TeXcount include/merge的实际运行验证，结果与项目去重口径分开。

`protected_before.json`是THESIS-LENGTH-01开始时的只读保护快照，用于本轮最终核对；统计工具每次运行另取新的保护快照，不会阻止后续用户授权写作。

这些样例验证已用规则，不构成对所有LaTeX宏的精确性保证。动态include、复杂条件编译、宏生成正文等需单独补例。
