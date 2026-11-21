# 论文LaTeX工程搭建验收报告

日期：2026-11-19。范围：仅工程、模板及规范核验；未写研究正文，未开展实验，未提交Git。

## 1. 运行平台与版本

- 平台：WSL2，Ubuntu 26.04 LTS，x86_64；内核6.18.58.1-microsoft-standard-WSL2。
- 经用户授权，项目本地安装TinyTeX v2026.11 / TeX Live 2026；XeTeX 0.999998，kpathsea6.4.2。
- Latexmk4.88（2026-03-09），BibTeX0.99e。
- 工具链路径：`.tools/.TinyTeX/bin/x86_64-linux/`。仅本项目构建进程添加PATH，没有sudo、没有修改系统TeX配置或Shell启动文件。
- LaTeX Workshop10.20.0已安装在当前WSL扩展主机。
- 另在`.tools/`放置本地antiword和PyMuPDF，仅用于读取学校附件及PDF核验。
- 工具目录约627MiB，上游仓库与Release约227MiB；详细安装日志及已安装包列表在`logs/`。

## 2. hithesis来源与使用方式

最新正式Release和Tag为[v3.2a](https://github.com/hithesis/hithesis/releases/tag/v3.2a)，发布2026-10-12；Tag commit `0640d2b67cf7a8d7af69a02c90b90dafa1f56040`。

完整原Git仓库保存在`upstream/hithesis/`，其master快照为`9f432a8690605ed7a90fd49629e1009cc5f0dd39`，未修改。构建实际使用官方`chinese-harbin-master-v3.2a.zip`，源于`examples/hitbook/chinese/`。

已读取README、Release/Tag、中文示例主文件、Makefile、latexmkrc、cover及前后置示例。Release含完整运行时文件，因此没有必要运行`latex hithesis.ins`或`make examples`；没有在原仓库中执行生成命令。

所有根目录模板核心文件、参考文献样式、图片及原latexmkrc均为逐字节复制，验证Hash未变化，见template_provenance.json。

## 3. 学校规范核验

已实际访问[研究生院2025-03-31通知](https://hitgs.hit.edu.cn/2025/0331/c17373a365618/page.htm)，成功下载并读取理工类学位论文指南、实践成果指南和选用表，原件、文本、XLSX单元格及Hash保存在`official/`。

电子信息0854在表中可选理工类论文或实践成果指南，但该行授权层级为博士；实践成果指南另明确涉及电子信息硕士。不能据此替代学院对本人的具体要求。按用户选择搭建学位论文工程，学位类别、培养学院和封面细节仍待确认，见school_requirements.md。没有声称已完成学院最新文件或备案题目核验。

## 4. 工程结构与内容

主文件`thesis.tex`只组织封面、摘要、目录和六章。六个body文件只有用户指定标题、label及注释，没有完整正文。个人信息、中文/英文摘要、关键词与英文题目均为占位符。暂定中文题目采用用户给出的题名，不代替学校备案题目确认。

`front/`存封面和摘要，`back/`预留致谢、成果及附录（默认未启用），`figures/`分framework/network/experiments，另有tables、tests、scripts和独立.vscode工作区。具体列表见project_files.txt，完整新文件列表见full_file_inventory.txt。

研究参考库reference.bib暂为空注释，未放入虚构作者、文章或DOI。引用测试唯一条目是已实际下载的hithesis官方仓库，使用该bst支持的`@online`及`citedate`字段，仅在tests中引用。

## 5. VSCode与构建配置

打开`thesis.code-workspace`或单独打开thesis文件夹，运行LaTeX Workshop Build（Ctrl+Alt+B），保存后自动构建；PDF使用内部标签页预览并刷新。不能只在外层工程打开单个tex文件，指望内层.vscode自动生效。

构建脚本复用原latexmkrc，其pdf模式实际调用XeLaTeX，参考文献为BibTeX/hithesis.bst。源文件和build输出分离，完整控制台日志按时间戳留在logs。自动清理关闭。

上游索引函数在新版latexmk自动切换到build目录后，仍使用带build前缀的源路径，触发重复路径错误；空白骨架同时没有索引条目。项目新增`latexmk.projectrc`，仅修正索引命令参数路径、空索引输出以及移除重复索引规则。原latexmkrc、类、cfg和sty都没有修改。

命令行已按LaTeX Workshop实际参数形式（绝对主文件路径、不带.tex后缀）成功运行。SyncTeX程序路径显式指向项目本地工具，搬迁工程后需更新此设置。

## 6. PDF及引用测试

- `build/thesis.pdf`：12页，空白论文骨架编译成功，目录和六章标题存在。
- `build/compile_smoke.pdf`：5页，独立工程测试编译成功。
- 两份PDF的中文可直接提取，`灵巧手`搜索返回结果，Fandol中文字体已嵌入。
- 主PDF没有上游示例姓名、题目等残留；封面与参考文献页已渲染查看。
- 测试中的图、表、式均有1-1编号和aux标签，引用已解析，最终日志没有未定义引用。
- BibTeX读取hithesis.bst并生成实际条目，官方URL可提取，未使用未经核实的研究文献。
- 两份SyncTeX数据已生成，命令行正向及反向搜索均实际通过，见synctex_check.json和对应日志。
- VSCode GUI的实际自动刷新、双击反向跳转及快捷键体验尚需在用户界面确认，未冒称已做GUI点击验收。

机器可读结果见verification.json。关键完整日志为`logs/main_build_final.log`、`logs/smoke_build_final.log`及build中的tex/blg日志。早期依赖缺失与索引失败记录均保留，未以早期生成PDF冒充完整构建通过。

## 7. 尚未解决或待确认

1. 学院、精确专业/学位类别、备案题名及个人信息待提供，封面还不能正式提交。
2. 未认证学院在已下载2025指南之后是否另有更新通知；最终送审应以学院与研究生院最新要求确认。
3. 非致命模板警告仍存在：RSFS数学字体在模板规定的特殊字号上进行尺寸替代；当前未使用术语表，故有未打印术语表提示。没有中文缺字、未定义引用或阻止构建的错误；没有为消除警告修改模板核心或字体配置。
4. GUI交互需用户确认；字体正式送审要求也需结合学院确认。当前按用户要求使用Fandol。
5. 研究正文、研究参考文献和真实图表均未开始写入，后续需逐项核对P0–P51来源。

## 8. 新建与修改文件

全部工程新增内容都在thesis/内；原实验和原项目VSCode配置未修改。

- project_files.txt：逐个列出工程源文件、配置、原样复制的模板文件及文档。
- full_file_inventory.txt：包含工具、上游、日志和构建产物的完整实际文件路径列表。
- project_integrity.json：项目源文件SHA256及模板/上游未修改检查。
- 本轮另安装了WSL端LaTeX Workshop扩展，属于用户授权的编辑器配置，不属于仓库文件。

Git HEAD保持`ff7b1a1`；没有git add、commit或push。没有模型训练、采集、原始实验数据或Checkpoint修改。
