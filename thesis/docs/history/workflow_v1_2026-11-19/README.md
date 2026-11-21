# 哈工大哈尔滨校区中文硕士论文工程

本目录只建立论文骨架和编译环境，没有撰写研究正文。暂定题目由用户提供，不代表已核实学校备案题目。

## 在VSCode开始使用

1. 在WSL侧打开 `thesis/thesis.code-workspace`（文件 → 从文件打开工作区），或单独打开整个 `thesis/` 文件夹。**只在外层项目打开某个tex文件，不会自动应用内层.vscode配置。**
2. 打开 `thesis.tex`，运行LaTeX Workshop的“Build LaTeX project”（默认Ctrl+Alt+B）。保存源文件也会触发构建。
3. PDF在 `build/thesis.pdf`；运行“View LaTeX PDF file”以标签页预览。成功构建后预览自动刷新。
4. 正向搜索使用LaTeX Workshop SyncTeX命令；PDF中双击执行反向搜索。已生成SyncTeX数据，但GUI快捷键操作仍需在你实际VSCode窗口确认。

LaTeX Workshop已安装到当前WSL扩展主机。工作区不覆盖原项目的Python设置。全局PATH没有修改；构建脚本只对子进程加入项目本地TeX路径。SyncTeX程序路径在本工作区中显式配置，搬动整个工程后需更新该路径。

## 命令行编译

从项目根目录：

```bash
./thesis/scripts/build.sh
./thesis/scripts/build.sh tests/compile_smoke.tex
```

或在本目录运行 `make thesis` / `make smoke`。依赖修复后若latexmk记住了此前失败状态，可运行：

```bash
THESIS_FORCE_REBUILD=1 ./thesis/scripts/build.sh
```

主论文输出与独立测试输出分别为 `build/thesis.pdf`、`build/compile_smoke.pdf`。控制台完整日志保存在带时间戳的 `logs/`，TeX/BibTeX日志及辅助文件保存在 `build/`；不会自动清理历史日志。

## 模板来源

官方仓库：https://github.com/hithesis/hithesis 。使用最新正式Release `v3.2a`（2026-10-12发布）的 `chinese-harbin-master`，对应 `examples/hitbook/chinese/`。

- `upstream/hithesis/`：完整原Git仓库，未在其中执行生成或修改。
- `upstream/chinese-harbin-master-v3.2a.zip` 及 `upstream/release-v3.2a/`：官方Release原件。
- 根目录 `.cls/.cfg/.sty/.bst/.ist/.eps` 与 `latexmkrc`：从Release逐字节复制，Hash见 `docs/template_provenance.json`。
- `docs/thesis.upstream.tex`、`docs/Makefile.upstream`：原示例及构建方式备查。

Release已包含生成的运行时文件，所以不需要再运行 `latex hithesis.ins` 或 `make examples`。若日后从纯源码重建，须在另一个构建副本中按该版本README/Makefile执行，不能改动保留的上游副本。

文档类固定为 `fontset=fandol,type=master,campus=harbin`，模板原有 `\usepackage{hithesis}` 保留。不自行添加ctex、改版芯或改核心文件。构建原样加载 `latexmkrc`，再加载项目级 `latexmk.projectrc`，后者仅修复输出目录切换/空索引兼容性，不改排版和BibTeX。构建复用原 `latexmkrc`：其pdf模式命令实际为XeLaTeX，文献使用BibTeX与hithesis.bst。

## 文件组织

```text
thesis.tex                    主文件，仅组织章节
reference.bib                 研究文献库，当前只有说明注释
front/cover.tex               待填写的封面信息
front/abstract.tex            中英文摘要占位
front/denotation.tex          符号表保留入口
body/ch01_introduction.tex    绪论
body/ch02_data_and_problem.tex 数据与问题定义
body/ch03_dual_vae.tex         双VAE与初始迁移
body/ch04_adversarial_transfer.tex 迭代对抗迁移
body/ch05_experiments.tex     实验结果与分析
body/ch06_conclusion.tex      总结与展望
back/                         附录、成果、致谢占位，默认不启用
figures/{framework,network,experiments}/
tables/
.vscode/                      独立工作区构建与预览配置
scripts/build.sh              项目本地工具链入口
scripts/verify_setup.py        编译产物与中文/引用检查
scripts/                       其他工程核验脚本
tests/compile_smoke.tex       独立工程测试，不属于论文正文
build/                        PDF与辅助文件
logs/                         完整构建/安装日志
.tools/                       授权安装的本地TeX与PDF核验工具
upstream/                     未修改的上游仓库与Release
docs/                        版本、学校规范、验收及文件清单
```

## 待本人/学院确认的信息

姓名、学号、导师及职称、培养学院、精确专业代码、学校封面学位名称、正式备案题目/英文题目、封面日期均待填写。未从旧实验、路径用户名或上游示例推断个人信息。

[学校规范核验](docs/school_requirements.md)记录了三份已下载官方附件。电子信息选用表中授权层级为博士，不能据此擅自确定电子信息硕士的学院要求；最终送审须向学院/导师确认。

## 文献和编译测试

主论文当前没有研究文献引用，因此 `\bibliography{reference}` 暂注释，避免空库BibTeX报错。录入核实条目并加入正文引用后启用；不要用占位文献凑数。

BibTeX、目录、图表及交叉引用测试独立位于 `tests/compile_smoke.tex`。其中唯一引用是实际下载的hithesis官方仓库，没有虚构论文、作者或DOI。测试图表明确标为工程占位，不能移作研究结果。

## 研究内容保护

后续只使用可追溯P0–P51材料。不得将开发Validation称为新盲测，不得把FR100%写成完整迁移成功，不编造实机实验或文献，不修改原始数据、模型及历史指标，不删除失败对照。将来引入正式图表时应先核验来源、配置和CSV，不直接引用临时截图。

本阶段没有Git提交，没有启动实验。详细验收、工具版本及仍未完成的GUI确认见 `docs/setup_report.md`。

## 论文写作协作入口（2026-11-19）

当前按用户明确的专业学位硕士研究定位规划，学院及封面行政字段仍待确认。正式写作前读取 [AGENTS.md](AGENTS.md) 及五份档案：

- [主参考论文分析](REFERENCE_THESIS_ANALYSIS.md)
- [论文写作规范](THESIS_WRITING_STANDARD.md)
- [六章研究规划](THESIS_PLAN.md)
- [实验依据档案](EXPERIMENT_EVIDENCE.md)
- [符号与术语](SYMBOLS_AND_TERMS.md)

原始证据索引与核验记录保存在 `docs/writing_audit/`。索引不等于全部实验重新验证；未核实内容明确保留缺口。本阶段不写正式正文，等待用户审核后逐节推进。P51为NO-GO，不启动P52/P53。
