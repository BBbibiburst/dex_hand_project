# 硕士论文初稿工程

题目：灵巧手仿真示教数据构建与触觉跨域重建方法研究。

截至2026-11-27，第一至第七章均已接入 `thesis.tex`，完整PDF为93页。本次为初稿提交，不代表全篇正式定稿或研究验收通过。

## 最新交付

- `releases/draft_2026-11-27/thesis_draft.pdf`：完整初稿。
- `releases/draft_2026-11-27/thesis_source.zip`：完整可编辑TeX源及图片、模板、文献。
- `releases/draft_2026-11-27/WORDCOUNT.md`：七章实际字数。
- `releases/draft_2026-11-27/SHA256SUMS`：校验清单。
- `THESIS_PERSONAL_INFORMATION.csv`：个人信息权威表；Markdown版本仅为历史记录。

## 编译与组织

主入口 `thesis.tex`；正文在 `body/`，第五至七章由聚合文件引用子节；封面与摘要在 `front/`，致谢在 `back/`。模板为hithesis，使用XeLaTeX与BibTeX，需安装Fandol字体及模板依赖。源码包提供 `build_draft.sh`，在解压后的thesis目录执行。现有工程仍可使用 `scripts/build.sh`。

源码包不包含本机TinyTeX安装、编译缓存、实验数组、模型权重、服务器采集数据或第三方参考论文。Git保留既有历史骨架及冻结版本，当前版本以主文件实际引用为准。

## 当前边界

第六章仍为未冻结初稿；不以本次Git提交代替章节审核。C27联合验收2/11；F0指定SWD相对下降49.77%不等于完整迁移成功。学位声明签名及签署日期留空，致谢需本人最终审阅。封面分类号按用户明确要求沿用参考论文中文封面。

已有正文和文献编译警告见交付日志；本次未运行研究实验，未读取Test/Holdout。论文修改与实验生产代码分别管理。
