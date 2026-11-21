# Git提交范围检查

当前main跟踪origin/main，fetch后本地与上游0/0，无暂存用户改动，tracked diff为空；不相关未跟踪apps/tests/docs/progress/references及根档案保留。授权提交仅论文项目根、审计目录和统计脚本中的明确清单。

实际原始命令包括status --short --branch、diff --stat、diff、diff --cached、log -5 --oneline、remote -v、branch -vv；执行记录见本轮报告。选入路径/排除理由与敏感模式扫描见GIT_SCOPE_REVIEW.json。排除安装环境、下载文献全文/抽取正文、原始信号/权重/视频、临时构建与页渲染。正式论文PDF和冻结PDF允许归档；参考文献第三方全文不公开。封面身份为待填写，不含用户真实身份记录。

不使用add -A、force push、reset、clean、rebase、amend。提交后在回复中报告真实commit与远程HEAD核对结果；不在提交内预造push成功记录。

## 换行与历史空白

根.gitattributes为论文/审计路径增加-text规则，避免原CSV CRLF被Git规范化破坏既有SHA256；不影响运行时代码策略。本轮git diff --check及cached检查发现347项历史空白提示、涉及63个历史阅读稿/diff/上游模板文件；未修改这些文件，不声称检查无警告。新建六份管理文档和正式正文无这类提示。原始警告存于WHITESPACE_REVIEW.txt。

全文抽取JSON专项检查也排除了师兄论文reference_pages.json，保留本论文自身PDF文本核验。构建兼容配置latexmk.projectrc与wiseref许可已纳入；全部暂存文件通过原始字节一致性核对。
