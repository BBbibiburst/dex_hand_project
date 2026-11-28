# #02 字体方案：先提交依据，不自动替换全篇字体

学校域名提供的《研究生学位论文写作指南（理工类）》第3.1节规定：中文正文用宋体，标题用黑体，数字及英文用Times New Roman。下载原件、Hash与来源在literature/sources.json；该文件不能单独证明学院2026年是否另有更新要求。

当前入口使用Fandol，模板将西文设为TeXGyreTermesX；二者不能直接称为微软SimSun和Times New Roman。本机fontconfig对SimSun和Times New Roman均回退为DejaVu Serif，说明当前没有通过这些名称找到对应字体。方程继续由现有数学字体配置排版，未以正文换字体顺带改动数学字体。

方案A：本轮保持现有模板与字体，以保证已审核版面可比较；新插图使用随现有TeX工具链提供的Fandol宋体。FONT_COMPARISON.pdf对照Fandol与Noto Serif CJK，只作开放字体外观比较，不假装其中任何一栏是Windows宋体。

方案B：取得有合法使用权限的SimSun/SimHei/楷体及Times New Roman字体，并确认学院要求后，在独立副本中设置字体并编译全篇，对照目录、公式、表格宽度、换行及页数，再决定迁入主源。不能仅安装字体后假定模板已经切换，也不能把Linux回退字体当作所请求字体。字体文件本身不纳入公开源码包。

Fandol的CTAN许可标为GPL-3附字体例外；TeX Gyre页面列GUST Font License。此处记录公开包许可，不推断用户是否有权复制某台Windows设备的字体。按任务书“先给出字体方案及依据，再执行”，本项保留为部分处理，尚未全篇替换，也不声称字体要求已最终认证。

来源：[学校域名指南](https://iac.hit.edu.cn/_upload/article/files/6f/7e/af9689084cb1bc8544f21b60ee59/7a1727a1-3cab-45ed-994b-cf113386e0ed.pdf)、[Fandol许可](https://ctan.org/pkg/fandol)、[TeX Gyre许可](https://ctan.org/pkg/tex-gyre)。
