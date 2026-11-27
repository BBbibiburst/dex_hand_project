# 独立编译与保护检查

使用现有TinyTeX xelatex×4、bibtex×1；入口为未修改的thesis/thesis.tex，所有辅助文件在stage_b2/build。完整控制台日志build.log，末轮TeX日志build/thesis.log、文献日志build/thesis.blg。

PDF：thesis_stage_b2_preview.pdf，93页；B1为93页，变化0。逐页文本差异页[]，完全空白页[]。没有未定义引用、重复标签、编译错误。

日志：2处Overfull、已有字体替代及glossaries警告，BibTeX字段警告42条；详见pdf_check.json。候选截图未接入正式PDF。

起止核对661个基线文件，意外变化0；其中包括B1登记的16活动文件与509保护文件、B1产物、目标原素材及相关静态代码。HEAD保持5a7900c55c09a3000067c8b19ab3e090a404c6a0。本轮产生的新文件均在stage_b2。具体起始Hash见baseline.json，输出Hash及对照见integrity.json。

本轮执行的是既有视频解码和静态检查，不产生新的机器人/模型实验。未冻结、未提交、未push。
