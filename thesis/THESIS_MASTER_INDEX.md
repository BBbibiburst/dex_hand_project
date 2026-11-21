# 论文主索引（2026-11-21阶段基线）

本文件为导航，不覆盖历史规划、任务书和冻结状态。正式主文件是`thesis/thesis.tex`，当前含第1–5章，不是旧六章文件名对应的V2章节。题目仍为《面向灵巧手触觉信号的仿真到现实迁移方法研究》，为现有封面暂定题目，研究范围比题名更广，须人工最终确认。

## 正式源及权威性

|章|当前正式源（以下相对thesis/）|权威判断|
|---|---|---|
|1|body/ch01_introduction.tex|与releases/ch01_2026-11-20/FREEZE_STATUS.json登记source_sha256一致|
|2|body/ch02_simulation_platform.tex|与releases/ch02_2026-11-20/ch02_simulation_platform.tex逐字节一致；冻结记录存在|
|3|body/ch03_grasp_demonstrations.tex|主文件实际include；compile_review已编译，但无正式冻结记录|
|4|body/ch04_multimodal_data.tex|主文件实际include；R2排版版，尚待用户审核冻结|
|5|body/ch05_dual_domain_reconstruction.tex|主文件实际include；内部input body/ch05_sections/六个片段；来源见CH05-FINAL manifest|
|6|MISSING|尚无V2正文；不能把旧body/ch04_adversarial_transfer.tex占位当第六章|
|7|MISSING|旧body/ch06_conclusion.tex只是空骨架，不是已完成第七章|

第五章5.1/5.3/5.4/5.5/5.6分别来自`releases/ch05_5_*_2026-11-21/`冻结源；5.2为`../docs/writing_audit/ch05_final_integration_2026-11-21/section_5_2_v1_1/section_5_2.tex`。正式副本不应独自改动，应同步来源映射。5.1 V1/R1/R2、5.3 R1/R2/R3等历史版均保留，不能按最高文件夹名猜权威。

## PDF、统计、图表与文献

- 当前本机构建：`thesis/build/thesis.pdf`（71页；临时构建不进Git）。
- 可共享完整PDF：`docs/writing_audit/ch05_final_integration_2026-11-21/full_thesis.pdf`；本轮核对与当前构建逐字节一致，构建输入Hash亦相符。
- 第五章：同目录`chapter05.pdf`，13页，印刷48–60；完整源`chapter05_complete.tex`及阅读稿`CH05_READING_DRAFT.md`。
- 图表：`thesis/figures/`，第2章与第4章使用真实资产/历史标量的图；第3章真实案例仍有待补图位；不能当图像已完成。
- 参考文献：`thesis/reference.bib`，30条，当前正文均有引用；第三方全文不随Git公开。
- 全部资产路径、分类、体积、SHA256：`docs/writing_audit/thesis_baseline_2026-11-21/THESIS_ASSET_INVENTORY.csv`。
- 当前统一统计：同目录`wordcount/counts.json`，正文1–5章42318，含标题/图表文字、公式另计。

## 审核与证据导航

- 总体规划/边界：THESIS_PLAN_V2.md、THESIS_WRITING_STANDARD.md、SYMBOLS_AND_TERMS.md。
- 平台/示教证据：ROBOTICS_EVIDENCE.md；触觉迁移：EXPERIMENT_EVIDENCE.md。
- 第三章：docs/writing_audit/ch03_writing_v1_2026-11-20/compile_review/。
- 第四章：docs/writing_audit/ch04_writing_v1_2026-11-20/revision_v2/。
- 第五章：docs/writing_audit/ch05_final_integration_2026-11-21/下FINAL_AUDIT、ISSUE_LIST、NUMERIC_CROSSCHECK、SOURCE_MANIFEST、integrity。
- 本轮进度/研究/问题/下一步/ChatGPT材料分别见同级THESIS_PROGRESS_REPORT.md、THESIS_RESEARCH_SUMMARY.md、THESIS_ISSUE_TRACKER.md、THESIS_NEXT_STEPS.md、CHATGPT_THESIS_REVIEW_PACKET.md。

上面docs/路径除thesis/docs明确前缀外均相对项目根目录。原始实验与权重仍在outputs/等既有位置，不复制成论文正式结果，不改划分与指标。
