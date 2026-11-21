# 第五章权威源清单

只读审计日期：2026-11-21。以本轮用户指定路径及实际冻结记录确定，不按修改时间或最高R编号盲选。5.1/5.2的搜索范围与命令见source_search.json，未定位不等于断言其他设备从未写过。

|节|权威源状态|接入主文件|实测TeXcount|
|---|---|---|---:|
|5.1|MISSING_NOT_LOCATED|否|MISSING|
|5.2|MISSING_NOT_LOCATED|否|MISSING|
|5.3|FROZEN_DRAFT|否|1999|
|5.4|FROZEN_DRAFT|否|1496|
|5.5|REVIEWED_R1_NOT_FROZEN|否|1225|
|5.6|FROZEN_DRAFT|否|1605|

当前主文件仍是六章结构；第5个include为body/ch05_experiments.tex，仅旧章标题和待写注释，不是V2第五章正文。5.3–5.6均为未接入的section片段。

## 5.1

权威正文未定位，SHA256=NOT_AVAILABLE，不能用规划/材料审计/旧章骨架代替。

## 5.2

权威正文未定位，SHA256=NOT_AVAILABLE，不能用规划/材料审计/旧章骨架代替。

## 5.3

源：`thesis/releases/ch05_5_3_2026-11-21/section_5_3.tex`

SHA256：`d95b38fdf9821ad10cb9969475fcfdc7d44392a27d4424745bacc55ab99df77b`

PDF：`thesis/releases/ch05_5_3_2026-11-21/section_5_3.pdf`

PDF SHA256：`0841bb3a9625728c4313cdb553f42ad3ee4c5385cb28dca57f97e9629a2f7ca9`

冻结记录：`thesis/releases/ch05_5_3_2026-11-21/FREEZE_STATUS.json`，源与PDF Hash均吻合。FROZEN_DRAFT不是未经记录的用户逐页PDF审核声明。

## 5.4

源：`thesis/releases/ch05_5_4_2026-11-21/section_5_4.tex`

SHA256：`4f083c1142be4c26f318f2808fd234db340cc4d8278a3391da1c9fb6ea2b6c7b`

PDF：`thesis/releases/ch05_5_4_2026-11-21/section_5_4_preview.pdf`

PDF SHA256：`eaf6d27954ccf6e813d10c4ceb81e622c953749a796925c9a7caf8eb3e5b8b0c`

冻结记录：`thesis/releases/ch05_5_4_2026-11-21/FREEZE_STATUS.json`，源与PDF Hash均吻合。FROZEN_DRAFT不是未经记录的用户逐页PDF审核声明。

## 5.5

源：`docs/writing_audit/ch05_writing_v1_2026-11-20/section_5_5_v1_2026-11-21/revision_r1/section_5_5.tex`

SHA256：`4dd9349979abcae4f2c88b2ecb11dc0dfc76b28c455c5e8fec14310a53b96bea`

PDF：`docs/writing_audit/ch05_writing_v1_2026-11-20/section_5_5_v1_2026-11-21/revision_r1/compile_review/section_5_5_preview.pdf`

PDF SHA256：`39d7c7edde4465e9eb6b16d040ca4f8df206d1e59ff5867c4ace4f4875d2c906`

无thesis/releases/ch05_5_5_2026-11-21目录或正式FREEZE_STATUS。现有记录为COMPILE_REVIEW_COMPLETE_AWAITING_USER_PDF_REVIEW；R1与临时构建副本字节相同，但编译通过不是冻结。本轮按用户指定R1审计，不自行提升状态。

## 5.6

源：`thesis/releases/ch05_5_6_2026-11-21/section_5_6.tex`

SHA256：`e6340ccebf835104f257dca05df1e398545230c59661291176fa862eb5586091`

PDF：`thesis/releases/ch05_5_6_2026-11-21/section_5_6_preview.pdf`

PDF SHA256：`151ab5240127b5bc11db018a58188e74a40c16614a0a3116b3228c6f03e795af`

冻结记录：`thesis/releases/ch05_5_6_2026-11-21/FREEZE_STATUS.json`，源与PDF Hash均吻合。FROZEN_DRAFT不是未经记录的用户逐页PDF审核声明。

各历史候选路径及是否与权威源同Hash见JSON的other_candidates。旧V1/R1/R2只作历史对照，不参与正文合计。V2 5.7小结在本次5.1–5.6范围之外，但完整七节正文尚未定位该源。
