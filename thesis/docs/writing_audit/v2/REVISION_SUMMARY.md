# V2修订交付与审核事项

日期：2026-11-19。此文件记录规划与规范修订，不是ChatGPT章节任务书。

## 新建文件

- THESIS_PLAN_V2.md：七章42个小节论证卡，逐项记录问题、论点、证据、公式、图表、承接和缺口。
- ROBOTICS_EVIDENCE.md：R-01至R-08机器人、阵列、抓取、Strict、五任务、LeRobot、关联和质量控制证据。
- scripts/audit_robotics_writing_sources.py：只读历史产物检查。
- scripts/check_writing_workflow_v2.py：规划、证据Hash及保护范围检查。
- docs/writing_audit/v2/robotics_source_audit.json：31个代码/产物来源Hash及读取结果。
- docs/writing_audit/v2/validation.json：42小节、22份LaTeX/模板文件未改、V1保留及来源检查结果。
- 本交付说明。

## 修改文件

- THESIS_WRITING_STANDARD.md：完整研究范围、五阶段门槛、ChatGPT任务书职责、核验先交付、用户确认后编译冻结、语言正反例。
- AGENTS.md：强制读取当前V2规划及机器人证据，禁止跳阶段和自动选择下一节。
- REFERENCE_THESIS_ANALYSIS.md、SYMBOLS_AND_TERMS.md：追加V2适用范围和工作流说明；既有范本阅读与公式记录保留。
- EXPERIMENT_EVIDENCE.md：追加V2范围和E-P48原始CSV核验，未改历史实验。
- README.md：当前入口指向V2与机器人证据。

## 历史保护

THESIS_PLAN.md逐字节保持；其余六个修改文件的原件快照保存在docs/history/workflow_v1_2026-11-19/。原校验记录与现有正文骨架、封面、模板核心均未改。

## 目录调整理由

第二章平台保证观测和执行基础；第三章论述机器人可达抓取及专家生成；第四章界定数据关联和质量；第五章论述统计伪配对与初始重建；第六章分析对抗优化及评价局限。避免仅围绕触觉实验收窄研究，也不把生产工程全部算成算法创新。原六章include不能直接代表七章；只提供审核后的命名迁移建议，未执行。

## 必要证据缺口

1. 模型XML/闭链及控制器精确配置、第三方算法原文与版本。
2. 候选级筛选排除、逐Episode Strict及各任务质量绑定；当前历史汇总缓存不等于新独立验收。
3. LeRobot源lineage、独立Root、动作因果、视频PTS及全量字段检查。已检查的5/30 episode_index不证明5/30独立采集主体。
4. 服务器当前数据库、Asset Hash和完整文件验收本机无法访问，标UNKNOWN。
5. Real通道物理坐标及单位未认证，不声称跨设备完全同步。
6. 文献库、备案题目、学院格式要求及创新称谓仍待核实。

## 待用户与ChatGPT审核

- 是否批准七章结构及三方向贡献定位。
- 当前暂定论文题名是否需要按正式备案及完整研究范围调整（未改封面）。
- 审核后是否授权单独进行七章空白骨架调整；未授权前保留六章。
- 后续具体小节由用户指定、ChatGPT给任务书。Codex先交材料核验，获通过后才写正文。

P51 NO-GO维持，P52/P53未启动。无训练、无新仿真、无正式正文、无LaTeX编译、无commit/push。当前停止等待审核。
