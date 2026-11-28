# 实验与资产清单

本轮：新训练0、新推理0、新仿真0、新采集0。131个候选资产目录只作L0导航，不声称全部131个已复验。下列26项为有证据范围的主要资产；明细可机读experiment_registry.csv。L1=历史报告；L2=本轮直接核对指定代码/JSON/CSV或字节Hash；本轮不产生新实验L3。

|ID|研究点/实验|性质及已有结果|统计单位/版本|证据/来源|复用限制|
|---|---|---|---|---|---|
|R1-01|一：平台/XML/曲面阵列|实现+指定历史观测；712/16布局和实际代码存在；模型渲染、采样图已更新|current source + static code|L2实现；历史功能L1；`source/sensors/tactile/dex_hand.py`|不证明真实标定或布局优于其他方法|
|R1-02|一：C8几何感知软皮肤|内部物理读出对照/密度诊断；16 STL、48组密度检查；已有B0–B4与邻域扩散开关对照；NO-GO|固定球探针历史记录；非自然任务新Root|L1报告+L2覆盖/面积CSV及几何代码；`outputs/tactile_p43_c8_geometry_aware_soft_skin_2026-11-17/c8_report.md`|不同读出不是TACTO/Taxim公平外部基线；712通道被覆盖≠全部皮肤面积覆盖|
|R1-03|一：C9空间分辨与处理归因|内部诊断/有限消融；330个合格多接触Patch帧，4种E；虚拟细分、死区/低通/邻域处理分层；已有失败画廊|源自C8同批记录，不增加独立采集|L1报告+L2 layer_summary.json；`outputs/tactile_p43_c9_taxel_resolution_limit_2026-11-17/layer_summary.json`|高分辨场是由冻结方程重建；不是真实压力真值；不能外推任意传感器|
|R2-01|二：Top100分级抓取|历史路由统计；100对象记录，98 LATTICE_SUCCESS/2 RL_SUCCESS|对象/缓存混合，非统一预算|L2 summary结构+L1任务状态；`outputs/dex_hand_top100_v2/summary.json`|不是GraspQP/DexEvolve官方方法的公平比较或100/100 Strict|
|R2-02|二：五任务批量记录|历史任务开发；PickPlace100、Push100、Stack48中43成功、Nut100 variant|不同缓存/对象域/物理版本|L1状态/当前summary直接核对；`outputs/stack_task_valid_v24/summary.json`|不相加独立Root，不比较算法性能|
|R2-03|二：10条Sugar扰动Strict|有限验证（既有审计复用）；10份历史FINAL_VERIFIED绑定及Hash已有第三章材料审计|同模板附近扰动，非多物体盲测|L1复用上轮L2审计，本轮不再读数组；`docs/writing_audit/ch03_material_audit_v1_2026-11-20/CH03_MATERIAL_AUDIT.md`|不是通用抓取成功率；14/127完整公平清单仍UNKNOWN|
|R2-04|二：Nut原始/降低/retain|单例开发控制对照；两条失败及一条retain成功，有原始记录；成功视频可追溯|局部物理条件/控制改变|L1结果；B2视频L2复用；`thesis_audit/2026-11-27/stage_b2/STAGE_B2_VIDEO_FORENSICS.md`|不是完整消融、多Seed提升或唯一失败原因证明|
|R3-01|三：LeRobot/P31动作及时序修复|指定样本转换一致性；指定477→340行映射最大差0；三组5/30/1 Episode|历史导出；来源有重用|L1复用第四章审计；当前正文核对；`docs/writing_audit/ch04_material_audit_v1_2026-11-20/CH04_MATERIAL_AUDIT.md`|不是36独立Root、不是本轮新转换|
|R3-02|三：20261125采集资产质量审计|分层抽查；107773目录历史登记；735元数据、125数组、10视频；非全量复验|目录计数/历史小时与抽样结果分开|L2现有JSON字段核对；`outputs/collection_quality_audit_20261125/audit_summary.json`|不是服务器20261128实时规模；大样本目录≠多样独立初态|
|R3-03|三：20261126修复与回放门槛|版本修复/历史对照；旧PickPlace可终态成功但轨迹偏差约0.207775；新开发604行样本偏差0|旧样本和新样本不是单因素对照|L2两个validation JSON直接核对；`outputs/collection_quality_fix_20261126/new_pickplace_case_01_validation.json`|开发修复不回填旧数据；其他任务对应记录按各自门槛|
|R3-04|三：现有质量检查测试|单元测试实现储备；248个相关测试函数静态定位，覆盖缺字段/NaN/时序/状态/Hash/重复等|包含采集与恢复测试，非248种缺陷|L2静态函数清单；本轮未运行；`tests/test_production_recorder.py`|没有已完成的缺陷注入Precision/Recall/F1统一基准|
|R4-01|四：P28旧transport可行性|旧Train组CV诊断；UPPER_BOUND_FAILED；transport C未执行|P24 Train内4折；不是当前P43/C0|L2 summary/transport状态；`outputs/tactile_p28_transport_feasibility_2026-10-27/transport_results.json`|不是统计不可达理论上界，不记0/11|
|R4-02|四：P38幅值校准|较明确的单因素消融；同次采样有/无校准；Mask相同；重新训练Probe；均未达旧0.90|P37旧Train/Validation 5153/1408 与1824/406|L1完整报告；实现/原结果留待协议卡复核；`outputs/tactile_p38_calibration_ablation_2026-11-15/p38_report.md`|不是冻结C0，不可与当前P43汇总横比|
|R4-03|四：C3与C13|基础方法/局部Real幅值修复；原Validation联合均1/11；Real自重建数值改善|原冻结C0，对应各自Real Decoder|L2 C13 CSV + current evidence；`outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv`|不能拼成修复后已迁移成功模型|
|R4-04|四：C14–C17|辅助Probe/采样/表示诊断；辅助Full、Mask、首帧及原型Oracle；不同评价角色|P43开发数据；Oracle非部署|L1决策表与报告；`outputs/tactile_c13_c22_research_decision_2026-11-18/experiment_protocol_summary.csv`|不是四个公平方法基线；不当原C0成绩|
|R4-05|四：C18|Mask改进后完整链筛选；Real Mask约0.815；完整Natural输出采样Seed101/102/103均0/11|冻结最佳45轮；3个生成采样Seed不是3次独立训练|L1报告+L2 final_decision；`outputs/tactile_c18_frozen_mask_screen_2026-11-18/final_decision.json`|初始c18_lossless目录的setup_error与最终screen状态分开；不能声称训练0轮|
|R4-06|四：C19|条件支持/阈值/组权重诊断；精确非零统计依任务/阈值变化；匹配阶段后仍有差异|Train和Validation分开，非力阈值|L1阶段研究报告；`outputs/tactile_c13_c22_research_decision_2026-11-18/REPORT.md`|不是物理接触个数或传感器纯物理差异|
|R4-07|四：C20/C21|条件替换及解析桥接筛选；各自Real Mask gate失败；主C0均NOT_EVALUATED|固定骨干/不同条件；C20预算7轮未证明收敛|L2 final_decision与gate；`outputs/tactile_c20_energy_condition_projection_2026-11-18/final_decision.json`|不是主迁移0/11、不能唯一归因容量/曝光偏差|
|R4-08|四：C22|同规模骨干适配/工程benchmark；已有30轮2430步；BF16 benchmark快2.71倍；主C0未执行|约108秒只是旧训练+Mask评价，不是完整迁移评测预算|L2 benchmark与training_complete；`outputs/tactile_c22_backbone_adaptation_2026-11-18/training_complete.json`|不能推广为所有新实验耗时，不能据此开追加训练|
|R4-09|四：C23/C24|Real自重建诊断/时间幅值修复；6/11与2/11是Real自重建读出；C24部分退化|Real来源；不是Natural迁移|L2 decision记录；本轮未推理；`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/decision.json`|旧C13–C22报告建议C23已被后续执行，不应重复开C23|
|R4-10|四：C25/C26|OT/共用Encoder可行性筛查；C25未执行跨域OT；C26未跑主C0；当前配置停止|无已认证共享物理坐标/代价空间|L2 decision；`outputs/tactile_c25_latent_gmm_ot_feasibility_2026-11-18/decision.json`|不记0/11；不由C25否定受监督Translator|
|R4-11|四：C27|原C3双VAE下映射改进；原Validation联合2/11；不保留为新默认|单候选；不含C13 Decoder修复|L2 decision/对照CSV；`outputs/tactile_c27_c3_route_improvement_2026-11-18/decision.json`|不是最终成功、不是统一模型|
|R4-12|四：P44/P45|共享场与Mask路径对照；同批次K调度；hard/STE梯度检查有效；A/B/C判别容量不完全相同|单Seed4401；B复用P44历史|L2 config/train.py/decision；`outputs/tactile_p45_mask_ablation_2026-11-18/config.json`|机制路径对照不等于容量受控优越性实验|
|R4-13|四：P48|统一合同的内部路线比较；A_K1登记R_W=0.4977202817338595；point90=False|4Session/10Root组等权；非逐样本同步目标|L2 metrics_swd.csv；`outputs/tactile_p48_unified_four_route_audit_2026-11-19/metrics_swd.csv`|49.77%是SWD相对下降；不是迁移成功率|
|R4-14|四：P49/P50|多轮优化/残差多视角结构探索；FR可达100%但SWD/MMD退化；终点关Refiner是诊断|逐轮D与D50不同；P50自F0开始|L1当前正式正文与已审统计；`thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex`|不是外部基线、不是独立重训的Refiner消融|
|R4-15|四：P51|冻结D50可靠性审计；NO-GO；Real Session宏接受0.6808333333|4开发Session；固定0.5，非独立盲测|L2 decision；`outputs/tactile_p51_discriminator_trust_audit_2026-11-19/decision.json`|仅约束D50，不外推P49 D3或全领域|

## 当前数据与模型状态

- 当前主线仍为P43/P44冻结合同：Real Train/Validation=5153/780窗口（32/4 Session）；Natural Sim=4049/905窗口（50/10登记Root）；统计伪配对1200/240窗口。它们与107773条生产登记目录不是同一个训练集合。
- P37旧开发5153/1408 Real、1824/406 Sim与P28旧P24 Train-CV分别登记。旧Validation改名或重新划组不产生未暴露盲测。
- Root为来源分组，不由目录/seed/窗口数自动认证独立；19种登记放置指纹跨划分复用与97个Validation窗口的问题保留原定义，不能认定相同物理初态或忽略复用风险。
- 五任务v4开发包2727行、5 lineage的旧审计不自动覆盖20261126新修复样本。新PickPlace开发样本604行是另一版本，不能把旧样本宣称已修复。
- 生产总量107773/745.14h为历史登记口径；20261125审计明确不是全数组全视频验收。本轮没有访问服务器数据库，20261128最新生产规模UNKNOWN。
- C3双VAE、Translator和C0，以及C13/C22/C24/C27/P44/P50的本地PT文件作存在与字节Hash登记，见model_inventory.json；不加载权重、不运行推理。文件存在不等于有效模型/可直接复用为同一初始化。
- 本地文献PDF仅定位references/答辩版.pdf与用户批注稿；它们不是候选研究论文全文库。第三方DexEvolve仓库存在，其版本与论文设置的完整对应须Phase1核对。

## 新进展和历史快照冲突处理

C13–C22阶段报告的“下一项建议C23尚未执行”是当时状态。当前C23/C24/C25/C26/C27已有结果，因此不再安排重复C23。P45“NO_P46”、C24“NO_C25”等历史停止文字同样保留其事件范围，不能无视后续正式产物，也不能据后续目录存在倒推此前已有授权。当前最晚明确停止边界是P51对D50的NO-GO，继续有效；P52/P53不自动启动。

验收预审包含旧开发Test归档数值，本轮只核对其明确角色及不通过结论，不打开Test/Holdout数组或预测文件，不用765个生成窗口的旧结果替换当前905窗口Validation成绩。
