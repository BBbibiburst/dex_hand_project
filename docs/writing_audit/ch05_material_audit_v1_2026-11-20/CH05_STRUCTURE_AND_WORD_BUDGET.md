# 第五章结构、材料与篇幅预算

最终建议保持已审核七节结构，总计**8000–9000 TeXcount，中心8500**。这是写作预算，不是本轮新增正文的实际统计。七节与V2论证卡逐项对应，不新增大量小标题。

|节|标题|最低|最高|建议|材料与段落安排|
|---|---|---:|---:|---:|---|
|5.1|真实与仿真触觉差异分析|1000|1100|1050|旧P37/P39规模与合同→Mask/幅值/空间→保留帧变化→Session/Task混杂和映射限制，4–5段|
|5.2|非配对条件下的迁移问题定义|700|800|750|完整4×712合同→三类数据与监督角色→伪配对可支持/不能支持，3–4段|
|5.3|Real统计先验引导的仿真触觉探针与伪配对|1900|2100|2000|Train-only索引GMM→软分量目标→字典/物理执行→准入→独立性/饱和/覆盖，6–8段|
|5.4|基于Dual-VAE的双域触觉表示|1350|1550|1450|独立表示→重参数与损失→AR Mask及Gamma幅值→训练/读出版本与C13对照，5–6段|
|5.5|基于潜变量Translator的初始跨域重建|1100|1250|1175|GRU时序映射→冻结与监督→训练/部署分开→覆盖及监督歧义，4–5段|
|5.6|初始迁移实验及局限分析|1550|1750|1650|数据版本/PRIMARY协议→逐实例结果→C23/C24与C27限定对照→物理及潜空间失败边界，5–7段加表|
|5.7|本章小结|400|450|425|完成的链路、未完成的联合目标、为第六章留下的问题，2–3段|
|合计||8000|9000|8500||

以上均有现存材料支撑。5.3、5.4可多写实际算法与实现合同；5.6选择有判别力的对照，不罗列全部C编号。5.2没有可认证物理条件是可写限制，不通过增加定义凑篇幅。

## 建议图表（后续写作授权后再制作）

- 图：三类数据及监督角色；GMM—字典—实际探针—712D构造链；独立Dual-VAE与冻结GRU训练/推理双路径；Train/Validation支持距离与典型统计响应。约4–5图，逐图核实来源角色，不取Test图直接贴入。
- 表：数据版本与分组；GMM/字典与配对来源；模型训练/冻结及参数；11实例Validation主结果；C23/C24目标侧机制对照；有限失败路线与未执行状态。约5–6表，必要时合并。
- 公式：完整输出F与统计伪配对；GMM/责任度及真实使用的匹配代价；VAE重参数与KL/NLL；Translator Huber+Decoder NLL；BA/AUC/D。约5组公式足够，不引入未执行OT作为本文算法。

每个图表有独立论证任务。Patch热图应按传感器方阵显示；固定索引不写成已认证真实曲面坐标。连续压力代理C7–C9不混同原生产processed数据，也不需要展开为另一整章。

## 第五/六章分工

第五章解释统计监督、独立表示、初始Translator及其受限结果，C13/C23/C24作为目标域读出瓶颈对照，C27作为原路线有限适配及专项边界。第六章从这些问题引出P44共享参数向量场，再论P45梯度路径、P46自然变异、P47多指标负对照、P48统一分布评价、P49/P50对抗及质量退化、P51评价器NO-GO。

P51只审计D50，不覆盖P49；P52/P53未执行。不在第五章用P49或P50的FR100%补成初始Translator成功。积分步数K、对抗轮数和Epoch分别定义。

## 源文件安排建议

V2第490行仅建议未来使用`ch05_dual_domain_reconstruction`，未批准整体迁移。本轮不建tex，不动六章include。正式写作任务书可明确新建`thesis/body/ch05_dual_domain_reconstruction.tex`，并保留旧ch03_dual_vae.tex及ch05_experiments.tex；不能自行覆盖或迁移它们。


## 路径简称（各表中的相对文件名以此为根）

- **C3**：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`。
- **C4**：`outputs/tactile_p43_c4_root_cause_audit_2026-11-16/`。
- **C13**：`outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18/`。
- **C23**：`outputs/tactile_c23_real_reconstruction_c0_2026-11-18/`。
- **C24**：`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/`。
- **C25**：`outputs/tactile_c25_latent_gmm_ot_feasibility_2026-11-18/`。
- **C26**：`outputs/tactile_c26_shared_frozen_encoder_2026-11-18/`。
- **C27**：`outputs/tactile_c27_c3_route_improvement_2026-11-18/`。
- **P39**：`outputs/tactile_p39_domain_decomposition_2026-11-15/`。
- **GMM**：`outputs/tactile_gmm_dual_vae_2026-11-16/`。

证据：L0仅定位；L1历史报告/配置声明；L2本轮直接核对源代码、指定Train/Validation数组、CSV及Hash；不执行L3。全部实际读取文件及SHA256见`audited_sources.json`；函数起止行见`source_code_index.json`。
