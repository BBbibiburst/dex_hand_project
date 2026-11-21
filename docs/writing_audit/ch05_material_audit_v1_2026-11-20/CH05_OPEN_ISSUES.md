# 第五章重要缺口、版本差异与写作禁区

## 不阻断初稿、必须如实保留的缺口

1. **共同物理语义UNKNOWN**：Real硬件Taxel方向、与Sim碰撞/感知面积的对应未认证，不能把索引同名当物理对应；任务/阶段同名也不证明同接触状态。
2. **没有同步物理配对**：Real-GMM引导物理Sim只匹配摘要，不能用伪配对重建误差当真正实机目标误差。Natural没有正确Real latent标签；其latent映射真值RMSE为UNKNOWN。
3. **独立性有限**：1200/240配对是数值窗口，不是独立Root。原来源CSV无Real源直接交叉，但reset、放置状态和拓扑复用；不能升级为全局无泄漏证明或新物理初态泛化。
4. **覆盖非百分比**：14.7339与0.00783来自同一Sim Encoder支持距离，但有不同参考/同Root条件；必须同时给域内参照及排同Root结果，不能称99.9%未覆盖。白化与Mahalanobis用同协方差时数学等价，不是两份独立证据。
5. **目标侧仍不完整**：C23 Real重建6/11，不是Decoder已完全可用；C24时序误差改善伴尾部收缩、MSE上升及联合2/11，不能只报42.68%。训练足够/容量不足不凭耗时或Loss下降判断。
6. **开发评价暴露**：Real Validation4Session及旧开发Test均有历史使用；不能包装成新盲测。P28也不能为冻结C0给出理论上界。
7. **模型细节缺失**：C3逐Epoch潜变量有效秩及部分分支分项未保存；早期优于ep30的模型权重未保存，其读出结果UNKNOWN。不能补造曲线或重选未保存checkpoint。
8. **GMM软信息未充分用于监督**：可以写“现行实现只用其中一部分”，不能写“改为软匹配必然达标”。拟合相对坐标与逆构造固定网格的语义需保留为工程统计假设；本轮不改实现。
9. **真实任务收益未验证**：冻结域评价、Patch统计和输入敏感性不等同于机器人任务成功率或物理标定。没有新的实机试验，也不建议为写本章补做。

## 必须区分的具体数字/状态

|容易误写的项目|核验后的正确限定|
|---|---|
|C3报告1/11与765窗口|历史开发Test汇总；本轮不读样本/预测，不改名Validation|
|当前C3原GRU Validation|通过C11-F已有复现CSV确认1/11；780Real+905Natural，与Test数值不同|
|C13冻结C0 vs C14重训Probe|独立目录/协议，不能互换；C14表frozen_C0为False|
|C23 MSE/IoU与6/11|Real同源780窗自重建；MSE0.02631960、IoU0.51171548|
|C24改善42.68%|保留帧log1p增量损失降低，不是每秒物理导数改善；Real非严格20Hz|
|C24完整结果|MSE0.027157449、IoU与C13相同；平均帧变化从8.76×Real到4.00×Real，联合6→2|
|C25 NO-GO|跨独立潜坐标代价前提不成立；GMM稳定性/W2/OT未执行，不写试过OT失败|
|C26|共享数值坐标成立，结构保持未通过；C0未运行，NOT_RUN不等于0/11|
|C27 D专项|11/11，原联合2/11；原C3双VAE，不是C13 Decoder版本|
|零比例0.82674|旧Test汇总，不能当Val；C4 CSV的Val输出零比例0.83420487，RealVal0.83428677|
|Mask IoU0.0862|Natural输入与生成输出的固定索引代理，非Real–Sim真实物理配对IoU|
|99.47%/99.75%|完整配对球体基础1mm命令饱和；不等同所有实际压入深度或四帧配置相同|
|C5饱和0%|有限可标定条件内，未同时改善全部配对数量与覆盖，不能说加载问题已整体解决|
|C6 NO-GO|空间逆映射支持不足；实际仅1次S1多球修正，不证明空间方案普遍失败|
|C7–C9|有限压力/Taxel/处理诊断，未产生新的联合迁移通过成绩|
|C10-R1|修复采样稳定性；PairedVal latent及IoU仍不优于GRU，未运行Natural或C0；不能否定所有Diffusion|

## 待审核的写作取舍

建议C3/C4为主，C13/C23/C24和C27组成结果/局限对照；C5–C9、C10/R1、C25/C26汇为精简限制表，不按编号逐项写开发日志。所有通过数必须标注数据来源与协议。建议直接进入初稿写作，但仍等待用户及ChatGPT正式任务书，不自行开始。

没有必须靠新增实验才能补齐的初稿缺口。若后续要求宣称真实物理配对、完整联合0.90或真实任务增益，则当前证据不满足；应收缩结论，而非编造或在本轮扩展实验。


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
