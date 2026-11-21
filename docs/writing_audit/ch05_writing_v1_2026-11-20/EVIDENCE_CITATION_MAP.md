# 5.3逐段论点—实现—数据对应

根目录简称：G=`outputs/tactile_gmm_dual_vae_2026-11-16/`；C3=`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`；审计A=`docs/writing_audit/ch05_material_audit_v1_2026-11-20/`。源代码均只读，不导入执行。既有审计完整索引与本次输入Hash保留在baseline.json。

|正文位置|论点与公式/图表|代码、配置与数据|证据及边界|
|---|---|---|---|
|第1段、流程图|无同步物理配对，以Real统计引导物理Sim；不直接画高斯响应|A/CH05_MATERIAL_AUDIT.md；C3/code/expand.py 41–80、inverse_physics.py::run；既有传感器调用链|L2代码；未新物理执行，不声明迁移成功|
|第2段、式GMM|网格、质心偏移、分Session抽样、48/16/2048|G/code/fit_gmm.py::patch_frames、samples，第6–18行；gmm.py::grid|L2；正值P5仅内部拟合筛选，无Real Validation拟合|
|第3段|完整协方差、EM、K1/2/3、60次、0.0025/0.0004正则、BIC、稳定性回退、ALL阶段|G/code/gmm.py::fit_em，第10–17行；fit_gmm.py第20–41行；C3/expand.py g[components][0,pi]|L2；不是VAE latent的GMM，不将48帧说成48独立主体|
|第4–5段、式责任度|w为四帧均值，责任度、响应量m、加权中心c；拟合/逆构造坐标差异|C3/code/expand.py第39–51行；inverse.py同构段；G/fit_gmm.py::samples|L2；公式含正常分量后验关系，代码用logsumexp稳定计算；质量m称响应量，不赋予物理质量单位|
|字典表|16×5×3×2=480，4mm，32帧取12–15，160指纹|C3/code/dictionary.py 11–25行；dictionary_complete.json；A/verified_observations.json dictionary及dedicated_array_contracts|L2已审计Manifest/数组和本轮代码复核；不称独立Root|
|字典检索段|Top-5、posture0/0.25mm非零条目、最近响应中心、深度插值、四帧profile|C3/code/expand.py第52–60行；dictionary_plan与registration；inverse_physics.py profile命令|L2；深度非真实材料变形，两域processed没有共同力学标定|
|反馈段、候选关系图|倍率范围0.5–2、基础深度0.1–1mm；最多3次候选，首个通过即停止|C3/code/expand.py第64–80行及registration.json|L2；真实代码反馈含指数1/0.7；正文简述未引入新计算方法|
|选择公式及解释|先在已执行候选中取结构代价最小，再检查准入；相似度/锚点/协方差与0.8/0.2/0.1门槛|C3/code/matching.py::inherited_metrics；expand.py第81–87行|L2；不是对所有未知物理条件求全局最优，也不是先过滤合格候选再取最优；距离在索引空间|
|登记段|固定元数据顺序、无C0评分、Real/Sim数值重复检查和来源绑定|C3/code/expand.py第20–38、81–91行；pair_expansion_manifest及既有审计|L2；不同Hash不证明独立Root；失败结果保留，不在本轮重建|
|末段|1200/240，Top-5与单一对应丢失的多解信息；Natural覆盖不获保证|C3/expansion_complete.json；A/verified_observations.json dedicated_array_contracts、来源重计；C3/pipeline_cpu.py读取sim/real生成潜变量监督|L2；不引用任何最终迁移通过率，不提前写5.4/5.5模型细节；未来软监督收益未验证|

## 公式和实现等价范围

式GMM描述二维统计密度；式责任度用固定网格q代入冻结参数，明确没有假设它与拟合偏移u具有已认证物理对应。BIC中的6K−1来自二维均值、完整对称协方差与混合权重自由度。选择代价与代码一致；距离无双方活跃帧时不满足登记条件，实际实现为选优提供有限替代项，不在公式中伪装存在有效质心距离。

图均为代码流程示意，不含仿造的GMM椭圆、接触截图或成绩。正文不新增外部引文、不修改BibTeX。项目方法事实由原代码和登记产物支撑；没有用新的未经核验文献补齐。
