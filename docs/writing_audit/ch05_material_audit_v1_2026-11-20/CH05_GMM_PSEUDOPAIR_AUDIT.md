# GMM统计伪配对专项审计

## 1. 三种不同对象

- 早期GMMRenderer：GMM目录code/gmm.py::GMMRenderer将索引空间核直接重分配到Sim Patch，尽量保留Patch代理总响应。它属于数学响应重建分支，不能用来解释C3的物理探针数据来源。
- **C3统计伪配对**：Real窗口→统计分量→字典检索→探针控制→实际MuJoCo接触→原生touch及信号处理→保存Sim窗口。本章主链采用此分支。不是直接把高斯数值写入712D。
- **C25潜空间GMM-OT**：在独立Real/Sim Encoder输出间定义跨坐标代价的前提审计，前提未通过，未做跨域W2/OT/GMM稳定性验证。该NO-GO不等同于统计GMM无效。

## 2. 冻结GMM原始拟合与来源

实际拟合源为GMM/code/fit_gmm.py、gmm.py，不应仅凭早期P41阶段名称替代其实现。读取P37/train.npz中的Real、Stage、Session；每Patch正值P5定内部拟合阈值，未用Real Validation拟合。该阈值用于先验抽样，不是后验调C0的Mask阈值。

Patch坐标由行列固定映到[-.5,.5]二维网格。每Session最多48个有效帧，每帧按正响应概率采样16个Taxel并减去该帧响应质心；合并至最多2048点、对Session池轮转抽样。拟合的是**质心相对的响应扩散偏移分布**，不是物理接触力/压力坐标，也不是完整712维联合概率。

自写EM拟合二维完整协方差GMM，K候选1/2/3，最多60迭代；初始化协方差加0.0025 I，更新加0.0004 I。BIC=-2 logL+(6K−1)log n。ALL/CLOSE/HOLD/LIFT各自拟合，样本不足回退同Patch ALL；按Session三折核总变差均值>0.25且K>1时回退K1。实际参数选中K如下，来自gmm_bic.csv：ALL为1个K2和15个K3；CLOSE、HOLD各2个K2及14个K3；LIFT为4个K2及12个K3。无K1最终选择。核稳定性是抽样索引先验的稳定性，不是硬件物理标定。

GMM原参数→集成P43副本→C3/gmm_parameters.npz的实际SHA256相同。C3直接取stage索引0即ALL，因此分阶段拟合虽存在，未用于该探针逆构造路径。

## 3. 字典、责任度、检索及物理链路

C3/code/dictionary.py：16Patch×5局部位置×3深度(0.25/0.5/1mm)×2姿态指令=480条件，球半径4mm，固定取32帧记录的第12–15帧。dictionary_manifest.csv重计只有160种放置后首帧状态指纹；姿态指令为0或0.0006，不能把条件数称独立Root。

C3/code/expand.py（pilot为inverse.py）逐Patch以四帧平均响应w_j评估冻结GMM责任度：r_jk∝π_k N(q_j;μ_k,Σ_k)，m_k=Σ_j w_j r_jk，中心为Σ_j w_j r_jk q_j /m_k。注意拟合阶段使用质心相对偏移、该逆构造在固定网格直接评估责任度；这是现行统计启发式，尚无证据认证为物理逆模型。

各Patch分量合并后按m_k从大到小只留最多5个。字典检索只用posture0、0.25mm且有非零响应的基元，按生成响应质心与目标分量中心的欧氏距离选最近位置。不是在所有姿态/位置/深度上进行无约束全局搜索。

深度初值np.interp(m_k, dictionaryPatchMass, depths)。四帧Patch总响应除以其均值形成depth_factors，并限于[0,2]作为物理指令因子。最多三次实际候选：检索、深度反馈、深度加一个site邻近修正；反馈倍率由(Real总响应/Sim总响应)^(1/0.7)并限[0.5,2]，基础命令限[0.1,1]mm。所有这些约束来自旧实现，不在本轮更改。

inverse_physics.py/physics.py::run、reset经mocap目标与自由球weld驱动实际刚体碰撞；sensor.read取得native/processed并保存Episode。source/sensors/tactile/dex_hand.py::augment_spec→_add_touch_sensor→bind→read_raw读mjData.sensordata，再经signal_processing.process。native法向标量与processed无量纲量不同；没有Flex硅胶形变、真实压力场或真实Taxel剪切分量。

## 4. 匹配准入并非完整触觉对应

matching.py::inherited_metrics逐帧计算16Patch总响应归一化比例的Cosine，再平均四帧；在Real总响应主Patch上计算Sim质心与“Real质心+冻结GMM混合均值”锚点距离（双方活跃帧均值）；计算Sim协方差到冻结GMM混合协方差的平均Frobenius误差。

通过门槛固定为Cosine≥0.8、质心距离≤0.2、协方差误差≤0.1且非零；结构代价为1−Cosine+距离+协方差误差，最多候选中取其最小且过门槛者。早期平均窗口评分另行封存，不作为正式配对准入。

这些只限制部分统计摘要，既不认证逐Taxel物理对应，也不保证完整Real latent可由Sim确定恢复。一个目标最终只留一个最佳窗口，seen及seen_real避免有效数值重复。1200/240是有效统计伪配对窗口数，不是1200/240物理主体。

## 5. 覆盖与饱和的直接证据

C4/paired_source_inventory.csv重计Train/Validation各机器人reset只有1种；19个放置后状态、17种拓扑跨组重叠，但Real源/Trial/Session和完整Real/Sim数值Hash无直接交叉。没有据此证明全局无泄漏，也不能称验证是全新物理初态泛化。

C4/inverse_control_saturation.csv重计Train基础命令到1mm为5968/6000=99.4667%，Validation为1197/1200=99.75%。这不是所有完整配置完全一样：四帧profile仍可不同，命令深度也不等于实际变形。processed两域数值未先建立共同量纲，直接插值易触及边界。

同一Sim Encoder下，排同Root后NaturalVal→NaturalTrain原始距离中位4.48091，NaturalVal→PairedTrain为14.73386；PairedVal→PairedTrain仅0.02163，允许同Root时0.007826。参考Train标准化/白化及Cosine也应与域内参照并报（见distance_summary.csv），不把绝对距离或密度比写成百分比覆盖率。

C5在可标定范围将基础命令饱和降为0，但完整固定目标Train有效32/48→19/48、Val10/12→8/12；20个位置缺乏对应标定曲线，不能把差距全归深度。C6实际仅1个Train目标形成可执行多球位置修正，29个沿用S0；S2与Validation空间候选未执行。因此不能说空间优化已被充分试验并证明不可能。

## 6. GMM能力使用清单

|状态|可核实内容|证据与边界|
|---|---|---|
|已用|混合权重、二维均值、完整协方差、软责任度、分量响应质量、分量中心、混合矩和统计匹配门槛|fit_gmm.py/gmm.py、expand.py、matching.py；不是纯粹完全无GMM的硬最近邻|
|已有信息但未进入现行监督|已拟合的分阶段参数（C3只取ALL）；目标完整分量及其质量；失败/备选物理候选及匹配得分|inverse_gmm_*.json、候选Manifest和代码；可以证明有这些记录，未证明利用它们的收益|
|已计算但被压缩|软责任度先用于分量质量，再top5、硬字典位置和单一最佳输出；完整逐Taxel责任矩阵未作为Translator输入或损失权重保存使用|expand.py及pipeline_cpu.py：训练只读paired sim/real，不读分量权重/条件ID|
|未来设想，未验证|质量加权/多解概率监督、条件多模态对应、更密空间与加载字典、随机传输|不新增训练任务、不声称已实现；C25不能为这些提供正面验证|

**判断：统计GMM尚未被穷尽，但扩大其利用并不能绕过字典支持、加载可辨识性、Real Decoder泛化及共同物理条件缺口。** 本轮不实现任何新算法。


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
