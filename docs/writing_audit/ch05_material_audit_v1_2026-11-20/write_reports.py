from pathlib import Path
import json,csv,hashlib,subprocess
O=Path('docs/writing_audit/ch05_material_audit_v1_2026-11-20');d=json.loads((O/'verified_observations.json').read_text());mr=list(csv.DictReader((O/'validation_metric_recheck.csv').open()))
paths={'C3':'outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16','C4':'outputs/tactile_p43_c4_root_cause_audit_2026-11-16','C13':'outputs/tactile_p43_c13_amplitude_context_repair_2026-11-18','C23':'outputs/tactile_c23_real_reconstruction_c0_2026-11-18','C24':'outputs/tactile_c24_temporal_amplitude_repair_2026-11-18','C25':'outputs/tactile_c25_latent_gmm_ot_feasibility_2026-11-18','C26':'outputs/tactile_c26_shared_frozen_encoder_2026-11-18','C27':'outputs/tactile_c27_c3_route_improvement_2026-11-18','P39':'outputs/tactile_p39_domain_decomposition_2026-11-15','GMM':'outputs/tactile_gmm_dual_vae_2026-11-16'}
source_map='\n\n## 路径简称（各表中的相对文件名以此为根）\n\n'+'\n'.join(f'- **{k}**：`{v}/`。' for k,v in paths.items())+'\n\n证据：L0仅定位；L1历史报告/配置声明；L2本轮直接核对源代码、指定Train/Validation数组、CSV及Hash；不执行L3。全部实际读取文件及SHA256见`audited_sources.json`；函数起止行见`source_code_index.json`。\n'
def write(n,s,links=True):(O/n).write_text(s+(source_map if links else ''),encoding='utf-8')
# One canonical list of numeric claims with source selectors and scope.
cards=[]
def card(id,source,selector,value,role,denom,meaning,seed='N/A',evaluator='统计诊断',level='L2'):
 cards.append(dict(claim=id,source=source,selector=selector,value=str(value),model_seed=str(seed),data_role=role,evaluator=evaluator,preprocessing='原版本合同，不重拟合；C0为原log1p及共享Train-only scaler' if 'C0' in evaluator else '对应源文件/代码原定义',threshold='C0:0.5; BA[.45,.55],AUC[.475,.525],D>=.9' if 'C0' in evaluator else '沿用原定义，不搜索',denominator=denom,evidence=level,allowed_claim=meaning))
metricsources={'C13_Natural_Validation':'outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv','C0_Original_Validation':paths['C3']+'/c0/metrics.csv','C3_Natural_Validation_C11F':'outputs/tactile_p43_c11_latent_support_audit_2026-11-17/probe_metrics.csv','C23_Real_reconstruction':paths['C23']+'/metrics.csv','C24_Real_reconstruction':paths['C24']+'/metrics.csv','C27_Natural_Validation':paths['C27']+'/metrics.csv'}
for r in mr:
 name=r['experiment'];origin=r['negative_origin'];den='780 Real参考+780 Real重建；同一4 Session来源' if origin=='Real reconstruction' else '780 Real/4 Session +905 Natural/10 Root'
 card(name+':'+r['instance'],metricsources[name],f"instance/probe={r['instance']}; Validation"+(';group=F' if 'C11F' in name else ''),f"BA={r['BA']};AUC={r['AUC']};D={r['D']};joint={r['joint_pass']}",'Validation',den,'这是Real读出对照，不是Sim迁移' if origin=='Real reconstruction' else '冻结C0开发结果，不是新盲测',r['instance'].rsplit('_',1)[-1] if '_' in r['instance'] else 'det','PRIMARY C0 / readout diagnostic' if origin=='Real reconstruction' else 'PRIMARY C0')
for role,c in d['loading_saturation'].items():card('saturation_'+role,paths['C4']+'/inverse_control_saturation.csv',f"role={role};sum(base_depth_at_max1mm)/sum(balls)",c['at_max']/c['balls'],role,f"{c['at_max']}/{c['balls']}个球体基础命令",'基础命令饱和，不是硅胶真实形变')
for role in ['train','validation']:
 for k,v in d['paired_independence_recount'][role].items():card('pair_'+role+'_'+k,paths['C4']+'/paired_source_inventory.csv',f'role={role};unique({k})' if k!='rows' else f'role={role};rows',v,role,'1200/240有效窗口来源元数据','来源计数，不等于独立物理Root')
card('C23_MSE',paths['C23']+'/reconstruction.json','MSE',d['C23_reconstruction']['MSE'],'Real Validation','780窗/4 Session','同源配对自重建误差',4313)
card('C23_Mask_IoU',paths['C23']+'/reconstruction.json','Mask_IoU',d['C23_reconstruction']['Mask_IoU'],'Real Validation','780窗/4 Session','同源索引Mask IoU，非跨域物理配对',4313)
for row in d['C24_structure']:
 for k in ['reconstruction_MSE','Mask_IoU','L_temporal','mean_abs_frame_delta','positive_P95','positive_P99']:
  if k in row:card('structure_'+row['source']+'_'+k,paths['C24']+'/Real_C13_C24_structure.csv',f"source={row['source']};{k}",row[k],'Real Validation','780窗/4 Session','指定Real读出比较，非Natural迁移成绩',4313)
for r in d['distance_summary']:
 if r['metric']=='euclidean' and (r['root_mode']=='cross_root' or r['query']=='Paired_validation'):
  card('distance_'+r['query']+'_'+r['reference']+'_'+r['root_mode'],paths['C4']+'/distance_summary.csv',f"query={r['query']};reference={r['reference']};metric=euclidean;root_mode={r['root_mode']};p50",r['p50'],'Train/Validation',r['valid_queries']+'有效query','同一个Sim Encoder下的相对支持诊断，不是百分比覆盖率',4302)
for name in ['real_vae','sim_vae','translator']:
 for k in ['epochs','best_epoch','optimizer_steps','parameters']:
  card('C3_'+name+'_'+k,paths['C3']+'/'+name+'/complete.json',k,d['training'][name]['complete'][k],'Train/Validation','完整训练记录','历史实际训练，不是本轮重新训练',{'real_vae':4301,'sim_vae':4302,'translator':4303}[name])
for r in d['P39_mask_ALL']:
 if r['candidate'] in ['Real','raw']:
  card('P39_activation_'+r['role']+'_'+r['candidate'],paths['P39']+'/mask_statistics.csv',f"role={r['role']},candidate={r['candidate']},stage=ALL,patch=ALL;activation_fraction",r['activation_fraction'],r['role'],r['windows']+'窗/'+r['groups']+'组','旧P37版本描述量，不代表新Natural同条件域差')
with (O/'claim_cards.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(cards[0]));w.writeheader();w.writerows(cards)

write('CH05_MATERIAL_AUDIT.md','''# 第五章材料盘点与写作就绪性评估

日期：2026-11-20。任务：CH05。**最终决策：可以直接进入正文写作。** 本句是就绪性判断，不是正文写作授权；本轮结束后等待用户与ChatGPT审核。

## 十项结论

1. **已有材料足以形成完整、可信的第五章初稿**。证据主线为“观测差异→无同步配对的监督缺口→Real统计GMM引导实际球形探针→独立双域VAE→GRU Translator→完整712D输出→冻结评价及失败归因”。不能把“链路实现”写成“迁移目标完成”。
2. **不需要新增实验才能写可信初稿**。现存代码、有效配对数组、45轮训练日志、冻结权重、逐实例CSV和后续失败诊断足够支持限定结论。缺少物理真值等事项可直接作为局限，不以本任务新增研究补齐。
3. **5.3、5.4、5.5材料最充分**；5.6有完整指标与失败证据但最易混用版本。5.1的历史差异数据充分，受任务构成/采样合同影响；5.2缺少认证共同物理语义，恰是问题定义的限制，不需虚构条件补足。
4. **不能说统计GMM潜力已充分挖掘**。责任度确实已用于统计分解，而分阶段先验、分量信息、保留的候选多解尚未作为Translator概率监督利用。这里只确认可用信息与实施缺口；没有证据证明软监督必然更好。C25否决独立潜坐标间直接OT，不能否决统计GMM探针构造。
5. **Dual-VAE完成各域独立表示和域内重建；Translator完成有界伪配对上的候选跨域映射**。二者不提供天然公共潜坐标，不保证Natural支持覆盖，也没有物理配对真值。C13仅修Real幅值条件路径，C27仍使用原C3双VAE，不可合并成同一升级模型。
6. **初始跨域重建未通过原冻结C0联合验收**。本轮由已有Validation CSV复核：原C3映射（C11-F复现）联合1/11，C27联合2/11；两者D均11/11。C3最终报告的1/11来自另一历史开发Test角色，不能以它替代Validation。C23的6/11、C24的2/11都是Real自重建读出诊断。
7. **应正文保留**：C3主链、C4覆盖/监督歧义、C13/C23目标侧读出限制、C24时序改善与质量退化并存、C27专项与联合口径区别。C5/C6用于解释配对限制；C7–C9压缩为物理读出归因，不能另写一条已成功迁移路线；C10/R1、C25/C26及P28宜短述或汇入限制表。
8. **建议8000–9000 TeXcount，中心8500**。按本轮七节预算非平均分配，见结构报告。不是本轮已写字数，也不靠罗列全部实验凑篇幅。
9. **第五章到初始潜映射及其局限为止**。P44共享向量场、P45梯度路径、P46–P48质量评价、P49/P50对抗及退化、P51可信度NO-GO留第六章。K为Euler积分步数，不是对抗训练轮数。P52/P53未执行。
10. **核心证据缺口仍存在，但不阻断诚实写作**：Real硬件映射与同步配对、统一共同条件、Natural的正确Real目标、独立新盲验、真实任务收益均未获认证；因此不能写“实现物理精确迁移”“全部11联合通过”“对机器人成功率有提升”。

## 本轮实际复核范围

只读代码、已有Train/Validation配对数组及字典、CSV、注册表和模型Hash。未加载Test/Holdout样本或预测文件，不访问服务器，不运行模型推理、训练或物理仿真。历史报告中的Test汇总只用于辨认结果角色，不用来挑模型/配置，不另行复算Test。

直接读取paired_train/validation各sim与real数组，确认1200/240个完整4×712窗口、finite、非负及各自数值唯一；字典480行、16 Patch、160种登记放置后初态指纹。来源CSV重计：Train1200源/566Trial/30Session，Validation240源/106Trial/4Session；两组机器人reset各仅一种，19种放置状态和17种拓扑跨组复用。**1440数值窗口不等于1440独立Root。**

核对原C0全部11实例、Scaler及C3三个转换权重实际SHA256与冻结登记相符；C27验收包的Real/Sim VAE和GRU权重亦与对应源一致。逐实例重算既有CSV中BA、D与联合判据，并由混淆矩阵核对BA；没有重新推理或拟合AUC。数字选择器、Seed、数据角色、分母及允许表述见claim_cards.csv。

## 写作时保持的三条区分

统计GMM的索引空间不是VAE潜空间；独立Encoder latent维数相同不是共同语义坐标；被冻分类器判为Real不是分布质量与输入结构同时通过。全局稀疏率接近Real也不意味着Taxel激活位置一致。

审计全过程与代码起止行、参数来源、Hash见附属JSON。L1报告记载、L2直接复核和UNKNOWN分别保留；本轮不产生L3证据。
''')

write('CH05_GMM_PSEUDOPAIR_AUDIT.md','''# GMM统计伪配对专项审计

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
''')

protocol='''# 冻结C0协议与指标复核

## 1. PRIMARY与SECONDARY

PRIMARY是C3在**原Real Train5153窗/32Session及Natural Train4049窗/50Root**上训练后冻结的11个C0：Logistic、kNN；MLP/GRU/Transformer各Seed11/12/13。冻结前数据共9202训练窗；当前开发Validation为780Real/4Session、905Natural/10Root。所有C0用同一pooled Train的log1p均值/标准差（std下限0.05），坐标为4×712，不重排通道，不按域重拟合。

C3/code/c0.py第15–16行给出标签与Scaler，c0/training_registration.json登记架构、epoch与Hash；c0/frozen.json列11实例和权重、kNN参照及ScalerHash。本轮逐文件Hash核对通过。常规深度预算20–100Epoch、patience15、AdamW lr1e-3/wd1e-3；原CPU续训路径记录保留，不能假定所有模型都在同一设备完成。

Real=1、Sim=0，概率p为Real分数，判定p≥0.5。common.metrics来自P37/code/common.py第23–27行：BA=(RecallReal+RecallSim)/2；AUC用正类Real的平均秩统计，平分数使用average rank；D=1−2|BA−0.5|。原门槛BA∈[.45,.55]、AUC∈[.475,.525]、D≥.9。BA门槛与D门槛数学等价，不是两份独立证据。AUC<.5不能改为1−AUC参加正式验收。

混淆矩阵行为实际0/1、列为预测0/1：[[Sim判Sim,Sim判Real],[Real判Sim,Real判Real]]。C23/C24中负类实际是重建Real，名称是读出诊断标签，不能将其重新命名为Natural。

SECONDARY是在Real与迁移Sim上重新拟合的Probe。P37/P39的表示分解与C14的freshly_trained结果属于这种问题，不能替代PRIMARY。特别核对C14/probe_metrics.csv，frozen_C0为False；C13原冻结结果另在`outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv`，已与C23历史对照逐值交叉核对。

## 2. 结果角色与可比性

|结果|负类来源/分母|联合通过|可支持结论|
|---|---|---:|---|
|C0原始Validation|905 Natural原始 vs780 Real|0/11|原域分类能力强，不是转换成绩|
|C3原GRU在C11-F复现|同905 Natural迁移 vs780 Real|1/11|初始转换未完成联合验收|
|C13修幅值后Natural|同905 Natural迁移 vs780 Real|1/11|此幅值修复没有带来全联合通过|
|C23|780重建Real vs同780原Real|6/11|目标侧读出能力改善但仍不完整；非迁移6/11|
|C24|同780重建Real vs同780原Real|2/11|目标侧最小时序修复不足，不能当Natural下降|
|C27|同905 Natural迁移 vs780 Real|2/11；D11/11|D专项达标，原联合未达标|

C3原final_evaluation及报告中的“1/11、765个迁移窗口”等属于历史开发Test角色（Real628/3Session、Natural765/10Root），不是上表Validation。本轮不读这些Test数组、预测文件或重新计算其成绩。只从历史报告辨认其角色和已公开结论；不能因通过数巧合相同而混为同一试验。

C11-F是原C3冻结链路的Validation复现；本轮仅核验已有标量CSV，不新运行模型。C27仍用原C3 Real/Sim VAE，未换成C13修复版。C27验收包三个模型与源权重Hash匹配，scope文件明确AUC不纳入专项、原联合2/11、非新盲测。

## 3. 全部实例实际Validation结果

以下由原CSV读取并重算D及联合判定；BA同时用保存混淆矩阵复核，未重新拟合概率或AUC。每条原路径/Selector/Seed/分母见claim_cards.csv；完整混淆矩阵见validation_metric_recheck.csv。
'''
for ex in metricsources:
 rr=[r for r in mr if r['experiment']==ex]
 protocol+=f"\n### {ex}\n\n来源：`{metricsources[ex]}`。筛选Validation"+('且group=F。' if 'C11F' in ex else '。')+'\n\n|实例|BA|AUC|D|联合通过|\n|---|---:|---:|---:|---|\n'
 for r in rr:protocol+=f"|{r['instance']}|{float(r['BA']):.6f}|{float(r['AUC']):.6f}|{float(r['D']):.6f}|{'是' if str(r['joint_pass']).lower()=='true' else '否'}|\n"
protocol+='''
## 4. Bootstrap和解释边界

旧C0开发指标与分组Bootstrap不构成全新盲测；C3的既有Test结果也已公开。C23/C24原与重建对应同一4个Session，Bootstrap必须同源成对重采样，不能当8个独立Session；仅4组的推断能力有限。本轮未新运行Bootstrap。

D高时若两域大多被判Real，BA会靠近0.5，但AUC仍可保留显著排序差异；这不是阈值需要被放宽的理由。高D、输出finite、输出数值不同和Patch总量接近均不足以单独证明完整迁移。P28 Real-only grouped-CV反映Session漂移，不是当前冻结C0的理论可达上界。
'''
write('CH05_C0_PROTOCOL_AND_METRIC_AUDIT.md',protocol)

matrix='''# 第五章逐节证据矩阵

本表只登记可写论点，不把某行代码存在当作有效研究结果。附属claim_cards.csv记录具体数值选择器。

|小节/论点|实际代码位置或模型合同|已有结果/原始产物|本轮级别|允许表述与限制|
|---|---|---|---|---|
|5.1 Mask/幅值/Full均残留差异|P39/code/run.py::prepare及表示构造；P37/common.metrics|P39/mask_statistics.csv、amplitude_statistics.csv、probe_metrics.csv、temporal_statistics.csv|L2 CSV与代码|旧P37开发差异；不是新C0主验收，非完整因果分解|
|5.1 幅值诊断表示实际是全帧正值分位数|P39/code/run.py 18–23行，registration|4×5正值P10/P25/P50/P75/P90，padding接口；representation_counts.csv|L2|不是每Patch固定分位数；排除含全零帧窗口，因此其分母不同，不能直接作信息量减法|
|5.1 原生touch与processed不同|source/sensors/tactile/dex_hand.py::augment_spec/bind/read_raw/read；signal_processing.py::process|C2/sensor_metadata与taxel_geometry_mapping；第二章已核验机制|L2代码，历史执行L1|仿真内部712通道与16Patch可追溯，Real物理方向仍未认证；不重写第二章|
|5.2 三类数据与无物理配对|C3/common.py::real/natural；pipeline_cpu.py|Real5153/780、Natural4049/905、Paired1200/240，注册/来源CSV|L2角色、配对数组|Natural参与Sim VAE且为部署输入；Paired用于Translator，不是同步物理接触|
|5.3 GMM仅Train拟合、K/BIC/稳定性|GMM/code/fit_gmm.py::samples及主流程；gmm.py::fit_em|gmm_bic.csv、gmm_parameters.npz哈希链|L2|二维索引偏移先验；不冒称力场/压力重建|
|5.3 软分量到单一物理候选|C3/code/expand.py、inverse.py、matching.py::inherited_metrics|inverse_gmm_*、pair_expansion_manifest.csv、paired_train/validation.npz|L2代码与指定数组|责任度已用；top5和单一最佳候选压缩多解，无后验加权Translator监督|
|5.3 480字典不等于480独立Root|C3/code/dictionary.py及physics.py::reset/run|dictionary_manifest.csv、dictionary_responses.npz|L2|16×5×3×2，放置后指纹160；仿真内部映射|
|5.3 来源交叉/加载饱和|C4/code/pairs.py、input_ambiguity.py；C3/expand.py深度段|paired_source_inventory.csv、inverse_control_saturation.csv|L2 CSV重计|1200/240数值唯一，但机器人reset复用、近拓扑交叉；不写全局无泄漏|
|5.3 C5/C6未解决全部覆盖|对应code/与summary|C5完整与同目标分母；C6实际S1只有1次，S2未执行|L1 summary，L2实现|局部改善、支持不足；不证明五球理论无效，不宣称扩展成功|
|5.4 两套独立VAE|C3/models.py::Encoder/Decoder/DomainVAE；pipeline_cpu.py::train_vae|各history.csv、complete.json、checkpoint.pt及converter_frozen.json|L2|4×64同维、无共享权重/天然语义；域内重建不等于迁移|
|5.4 C13只修幅值上下文|C13/code/run.py、registration|training_summary、checkpoint、C23/reconstruction.json|L2|594512可训练参数，Encoder/Mask冻结；不改称新联合共享VAE|
|5.5 GRU潜变量映射及冻结|C3/models.py::Translator、pipeline_cpu.py；inference.py::Converter.transform|translator/history.csv、complete、冻结权重|L2|Huber+0.1 RealDecoder NLL；两个Encoder及Decoder冻结，梯度可经冻结Decoder回传到Translator|
|5.5 Paired支持不足、监督歧义|C4/code/latent.py、input_ambiguity.py|distance_summary.csv、input_mapping_ambiguity_summary、paired_same_targets_oracle_vs_translator.csv|L2 CSV/代码|同一Sim Encoder内可比较Natural/Paired，不能跨两个独立Encoder裸欧氏匹配|
|5.6 原11冻结C0协议|C3/c0.py、P37/common.py::metrics；各后续评价器脚本|c0/frozen.json、scaler、11checkpoint、Validation指标|L2 Hash与CSV|PRIMARY/SECONDARY严格分开，D专项/原联合分开|
|5.6 C23/C24机制对照|C23/evaluate.py；C24/code/train.py及registration|C23/metrics、reconstruction；C24/metrics、Real_C13_C24_structure.csv|L2|6→2仅同源Real自重建读出，不是Natural通过数变化|
|5.6 C25/C26停止|C25/code/audit.py；C26/code/run.py/evaluate.py|coordinate_gate、decision、C0_complete|L2元数据与代码|OT未执行；C26 C0未运行，不能写0/11|
|5.6 C27专项边界|C27/config、candidate checkpoint与release models|metrics.csv、acceptance_scope.json、实际模型Hash|L2|D11/11、联合2/11；固定原双VAE，不混C13|
|5.7 引向第六章|P44–P51各报告仅边界核对|P51 NO-GO及P52/P53未执行|L1|提出后续研究问题，不预告对抗已解决|

## 可以复用的定量差异材料

'''
mask=[r for r in d['P39_mask_ALL'] if r['candidate'] in ['Real','raw']]
matrix+='|角色|域|窗口/组|精确正值比例|\n|---|---|---:|---:|\n'+''.join(f"|{r['role']}|{r['candidate']}|{r['windows']}/{r['groups']}|{float(r['activation_fraction']):.6f}|\n" for r in mask)
matrix+='\n同一P39统计口径Train正值中位数：Real0.00500250，旧原Sim0.0000103974；P95分别1.5010055、0.4594847。数值只支持该旧数据版本的描述差异。P39校准前后Mask比例完全相同是已有一致性检查结论，不能由总体比例相同单独推导逐位一致。\n\n'
matrix+='P39各表示最差D（11实例）由原CSV复算：\n\n|表示|最差D|BA范围|\n|---|---:|---|\n'
for name in ['mask_raw','mask_before','amplitude_raw','amplitude_before','amplitude_after','full_raw','full_before','full_after']:
 if name in d['P39_probe_ranges']:
  r=d['P39_probe_ranges'][name];matrix+=f"|{name}|{r['D_min']:.6f}|{r['BA_range'][0]:.6f}–{r['BA_range'][1]:.6f}|\n"
matrix+='''
这些表示有相同五架构与深度固定Seed，但分母/特征内容/预处理各自拟合；Amplitude剔除含全零帧窗口。不能将不同表示D相减解释成Mask贡献百分比。P40是共享C特征4×64，没有新712D Decoder；所选候选最差D0.2427仅历史报告结果，不能当本章完整712D重建成绩。

## 图表来源建议

1. 差异图优先从P39/mask_statistics、amplitude_statistics、temporal_statistics按Train/Validation单独绘制；用16个Patch方阵展示，不把712通道全拉平替代空间图。
2. 物理链图可用C3字典manifest/plan与C2仿真内部映射；取指定Train/Validation实例，非自然实机对应图。
3. 模型图按C3/models.py画训练/推理分离图，标C13与C27支线版本；不得拼合参数形成不存在的最终模型。
4. 覆盖图从C4/distance_summary中仅Train/Validation重绘。原PCA/示例PNG可能混有历史Test，未经逐图来源核实不得直接用于正文。
5. 评价表使用本目录Validation逐实例CSV；C3 historical Test不取样本或预测，不用旧Test示例冒充Validation。

所有图本轮仅规划，不生成图、不编译、不写正文。既有可视化路径是否可以直接使用需以元数据核实，不能只看图名。
'''
write('CH05_EVIDENCE_MATRIX.md',matrix)

# Append the directly checked representation and learning contract to main audit.
p=O/'CH05_MATERIAL_AUDIT.md'
with p.open('a') as f:
 f.write('''
## 双域表示和Translator的实际训练合同

C3/models.py的Encoder接收[B,4,712]，拼接精确正值Mask与log1p(x)/encoder_scale形成每帧1424维，经Linear1424→128、GELU、LayerNorm和GRU128，分别输出4×64的mean、logvar（后者限[-8,4]）。训练以mu+exp(logvar/2)·epsilon重参数，推理取posterior mean。两个域模型分别创建、分别拟合尺度和训练，不共享参数。

Decoder按帧自回归：前帧Mask的32维嵌入（首帧BOS）与当前64维latent一起进入GRUCell96→128；Mask头输出712logits，推理以logit≥0作MAP。Amplitude以128维隐状态与当前Mask的32维嵌入为条件，为每Taxel输出两个Gamma分量的混合logits/shape/rate；推理取混合均值乘Mask。当前C3已经是Mask自回归机制，不能照搬更早阶段“每帧独立Bernoulli”诊断。

VAE损失为Mask BCE与正值Gamma-mixture NLL（每窗按正值数归一化）加0.001 KL。NLL包含positive_scale的Jacobian，似然数值可为负，不能用其符号判断训练失败。训练NLL使用目标Mask上下文，部署自由运行使用预测Mask，C13专门修复其中的幅值条件差异；Teacher Forcing损失不能当自由输出质量。

''')
 f.write('|模块|Seed|实际Epoch/最佳Epoch|Steps|参数量|训练来源|\n|---|---:|---|---:|---:|---|\n')
 for name in ['real_vae','sim_vae','translator']:
  c=d['training'][name]['complete'];f.write(f"|C3 {name}|{dict(real_vae=4301,sim_vae=4302,translator=4303)[name]}|{c['epochs']}/{c['best_epoch']}|{c['optimizer_steps']}|{c['parameters']}|"+dict(real_vae='5153 RealTrain',sim_vae='4049 NaturalTrain+1200 PairedTrain',translator='1200 PairedTrain→对应Real目标')[name]+'|\n')
 f.write('''
Real每Epoch81step；Sim每batch32Natural+32Paired、每Epoch127step；Translator batch64、每Epoch19step。Sim混合采样重复遍历Paired并补足Natural批，完整覆盖不等于每个样本恰好访问一次；原高级索引计数纠正不改变训练权重或模型。C3三个模块均45轮、按额外ep≥30限制只允许后期checkpoint，均选30；Real历史最低Validation NLL出现在第4轮，早期权重未保存，不能回溯声称其分类成绩会更好。

Translator是两层GRU64→128加Linear128→64，181824参数。两个VAE全部冻结后，以Paired Sim的Sim Encoder posterior mean监督到对应Real Encoder的mean，L=Huber(T(z_s),z_r)+0.1·NLL(D_r(T(z_s)),x_r)。冻结Decoder不等于切断对输入latent的梯度；只有Translator更新。部署链是Natural processed→Sim Encoder mean→GRU→Real Decoder→4×712；不使用Real窗口、GMM、探针、域标签或Sim Decoder生成最终输出。

**独立潜坐标并不使配对监督Translator在数学上无效**：它恰是在学习两套坐标之间的函数。被C25阻断的是未证明公共坐标就直接对两套latent算欧氏/W2并做OT，不是Huber监督本身。映射可靠性仍依赖伪配对是否携带可辨识信息。

C4/translator_paired_error.csv显示Paired Train/Validation Huber约0.25565/0.32592，优于均值参考，说明学到了一定配对关系；Natural没有正确Real目标，不能给其latent RMSE。C4的近输入—远Real目标现象支持监督歧义风险，不是网络完全没训练的证据，也不是更大网络必然有效的证据。

C13不重建新架构，只训练decoder.current及decoder.amplitude共594512参数；Seed4313，实际18轮、最佳3轮、1458step，Real5153/780。Encoder、Mask、循环骨干及Sim VAE/Translator仍冻结。C23仅对这一版本已有Real重建输出做冻结C0读出。C24延续同一最小参数集合加保留帧增量损失，未得到无损改进。

C4的Sim positive_scale=1e−7来自NaturalTrain+PairedTrain正值中位数8.78499e−9触发下限；Encoder实际采用log1p(x)/0.04088746，不能把似然缩放当Encoder输入直接放大。NaturalTrain与PairedTrain正值中位相差约七数量级，存在数值条件风险，但没有受控证据证明它是唯一或主要根因。完整量纲、Jacobian和极端值记录见C4/scale_formula_audit.json、amplitude_scale_audit.csv。
''')

write('CH05_STRUCTURE_AND_WORD_BUDGET.md','''# 第五章结构、材料与篇幅预算

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
''')

write('CH05_OPEN_ISSUES.md','''# 第五章重要缺口、版本差异与写作禁区

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
''')

# Focused numerical mechanism cards and registered stop cases.
with (O/'CH05_C0_PROTOCOL_AND_METRIC_AUDIT.md').open('a') as f:
 f.write('''
## 5. 目标读出和替代路线的限定机制结果

C24/Real_C13_C24_structure.csv在同780 Real窗上：C13配对L_temporal=0.0028149674，C24=0.0016136414，按1−C24/C13计算约42.68%降低；MSE从0.026319602升至0.027157449（约+3.18%）。Mask逐位不变的登记与IoU0.51171548一致，但不能仅由IoU相等反推逐位相同。平均绝对帧变化0.0055292053→0.0025255096，Real为0.00063119946，约8.76×→4.00×；P95和P99下降伴样本差异收缩，联合通过6→2。C24/stage0.json确认四帧按原保留完整帧顺序、未跨Trial/Stage，Real并非均匀20Hz；这是索引增量一致性，不是物理速度误差。

C25/coordinate_gate.json、decision.json：两个独立4×64表示不能直接共坐标比较，cross-native距离与OT门槛未通过。文件gmm_model_selection.csv、ot_plan_audit.csv存在不代表已实际拟合/求解；状态为NOT_EVALUATED。其限制不否定C3受监督Translator的坐标映射。

C26/decision.json、C0_complete.json：共享C13 Encoder建立共同数值坐标，不提供共同物理语义；输出输入保持条件失败后停止，C0新推理0次、joint_pass_count=null，不写0/11。

C10-R1/decision.json与paired_metrics.csv：Seed11，PairedVal1200/240中的240验证窗，GRU latent MSE0.7525827、decoded MSE0.08118970、IoU0.4966360；v-prediction八个单样本误差均值为1.1580479、0.05348208、0.4879714。latent误差更大、解码MSE变小而IoU略低，预注册的三项共同改善门槛未通过；不选择best-of-K。八样本平均输出仅诊断，不代表单次MAP生成。原epsilon实验爆炸与v采样稳定性改善分开解释；该固定配置停止，不泛化为所有扩散模型无效。本轮不加载任何Natural Test，R1历史亦无Natural/C0评价。

C7–C9的连续场、Taxel积分与处理诊断只解释空间可分辨性，不是新Translator成绩。C9已有29多接触执行、330 Patch帧归因至25命令配置/16放置状态/1reset，Root独立性UNKNOWN；连续支持与最终processed的四连通指标不同，不把多帧当独立样本。具体采用哪张图须另核实其物理量单位和来源，不混入原C3触觉分布主表。
''')
print('wrote reports and',len(cards),'numeric cards')

# Additional claim cards for every mechanism summary directly proposed for the chapter.
for row in d['C10R1_validation_rows']:
 if row['aggregation'] in ['DETERMINISTIC','EXPECTED8_SINGLE_ERROR']:
  for key in ['latent_MSE','decoded_MSE','Mask_IoU']:
   card('C10R1_'+row['method']+'_'+key,'outputs/tactile_p43_c10_r1_v_prediction_2026-11-17/paired_metrics.csv',f"role=validation;method={row['method']};aggregation={row['aggregation']};{key}",row[key],'Paired Validation','240窗；随机项8固定单样本误差均值','有限配置结果，不泛化所有扩散',11)
for key in ['tactile_p43_c5_loading_calibration_2026-11-17','tactile_p43_c6_spatial_configuration_2026-11-17','tactile_p43_c9_taxel_resolution_limit_2026-11-17']:
 for k,v in d[key].items():
  if isinstance(v,(int,float)) and not isinstance(v,bool):card(key+'_'+k,'outputs/'+key+'/summary.json',k,v,'Train/Validation或历史诊断','按对应summary固定分母','历史已保存诊断，不是本轮物理执行',level='L1')
for role in ['train','validation']:
 for k,v in d['paired_independence_recount']['cross_split'].items():
  if role=='train':card('paired_cross_split_'+k,paths['C4']+'/paired_source_inventory.csv','intersection(train,validation):'+k,v,'Train/Validation','1200/240来源记录','只检查登记字段的直接交叉，不证明全局无泄漏')
for key in ['C13_training','outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/training_summary.json']:
 for k in ['epochs','best_epoch','optimizer_steps']:
  card(key+'_'+k,paths['C13']+'/training_summary.json' if key=='C13_training' else key,k,d[key][k],'Real Train/Validation','5153/780窗，32/4Session','历史实际幅值支线训练',4313)
for r in d['P39_amplitude_ALL']:
 if r['candidate'] in ['Real','raw'] and r['role']=='train':
  for k in ['positive_P50','positive_P95']:
   card('P39_'+r['candidate']+'_'+k,paths['P39']+'/amplitude_statistics.csv',f"role=train;candidate={r['candidate']};stage=ALL;patch=ALL;{k}",r[k],'Train',r['windows']+'窗/'+r['independent_groups']+'组','原始processed正值幅值分位数，非共同力学量纲')
for name,val in [('temporal_loss_reduction',1-float(next(r for r in d['C24_structure'] if r['source']=='C24')['L_temporal'])/float(next(r for r in d['C24_structure'] if r['source']=='C13')['L_temporal']))]:
 card('C24_'+name,paths['C24']+'/Real_C13_C24_structure.csv','1-L_temporal(C24)/L_temporal(C13)',val,'Real Validation','同780窗/4Session','保留帧索引增量误差降低，非物理每秒导数',4313)
with (O/'claim_cards.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(cards[0]));w.writeheader();w.writerows(cards)
# Small corrections based on actual definitions, not prose polishing.
p=O/'CH05_EVIDENCE_MATRIX.md';p.write_text(p.read_text().replace('P39/code/run.py::prepare及表示构造','P39/code/run.py::features/pool及表示构造'))
p=O/'CH05_C0_PROTOCOL_AND_METRIC_AUDIT.md';p.write_text(p.read_text().replace('混淆矩阵行为实际0/1','点估计为窗口级的类平衡BA及窗口分数AUC，并非Session/Root等权宏指标；训练时分组平衡Loss不能代替评价端的组平均。Bootstrap才按既定Session/Root分组重采样。\n\n混淆矩阵行为实际0/1'))
print('final numeric cards',len(cards))
