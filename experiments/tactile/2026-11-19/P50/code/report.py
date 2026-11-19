from audit_utils import *
import torch
P49=Path('outputs/tactile_p49_round_frozen_adversarial_2026-11-19')
q=json.loads((O/'quality_summary.json').read_text());dec=json.loads((O/'decision.json').read_text());comp=json.loads((O/'complete.json').read_text());infer=json.loads((O/'inference_audit.json').read_text());backend=json.loads((O/'initial_backend_audit.json').read_text());a=list(csv.DictReader((O/'acceptance_metrics.csv').open()));d=list(csv.DictReader((O/'D50_metrics.csv').open()));grad=list(csv.DictReader((O/'branch_gradients.csv').open()));params=list(csv.DictReader((O/'parameter_inventory.csv').open()))
def row(n,b='Joint'):return next(v for v in a if v['model']==n and v['branch']==b)
def md(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in rr)+' |' for rr in rows)+'\n'
j=row('F50');dec['primary_acceptance_pass']=dec.pop('joint_FR_pass',False);dec['Joint_Root_FR_at_least90']=float(j['FR_macro'])>=.9;dec['Real_joint_acceptance_at_least90']=float(j['Real_acceptance_macro'])>=.9;save(O/'decision.json',dec)
name=['Original','F0','P49_F3','F50_without_refiner','F50'];dist=md(['输出','SWD↓','MMD↓','PRDC P@5','PRDC R@5','跨窗口Std','HSIC'],[[n,f"{q[n]['SWD']:.6f}",f"{q[n]['MMD']:.6f}",f"{q[n]['PRDC']['5']['precision']:.6f}",f"{q[n]['PRDC']['5']['recall']:.6f}",f"{q[n]['structure']['window_std']:.6g}",f"{q[n]['HSIC']:.6f}"] for n in name])
accept=md(['分支','F0 FR','F50 FR','Real接受率','F50 AUC','混淆矩阵（Sim/Real）'],[[b,f"{float(row('F0',b)['FR_macro'])*100:.3f}%",f"{float(row('F50',b)['FR_macro'])*100:.3f}%",f"{float(row('F50',b)['Real_acceptance_macro'])*100:.3f}%",f"{float(row('F50',b)['AUC']):.6f}",row('F50',b)['confusion_matrix']] for b in ['Global','Spatial','Temporal','Joint']])
dtab=md(['分支','fit BA','inner BA','inner AUC','inner Real Recall','inner Sim Recall','CE差'],[[b,*[f"{float(v):.6f}" for v in [next(v['BA'] for v in d if v['branch']==b and v['role']=='fit'),*(next(v for v in d if v['branch']==b and v['role']=='inner')[k] for k in ['BA','AUC','Real_recall','Sim_recall','CE_gap'])]]] for b in ['Global','Spatial','Temporal']])
neg=md(['负对照','Joint FR','完整护栏通过','HSIC','跨窗口Std'],[[n,f"{float(row(n)['FR_macro'])*100:.3f}%",q[n]['all_quality_pass'],f"{q[n]['HSIC']:.6f}",f"{q[n]['structure']['window_std']:.6g}"] for n in ['Constant','RealDonor','PairShuffle']])
gates=md(['原P49质量护栏','F50'],[[k,'通过' if v else '失败'] for k,v in q['F50']['gates'].items()])
rt=md(['输出','零比例','正值P50','P95','P99','最大值','lag1','帧差'],[[n,*[f"{q[n]['structure'][k]:.7g}" for k in ['zero_fraction','positive_P50','positive_P95','positive_P99','maximum','lag1','frame_delta']]] for n in name])
text=f'''# P50：残差触觉重建与多视角判别器单轮实验

日期：2026-11-19。Seed=5001，K=1，仅一次D→G。所有结果属于既有开发Validation，非盲测。

## 1. 结论

**分类C：本轮架构改进失败，不进入下一轮。**

F50在Global、Spatial、Temporal与Joint上的Root Macro FR都达到100%，但原始Real Validation的Joint Session Macro Acceptance只有{float(j['Real_acceptance_macro'])*100:.3f}%，未达到90%。F50的SWD从F0的{q['F0']['SWD']:.4f}升至{q['F50']['SWD']:.4f}，MMD从{q['F0']['MMD']:.4f}升至{q['F50']['MMD']:.4f}。完整护栏没有通过，不能报告有质量约束的0.90迁移成功，也不应只按“FR达标”归为成功。

关闭终点Refiner、不改向量场、不重新训练，SWD为{q['F50_without_refiner']['SWD']:.4f}，MMD为{q['F50_without_refiner']['MMD']:.4f}，均明显好于F50；但仍差于F0。说明当前终点残差修正在分布质量上产生了负作用，不能通过删除该分支并换主模型来规避本轮失败。

## 2. 数据及冻结边界

Real Train为5153窗、32Session，Real Validation为780窗、4Session；Natural Train为4049窗、50Root，Validation为905窗、10Root、五任务各2Root。完整Trial/Episode来源沿用P49 data_inventory及原注册数据；本轮不把窗口当作独立物理样本。沿用P49的24/8 Real Session及40/10 Sim Root内部隔离，组集合交叉为空。内部验证只诊断D，F0历史上使用过Train，不能称为生成器盲测。

只读取注册Train/Validation，没有Test/Holdout、C0推理或生产目录操作。原始数据、P44–P49权重和旧验收包均保护；最终Hash见integrity.json。输入输出保持float32[N,4,712]，窗口ID、Root、Task、Stage与帧/通道顺序不变。

## 3. 架构与实现审计

### 3.1 原生成器及Refiner

F0加载P45 Continuous-only A_K1；Dual-VAE和Real Decoder冻结，只训练原共享field及新增Refiner。基础模型2494640参数，原field306176参数；Refiner120968参数，总可训练生成器427144参数。Refiner输入每帧log1p(Y0)712维与冻结Sim posterior mean64维，776→64 GELU→GRU64→Linear712。不是Real donor，也没有推理时读取Real。

设l=log1p(Y0)，r=Rψ(Y0,z)，实际输出：

`delta = expm1(max(l+r,0)) - expm1(l)`

`Y1 = max(Y0+delta,0)`

最后线性层全零初始化。相同后端下零残差输出逐值等于Y0；采用连续非负log响应残差，不增加可搜索阈值。max只用于非负参数化与浮点误差防护，未按Validation选择截断值。边界采用PyTorch clamp_min默认子梯度；数值边界局限见第9节。新模块不是独立无条件生成器。

F0 Train和Validation按历史CPU路径重新生成，与P49保存数组逐值一致。训练及最终推理沿用CPU Sim Encoder缓存、CUDA field/Decoder；初始CPU/CUDA基线差异另完整披露，不暗称跨后端逐位相同。

![生成器](figures/generator_architecture.png)

### 3.2 三分支连续判别器

Global：log1p完整四帧2848维；Spatial：每帧每个固定索引Patch的均值、二阶矩、最大值，共192维；Temporal：相邻帧log1p差，共2136维。三支分别为Linear128→LayerNorm→LeakyReLU→Linear64→LeakyReLU→1，共688643参数，少于P49单一全局D的1106625参数。三支BCE等权平均，没有删除弱分支。

16Patch切分为`[32,32,56]*5+[112]`，总和712，直接复用P49已登记offset。这里只证明索引Patch成员关系，没有证明Taxel真实邻接或物理坐标。max特征是分段可微，未将硬二值Mask作为判别输入。Temporal只描述四帧索引关系，Real采样不均匀，不能解释为物理时间校准后的动力学。

逐分支Smoke均有到field和Refiner的有限非零梯度，输入梯度覆盖712通道；详见smoke.json。原基础生成器训练路径沿用P49的hard-forward及原有straight-through内部实现，不偷偷修改其Mask生成机制。

![判别器](figures/discriminator_architecture.png)

## 4. 预算、训练与恢复

D固定10Epoch、660步；G固定20Epoch、1280步，全部Natural Train按每轮打乱后完整遍历。D fit每Epoch66个batch，两个域各64，组等权Loss、类别等权；短域打乱循环，所有fit窗口均覆盖。没有按Validation挑Checkpoint，主模型严格为Epoch20。G共访问4049×20=80980个源窗口；D Real fit4217、Sim fit3249，通过循环补齐各4224窗口访问/epoch。梯度只来自Train。

Adam lr1e-4、betas(0,.99)、clip5；D沿用R1 gamma0.1、每16步一次，三支R1取平均。G损失为三支等权BCE +0.5295653698327406×原Patch份额/相对帧差preserve +0.001×原latent std归一化flow。没有新增分布损失、数据重标度、Seed或权重搜索。

累计CUDA同步优化墙钟{comp['optimization_seconds']:.3f}秒，GPU峰值分配{comp['GPU_peak_bytes']/2**20:.1f}MiB。此时间只包含包围优化步骤的同步墙钟，包括梯度监测，不包含数据准备、逐步Checkpoint磁盘写入和全部离线统计；不是整个任务耗时。训练快不表示使用了更大数据，也不据此声称模型已收敛，见实际规模、步数与曲线。

每步原子保存resume.pt，包含下一步索引、完整模型/Adam/RNG、累计预算与日志；每Epoch保存模型。已完成步骤不重复。本轮正式优化没有恢复或重训。开始前修正了脚本语法问题及基线CPU/CUDA比较路径，均在第一步优化前；代码与历史保护没有混用。D50所有参数在G阶段冻结，前后状态Hash均为`{comp['D_frozen_hash']}`。原VAE状态逐位不变。

![训练曲线](figures/branch_training_curves.png)

## 5. D50判别能力与梯度

{dtab}

CE差距只诊断，没有0.2硬停止条件。三个分支概率均非恒定、两类都有预测，fit/inner无组交叉。Global的内部Real Recall约0.680，是明确的泛化薄弱项；Spatial/Temporal在外层Real Validation同样有误拒，不能把FR高解释为判别器全面可靠。

branch_gradients.csv记录每Epoch首batch三支对field/Refiner的梯度；均完整保留，不能仅看平均总梯度。梯度非零支持“新分支确有可用梯度路径”，不证明它们学到了充分物理结构。没有单独移除分支重训，不能宣称某一个分支的因果收益或声称空间/时序已经改善。

![梯度](figures/branch_gradients.png)

## 6. 同一个D50下的接受率

{accept}

固定0.5阈值，Real=1。Joint要求三支同时接受，Joint连续分数仅为min(logit)，用于辅助AUC。混淆矩阵行真值Sim/Real，列预测Sim/Real。所有三支都接受了F50，因此不是靠某支未参与Joint计算获胜；但这些输出的分数严重偏向Real一侧，AUC甚至接近0。没有改变正式AUC或移动阈值。AUC不作为FR单项否决，失败来自明确Real接受门槛及完整质量护栏。

冻结D下Real接受率与生成器无关，更新前后完全相同。F0 Joint FR为{float(row('F0')['FR_macro'])*100:.3f}%，F50为100%；Real Joint仅68.083%，不能靠更新F修复D对Real的误拒。完整逐Root/Session数据见acceptance_groups.csv；1000次组Bootstrap沿用P49/P48固定索引。样本全部接受时经验Bootstrap为[1,1]，不代表总体无不确定性。

![接受率](figures/acceptance_comparison.png)

## 7. 完整质量与输出结构

{dist}

{gates}

F0首先进入完全复用的P49 Evaluation，固定原参考；SWD使用原1024投影及Root/Session等权，MMD使用原RealTrain标准化与核；PRDC使用原P47的100次样本索引及k3/5/10/20，不把Monte Carlo重复视为独立采集。门槛未改。HSIC保持统计依赖，只是必要条件，不是触觉语义正确证明。

{rt}

![分布](figures/distribution_comparison.png)

Refiner启用/关闭是终点离线消融，不重新训练、没有参与选择。关闭时Joint FR为{float(row('F50_without_refiner')['FR_macro'])*100:.3f}%，启用后100%，但SWD/MMD明显恶化。当前残差模块更有效地提高了判别器Real分数，却没有带来分布质量改善。其响应域残差RMS约{infer['residual_response_RMS']:.6f}，最大绝对变化约{infer['residual_response_max_abs']:.6f}。这是对现有参数化、固定权重与本次Seed的反证，不能外推为所有Refiner必然失败。

Root/Stage结构、每Taxel激活、各Patch响应、幅值P1/P10/P50/P90/P95/P99、双方均正值变化、连续激活长度等见structure_metrics、extra_structure_metrics、patch_metrics和taxel_metrics。它们是离线诊断，没有据此改阈值或筛选样本。

## 8. 负对照

{neg}

Constant、RealDonor及PairShuffle全部被完整护栏拒绝。PairShuffle保持边际输出集合不变，故FR和分布距离理论上不变，主要由依赖检查揭示对应关系损失。FR不能单独认证输入保持；即使F50自身Root HSIC较高，也不能抵消分布质量严重恶化。

## 9. 数值与独立推理局限

完整F50输出finite、非负，窗口元数据保持。独立重新加载Checkpoint后，CUDA响应最大差{infer['CUDA_replay_max_abs']:.9g}，通过预登记的数值比较rtol1e-5/atol2e-5；但精确零Mask有{infer['CUDA_replay_mask_difference_entries']}个通道帧不同，涉及响应不超过{infer['mask_disagreement_max_response']:.9g}。当前残差差分算式在近零处存在浮点消去敏感性，不能声称精确零结构逐位可复现。没有为了消除这些差异添加阈值、微正数或重写终点输出。

初始CUDA与历史CPU F0还有{backend['initialCUDA_mask_different_entries']}个Mask差异、最大幅值差{backend['initialCUDA_vs_reference_max_abs']:.7g}，来自原自回归Decoder的后端敏感性。F0原参考按历史CPU逐值保留，主F50沿原注册训练后端；这限制严格跨后端比较，未将其隐藏为Refiner收益。

独立入口code/infer.py不读取Real或donor，仅接收float32 processed输入及F50 Checkpoint；推理示例见REPRODUCE.md。此模型为失败研究产物，不替换旧验收版。

## 10. 对任务问题的回答与停止

1. Refiner改善F0质量？没有；SWD/MMD和多个结构护栏恶化。
2. Multi-view是否提供更多有效梯度？三个连续分支均提供非零梯度，解决了原硬Mask判别输入无梯度的问题；“更多梯度路径”并未转化为质量改善，不能声称因果优于旧D。
3. 三支分别改善什么？能分别学习不同视角的域差异，并共同提高当前D接受率；没有证据支持它们改善了真实输出结构。
4. 是否有分支被忽略？参与Loss和Joint的三个分支均保留，梯度非零；但不能因此认定梯度强度平衡或判别特征充分。
5. 保留F0分布优势？没有。
6. Joint FR≥90%？是，100%；Real联合接受≥90%？否，68.083%。
7. 完整质量通过？否。
8. 进入多步对抗？**不建议直接进入。** 当前证据首先指向D的跨Session Real误拒和输出残差在无显式分布约束下的判别器分数投机；继续轮数没有得到本实验支持。若人工决定后续研究，只应先解决判别器有效性和残差质量约束这一受控问题，不在本轮新增模型、权重搜索或训练。

P50完成后停止。未启动P51，未commit/push，未操作生产服务器。
'''
(O/'REPORT.md').write_text(text)
save(O/'report_complete.json',dict(classification=dec['classification'],main_FR=1,Real_joint_acceptance=float(j['Real_acceptance_macro']),quality_pass=False,no_more_training=True))
print('REPORT COMPLETE',dec['classification'])

# Additional analytical audit is recorded separately, not a model candidate.
if (O/'loss_invariance_audit.json').exists():
 with (O/'REPORT.md').open('a') as fp:fp.write('\n解析约束缺口：preserve对整体幅值倍增不敏感，flow只约束原field、不约束Refiner。详见loss_invariance_audit.json；本轮不修改损失。\n')
