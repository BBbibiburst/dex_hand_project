from c0_support import *
from model import Generator,VectorField,Discriminator
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
assert (O/'evaluation_complete.json').exists();frozen=json.loads((O/'frozen.json').read_text());assert sha(O/'generator.pt')==frozen['generator_sha256'];assert sha(O/'discriminator.pt')==frozen['D_sha256'];m=pd.read_csv(O/'C0_metrics.csv');quality=pd.read_csv(O/'output_statistics.csv');d=pd.read_csv(O/'D_metrics.csv');hist=pd.read_csv(O/'training_history.csv');cfg=json.loads((O/'config.json').read_text());tr=json.loads((O/'training_summary.json').read_text());grad=json.loads((O/'gradient_path_audit.json').read_text());branch=pd.read_csv(O/'D_branch_ablation.csv');summ=[]
for name,g in m.groupby('candidate',sort=False):summ.append(dict(candidate=name,joint_passes=int(g.passed_joint.sum()),D_only_passes=int((g.domain_confusion>=.9).sum()),min_D=float(g.domain_confusion.min()),AUC_min=float(g.roc_auc.min()),AUC_max=float(g.roc_auc.max()),mean_abs_AUC_from_half=float(np.abs(g.roc_auc-.5).mean()),mean_Sim_predicted_Real=float((1-g.recall_sim).mean()),Real_reference_mean_recall=float(g.recall_real.mean())))
table(O/'C0_summary.csv',summ)
# Formal original Real reference must be identical across all seven comparisons.
for name,g in m.groupby('instance'):assert np.max(g.recall_real)-np.min(g.recall_real)<1e-12
ck=torch.load(O/'generator.pt',map_location='cpu',weights_only=False);G=Generator(**ck['scales']);G.load_state_dict(ck['model']);G.eval();G.requires_grad_(False);numfields=sum(isinstance(v,VectorField) for v in G.modules());assert numfields==1;originalR=torch.load(C3/'real_vae/checkpoint.pt',map_location='cpu',weights_only=False);originalS=torch.load(C3/'sim_vae/checkpoint.pt',map_location='cpu',weights_only=False);assert all(torch.equal(v,G.real.state_dict()[k]) for k,v in originalR['model'].items());assert all(torch.equal(v,G.sim.state_dict()[k]) for k,v in originalS['model'].items())
# Parameter hashes, not only serialized-container hashes.
import hashlib
def component_hash(state):
 h=hashlib.sha256()
 for k in sorted(state):h.update(k.encode());h.update(state[k].contiguous().numpy().tobytes())
 return h.hexdigest()
fieldhash=component_hash(G.field.state_dict());sharing=json.loads((O/'parameter_sharing_final.json').read_text());save(O/'parameter_sharing_report.json',dict(one_field_module=numfields,one_training_discriminator=True,shared_generator_checkpoint_sha=frozen['generator_sha256'],shared_D_checkpoint_sha=frozen['D_sha256'],field_state_sha256=fieldhash,parameter_count=cfg['trainable_field_parameters'],D_parameter_count=cfg['D_parameters'],K_call_audit=sharing,G_updates_by_K={str(k):288 for k in [1,2,4,8]},D_updates_by_K=tr['K_step_counts'],no_distinct_G1_G2_G4_G8=True,VAE_bit_identical_to_original_C3=True,deploy_source_encoder_once_decoder_once=True))
# Report nonmonotonic changes, no choosing K by C0.
pivot=m.pivot(index='instance',columns='candidate',values='roc_auc').reindex(['logistic','knn','mlp_11','mlp_12','mlp_13','gru_11','gru_12','gru_13','transformer_11','transformer_12','transformer_13']);pivot.to_csv(O/'AUC_comparison.csv');table(O/'checkpoint_selection.csv',[dict(selected_epoch=20,criterion='fixed final fully completed epoch',generator_hash=frozen['generator_sha256'],D_hash=frozen['D_sha256'],C0_used=False,K_selection='none, report all1/2/4/8')]);save(O/'decision.json',dict(status='COMPLETED_STOPPED_NO_P45',formal_acceptance=False,joint_passes={r['candidate']:r['joint_passes'] for r in summ},all_K_D_only_11=True,iterative_extra_benefit='not supported by joint C0 or adversarial D',failure_scope='current configuration; does not establish shared-field architecture impossible',major_implementation_limit='D binaryMask branch is a nondifferentiable shortcut; G adversarial gradient exists only through amplitude branch, despite decoder ST',shared_weights_and_full_output_valid=True,global_output_constant_collapse=False,source_information_preservation_fully_validated=False,existing_C27_D090_release_replaced=False,next_single_recommendation='Only if separately approved: repair discriminator occupancy gradient path with forward-identical hard occupancy and explicit decoder Mask surrogate; keep architecture/checkpoints/protocol otherwise fixed. Do not enlarge model or training budget.',no_automatic_new_training=True))
# Figures: all instances, all K and original baselines.
fig,ax=plt.subplots(figsize=(11,6));order=['Natural_raw','C3','C27','K1','K2','K4','K8'];A=pivot[order].to_numpy();im=ax.imshow(A,vmin=0,vmax=1,cmap='coolwarm',aspect='auto');ax.set_xticks(range(7),order);ax.set_yticks(range(11),pivot.index)
for i in range(11):
 for j in range(7):ax.text(j,i,f'{A[i,j]:.3f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=ax,label='AUC (target .475-.525)');fig.tight_layout();fig.savefig(O/'AUC_heatmap.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4))
for instance in pivot.index:ax.plot([1,2,4,8],pivot.loc[instance,['K1','K2','K4','K8']],'o-',label=instance,alpha=.75)
ax.axhspan(.475,.525,color='green',alpha=.15);ax.set(xlabel='Euler steps K (same weights)',ylabel='Frozen C0 AUC');ax.legend(ncol=2,fontsize=7,bbox_to_anchor=(1.02,1));fig.tight_layout();fig.savefig(O/'AUC_vs_K.png',dpi=180);plt.close(fig)
(O/'confusion_matrices').mkdir(exist_ok=True)
for K in [1,2,4,8]:
 fig,ax=plt.subplots(3,4,figsize=(12,9));rows=m[m.candidate==f'K{K}']
 for a,row in zip(ax.ravel(),rows.itertuples()):
  cm=np.asarray(__import__('ast').literal_eval(row.confusion_matrix));a.imshow(cm,vmin=0,vmax=905,cmap='Blues');a.set_title(row.instance,fontsize=9);a.set_xticks([0,1],['Sim','Real']);a.set_yticks([0,1],['Sim','Real'])
  for i in range(2):
   for j in range(2):a.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>450 else 'black')
 ax.ravel()[-1].axis('off');fig.suptitle(f'K={K}: rows true / columns predicted; original frozen C0');fig.tight_layout();fig.savefig(O/'confusion_matrices'/f'K{K}.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(13,4));ax[0].plot(hist.epoch,hist.D_loss_mean,label='D loss');ax[0].plot(hist.epoch,hist.G_adv,label='G adversarial loss');ax[1].plot(hist.epoch,hist.monitor_Train_AUC,label='D Train monitor AUC (512fixed)');ax[2].plot(hist.epoch,hist.G_preserve,label='G preserve proxy')
for a in ax:a.set_xlabel('Epoch');a.axvline(2.5,ls=':',color='gray');a.legend(fontsize=8);a.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'training_curves.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(12,4));q=quality.set_index('candidate')
for a,key in zip(ax,['zero_fraction','lag1_correlation','mean_window_std']):
 a.plot([1,2,4,8],[q.loc[f'K{k}',key] for k in [1,2,4,8]],'o-');a.axhline(q.loc['Real',key],label='Real',color='gray',ls=':');a.set(xlabel='K',ylabel=key);a.legend()
fig.tight_layout();fig.savefig(O/'output_structure_vs_K.png',dpi=180);plt.close(fig)
arch='''# P44架构与实际数据流

```mermaid
flowchart LR
 X["Natural Sim 4×712"] --> ES["冻结C3 Sim Encoder"]
 ES --> Z["源条件 z_s 4×64"]
 Z --> V["唯一共享向量场 v(h,z_s,t)"]
 H0["目标坐标 h0=0"] --> V
 V --> E["Euler: h←h+v/K，重复K次"]
 E --> DR["冻结C3 Real Decoder"]
 DR --> Y["完整 Real-like 4×712"]
 Y --> D["唯一训练判别器 Dφ"]
 R["Real Train"] --> D
```

- v=T_C3(z_s)+CorrectionGRU(h,z_s,t)。T_C3在v内部，不是独立前置G0，不先生成触觉再修正。
- Correction投影零初始化；K=1第一步就继承完整跨域映射。h0不是独立Sim latent。
- 四帧是触觉时间轴；K是潜状态Euler积分步数。二者不能混淆。
- 最终G包含两个完整VAE以保留双域模块；部署只调用Sim Encoder、共享field、Real Decoder，Real Encoder和Sim Decoder不在部署路径中。
- Real/Sim两个原VAE全部冻结，流场及唯一D更新。每个K均引用相同模型参数。
- GMM仅通过原C3伪配对训练过的Translator及训练期C3教师间接继承，本轮没有额外GMM损失或跨独立latent的OT。

## 梯度合同

Decoder前向为原硬MAP Mask×Gamma混合均值。训练时使用hard+(p−stopgrad(p))的替代梯度，前向与原Decoder逐位一致，Smoke已核验；正式推理完全使用原deterministic_decode。

当前D将输入拆为exact-positive Mask和log1p幅值。后者可回传到G，前者的x>0是离散操作、不可回传。这是本配置的重要梯度缺口，见gradient_path_audit.json，不能隐去或将其简单归咎于模型容量。
''';(O/'architecture.md').write_text(arch)
summarydf=pd.DataFrame(summ);shared=json.loads((O/'parameter_sharing_report.json').read_text());checks=[]
questions=[('Dual-VAE真正整合？','是，G Checkpoint保留两套完整原C3 VAE；部署Es→共享field→Dr。'),('首次调用从原Sim开始？','是，Es只读原processed；h0=0，K1为完整迁移，未用zs直接当Real latent。'),('独立G0前置推理？','没有。原GRU为field内部的可训练基项；C3完整输出只作为训练期教师缓存。'),('Translator继承？','权重按原结构直接加载，零初始化修正保留原映射初值；不是C27初始化。'),('GMM作用？','仅历史统计物理伪配对监督通过原GRU和教师继承，未新增GMM拟合或OT。'),('共享参数？','K1/2/4/8同一G、同一field与同一D；逐K调用和Hash已核验。'),('只有一个D？','训练Dφ只有一个；11C0是额外冻结评价模型，没有参与训练。'),('增加K改善？','没有联合门槛改善，四种K均1/11；各实例变化不单调。'),('K1/K8完整输出？','均为905×4×712 float32、finite、非负。'),('C0 AUC逐步接近.5？','没有，AUC热图/曲线保留所有实例，未挑选K。'),('Sim→Real识别率提高？','整体阈值预测多为Real；该事实无法消除AUC排序差异，不等于完成迁移。'),('Real→Real保持？','原参考780窗完全相同，11实例各自在7种比较中的Real recall逐值相同。'),('新D发现残余差异？','是，四种K的Validation均BA=AUC=1，绝非D没有学会。'),('输出坍缩？','非全零/常数，均880个不同输出，与C3/C27的880一致；不能把它等同于全部来源信息保留。'),('来源样本差异？','存在输出差异和输入减半响应；总响应相关仍为负，任务语义保持未认证。'),('空间/时间保持？','部分数值指标改善；同索引Mask IoU约.092而非物理位置认证，时序相关约.986仍低于Real.9993。'),('多步误差累积？','无明显数值爆炸；K结果有变化但没有联合收益，不能称增加K实现渐进域迁移。')]
for i,(question,answer) in enumerate(questions,1):checks.append(dict(item=i,question=question,answer=answer))
table(O/'research_questions.csv',checks)
report=['# P44：Dual-VAE与共享参数迭代对抗迁移首次可行性实验','', '**实际完成：单Seed训练20轮、同一G/D的四种K完整推理、77组原C0评价及新D完整Train/Validation评价。正式联合验收未通过：K=1/2/4/8均为1/11，未超过C27的2/11。** 原D专项封存版保持不变，本轮未产生新的正式通过版本。','', '**方法检验存在重要实现限制：新D的二值Mask分支不可对输入求导，而该分支能单独识别生成数据。生成器实际只从D的log幅值分支获得对抗梯度；Decoder自身的直通梯度没有自动穿过D的二次二值化。** 本轮应被解释为这套具体配置失败，不能据此否定共享迭代框架、或直接认定Decoder/模型容量是根因。未追加修复训练。','', '## 1. 数据、已有模型与保护范围','',pd.read_csv(O/'data_contract.csv').to_markdown(index=False),'', 'C27完整结束，原候选2/11联合、D专项11/11，其Checkpoint与原封存材料Hash在protected_hashes_before.json中。P44复用原C3 Real/Sim VAE和原GRU，采用本地CPU后验均值路径准备冻结特征。输入原通道顺序、非负float32 processed，不把原始接触力作为输入。','', 'Real物理Taxel方向、部分指节对应与单位仍未认证，Real时间间隔不均匀、Natural为连续20Hz。四帧索引严格递增；未重新采样、换通道、改原数据或读Test/Holdout。不操作服务器或实机，不终止其他实验。','', 'GMM作用沿用C3：480个字典条件构造1200/240统计物理伪配对，监督原GRU。P44没有拟合跨域GMM、使用独立潜空间距离或求OT；它不宣称本轮解决伪配对覆盖和歧义。','', '## 2. 生成器、共享参数与梯度实现','', '完整架构见architecture.md。原双VAE都包含在G中并冻结；field由原2层GRU基项与带时间输入的零初始化修正GRU组成，h0固定为目标空间零向量。source条件zs在每一步均存在，v在每次调用中完成映射计算，不调用独立触觉G0。','',f"field可训练参数{cfg['trainable_field_parameters']}，新D参数{cfg['D_parameters']}。同一个field在K=1/2/4/8分别调用1/2/4/8次，Hash与参数ID一致；一个Dφ持续更新，未为不同K创建新模型。参数完整列表和初始/最终共享检查保留。",'', '首次Smoke：K8展开、G梯度、D/R1二阶梯度均finite；ST Decoder前向与原硬Mask接口逐位一致，峰值显存在2GB门槛内。ST只用于替代反向导数，前向没有软Mask、填入微小正数或修改阈值。','', '## 3. 唯一训练配置与实际预算','',f"Seed4401；上限20轮或训练30分钟先到即停。实际20轮=前2轮只预热唯一D，后18轮交替更新G/D。D更新{tr['D_steps']}步，G更新{tr['G_steps']}步。每轮完整覆盖Natural4049及Real5153；累计{tr['Natural_visits']}和{tr['Real_visits']}次访问。各K有320个D训练步、288个G训练步。",'',f"训练{tr['train_seconds']:.2f}秒（20轮上限先到，未声称收敛），峰值显存{tr['GPU_peak_bytes']/2**20:.1f}MiB。速度来自冻结VAE、源latent和C3教师缓存，主训练只更新field及轻量全局MLP。准备、Smoke与最终评价耗时另存，不把它们算成训练时间。",'',f"Adam lr1e-4、betas(0,.99)、clip5；Lazy R1 gamma.1每16个D步一次。λpreserve={cfg['lambda_preserve']:.8g}由固定128条Train初始损失比例确定；λflow=.001；C3教师log输出损失权重从1线性减到第6个GAN轮的0。没有不合理Real/Sim逐样本MSE。",'', 'Lpreserve为索引Patch份额差和相邻相对响应变化差，不要求原Sim Mask逐位不变。两种量只作代理，不能当作真实接触语义。教师只对同源Sim的C3输出蒸馏，最终部署不依赖外部C3生成器。','', 'Checkpoint预先规定为最后完整轮（第20轮），无C0选权重或选K，没有额外Seed、额外网络或延长Epoch。config.yaml内容为合法YAML1.2的JSON写法，与config.json一致。','', '## 4. 原冻结11C0正式评价','',summarydf.to_markdown(index=False,floatfmt='.7g'),'', 'BA与D数学等价。四个K的D专项均11/11，但只有kNN通过BA/AUC/D联合门槛，不能按D宣布本轮正式成功。完整逐实例BA/D/AUC、混淆矩阵、Sim→Real=1−recall_sim、Real→Real=recall_real、300次Session/Root Bootstrap和预测分数均在C0_metrics.csv及evaluation/。不同K下原Real参考的各实例recall相同，未修改阈值或参考数据。','', pivot[['Natural_raw','C3','C27','K1','K2','K4','K8']].to_markdown(floatfmt='.6f'),'', 'K0只代表原始Natural，没有执行G(K=0)。C3/C27是冻结历史完整输出基线，未分别重新训练。对所有K用同一份Generator和同一份原C0，未从好看的实例或K构建新结论。','', '## 5. 新共享判别器','',d[['role','K','balanced_accuracy','roc_auc','Loss_BCE','confusion_matrix']].to_markdown(index=False,floatfmt='.7g'),'', '最终D在全部四种K的Train及Validation上均AUC=1，Validation混淆矩阵为[[905,0],[0,780]]；说明它确实发现强残余域差异，不是欠拟合造成随机AUC。K0原始Natural不是D训练的fake类型，其结果仅作分布外诊断，不拿它替代生成数据的评价。','', '### 梯度和分支诊断','',branch.to_markdown(index=False,floatfmt='.7g'),'', '此表是已训练D的输入分支消融，没有训练Mask-only/Amplitude-only新Probe，也不是正式C0验收。Mask分支单独保留时仍可在Validation完全区分；幅值分支单独保留仍有较高AUC。遮蔽一个分支属于诊断分布改变，不能把消融数值当作该分支独立训练的性能。','', '代码证据：D.forward的(x>0).float()不保留到x的梯度。即使Decoder的Mask用了ST，D重新二值化仍会断开该支路。D输入log幅值支路有finite非零梯度，所以本轮不是完全没有生成器训练，但对抗目标的所有判别特征没有得到同等梯度响应。这个实现限制妨碍了对“共享迭代机制是否有效”的充分判断，也不证明它是唯一失败根因。','', '## 6. 完整712D质量与信息保持','',quality.to_markdown(index=False,floatfmt='.7g'),'', '每个K保存全部905窗，不裁剪、删除或best-of-K。输出均finite、非负；均880个不同窗口，与原C3/C27相同，不是新增全常数坍缩。但这些数值不能证明完整源信息保持。','', '相对C3，P44时序相关从约.7895提高到约.9862–.9865，峰值从48.67降至约17.5；但Real时序相关约.9993、峰值约10.49，零比例也从Real83.43%偏到生成79.31%。部分边际/时序改善没有转化为AUC联合通过。','',pd.read_csv(O/'input_response.csv').to_markdown(index=False,floatfmt='.7g'),'', '索引Mask IoU约.092，不能冒充物理位置保持；总响应关联仍为负、时间变化关联较弱。输入减半诊断虽显示非零响应，也只是固定64窗的数值依赖，不是任务保持证明。逐Root/Task结构、Patch激活/空块比例、C0分组结果全部另存。','', '## 7. 17项研究问题','',pd.DataFrame(checks).to_markdown(index=False),'', '## 8. 决策、未执行项与唯一建议','', '**正式验收未通过，不替换C27封存版，不自动进入P45。** 当前证据支持“一个G/D及共享K架构可运行”，不支持“对抗训练消除了域差异”或“增加K带来渐进迁移收益”。本轮不把失败归为容量不足，也未证明需要解冻VAE。','', '**若后续另获确认，唯一优先项是补通D占用特征到G的梯度，保持硬前向不变，再做受控验证。** 不先加大网络或训练时间，不重开OT/Diffusion，也不修改C0。该建议未实施，本轮仅做冻结诊断，没有追加训练。','', '本轮未执行：重新训练VAE、新GMM拟合、新物理采集、跨独立空间OT、额外Seed、按K分别训练、Test/Holdout评价、服务器改动、自动提交或push。','', '## 9. 模型与复现','', 'generator.pt包含原双VAE及唯一field的全部权重，discriminator.pt包含唯一训练D。code/infer.py与同目录model.py/vae.py即可独立推理，不依赖Real、GMM、C0或外部前置G0。K参数只允许1/2/4/8。','', '```bash','python code/infer.py --checkpoint generator.pt --input NATURAL_WINDOWS.npz --key values --K 8 --output NEW_OUTPUT.npz','```','', 'code/train.py、evaluate.py、gradient_audit.py、report.py保存实际过程；训练复现仍需原项目Train/Validation数据。所有Checkpoint、代码、源数据索引与输出有Hash，原数据、C3/C27和D专项包未修改。执行命令见run_commands.json，完整性见integrity.json。']
(O/'REPORT.md').write_text('\n'.join(report)+'\n');protected=json.loads((O/'protected_hashes_before.json').read_text());assert all(sha(p)==h for p,h in protected.items());assert sha(O/'generator.pt')==frozen['generator_sha256'];assert sha(O/'discriminator.pt')==frozen['D_sha256'];save(O/'integrity.json',dict(all_protected_unchanged=True,files=protected,all_K_share_same_G_D=True,VAE_weights_bit_identical=True,Generator_Discriminator_frozen_during_C0=True,C0_not_trained_or_used_for_G_gradients=True,original_C27_D090_release_unchanged=True,Test_Holdout_not_read=True,no_server_or_real_hardware_operation=True,no_further_training=True));save(O/'run_commands.json',dict(python='/home/zmy/miniconda3/envs/mujoco/bin/python',environment={'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},executed=[f'python {O}/code/{p}.py' for p in ['train','evaluate','gradient_audit','report']],inference=f'python {O}/code/infer.py --checkpoint {O}/generator.pt --input NATURAL.npz --K 8 --output NEW.npz',warning='Training requires fresh result directory and original registered Train/Validation; never overwrite historical artifacts',code_hashes={str(p):sha(p) for p in (O/'code').glob('*.py')}));print('P44 REPORT COMPLETE NO_P45',flush=True)
