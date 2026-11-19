from c0_support import *
from model import Generator
import pandas as pd
import hashlib,ast,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P44=Path('outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18');assert (O/'evaluation_complete.json').exists() and (O/'diagnostics_complete.json').exists();cfg=json.loads((O/'config.json').read_text());m=pd.read_csv(O/'C0_metrics.csv');m['Sim_to_Real_rate']=1-m.recall_sim;m['Real_to_Real_rate']=m.recall_real;m.to_csv(O/'C0_metrics.csv',index=False);q=pd.read_csv(O/'output_statistics.csv');d=pd.read_csv(O/'D_metrics.csv');grad=pd.read_csv(O/'gradient_audit.csv');summary=[]
for (group,K),g in m.groupby(['group','candidate'],sort=True):summary.append(dict(group=group,K=int(K[1:]),joint_passes=int(g.passed_joint.sum()),D_only_passes=int((g.domain_confusion>=.9).sum()),min_D=float(g.domain_confusion.min()),mean_abs_AUC_minus_half=float(np.abs(g.roc_auc-.5).mean()),AUC_min=float(g.roc_auc.min()),AUC_max=float(g.roc_auc.max())))
summ=pd.DataFrame(summary);summ.to_csv(O/'C0_summary.csv',index=False)
for instance,g in m.groupby('instance'):assert g.recall_real.max()-g.recall_real.min()<1e-12
# Check full frozen VAE/buffer identity and same initial state, without training.
initial=torch.load(P44/'generator_initial.pt',map_location='cpu',weights_only=False);states={}
for group in ['A','C']:
 ck=torch.load(O/group/'generator.pt',map_location='cpu',weights_only=False);assert all(torch.equal(v,initial['model'][k]) for k,v in ck['model'].items() if not k.startswith('field.'));f=json.loads((O/group/'frozen.json').read_text());assert sha(O/group/'generator.pt')==f['generator_sha256'];assert sha(O/group/'discriminator.pt')==f['D_sha256'];states[group]=f
init=pd.DataFrame(json.loads((O/'initialization_audit.json').read_text()));trainrows=[]
for group in ['A','B','C']:
 tr=json.loads(((P44/'training_summary.json') if group=='B' else (O/group/'training_summary.json')).read_text());trainrows.append(dict(group=group,epochs=tr.get('epochs',tr.get('completed_epochs')),G_steps=tr['G_steps'],D_steps=tr['D_steps'],train_seconds=tr.get('train_seconds'),GPU_peak_MiB=tr['GPU_peak_bytes']/2**20,source='historical P44' if group=='B' else 'new controlled run'))
table(O/'training_comparison.csv',trainrows)
# Side-by-side per-instance AUC, no selected K.
pivot=m.assign(groupK=m.group+'_'+m.candidate).pivot(index='instance',columns='groupK',values='roc_auc').reindex(['logistic','knn','mlp_11','mlp_12','mlp_13','gru_11','gru_12','gru_13','transformer_11','transformer_12','transformer_13']);columns=[f'{group}_K{k}' for group in ['A','B','C'] for k in [1,2,4,8]];pivot[columns].to_csv(O/'AUC_comparison.csv')
fig,ax=plt.subplots(figsize=(14,6));a=pivot[columns].to_numpy();im=ax.imshow(a,vmin=0,vmax=1,cmap='coolwarm',aspect='auto');ax.set_xticks(range(12),columns,rotation=35,ha='right');ax.set_yticks(range(11),pivot.index)
for i in range(11):
 for j in range(12):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',fontsize=7)
fig.colorbar(im,ax=ax,label='Frozen C0 AUC (target .475-.525)');fig.tight_layout();fig.savefig(O/'AUC_comparison.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(13,4))
for group in ['A','B','C']:
 ss=summ[summ.group==group];ax[0].plot(ss.K,ss.mean_abs_AUC_minus_half,'o-',label=group);qq=q[q.group==group].sort_values('K');ax[1].plot(qq.K,qq.positive_P99,'o-',label=group);ax[2].plot(qq.K,qq.mean_window_std,'o-',label=group)
ax[0].set_ylabel('Mean abs AUC-.5 (descriptive, NOT acceptance)');ax[1].set_ylabel('Positive P99');ax[2].set_ylabel('Mean across-window std')
for a in ax:a.set_xlabel('K (same group weights)');a.legend();a.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'summary_and_quality.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(13,4))
for group in ['A','B','C']:
 hist=pd.read_csv((P44/'training_history.csv') if group=='B' else O/group/'training_history.csv');ax[0].plot(hist.epoch,hist.D_loss_mean,label=group);ax[1].plot(hist.epoch,hist.G_adv,label=group);ax[2].plot(hist.epoch,hist.monitor_Train_AUC,label=group)
ax[0].set_yscale('symlog',linthresh=.1);ax[0].set_ylabel('D loss incl lazyR1');ax[1].set_ylabel('G adversarial loss');ax[2].set_ylabel('D monitor Train AUC (fixed512)')
for a in ax:a.set_xlabel('Epoch');a.legend();a.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'training_comparison.png',dpi=170);plt.close(fig)
(O/'confusion_matrices').mkdir(exist_ok=True)
for group in ['A','B','C']:
 for K in [1,2,4,8]:
  rows=m[(m.group==group)&(m.candidate==f'K{K}')];fig,ax=plt.subplots(3,4,figsize=(12,9))
  for a,row in zip(ax.ravel(),rows.itertuples()):
   cm=np.array(ast.literal_eval(row.confusion_matrix));a.imshow(cm,vmin=0,vmax=905,cmap='Blues');a.set_title(row.instance,fontsize=9);a.set_xticks([0,1],['Sim','Real']);a.set_yticks([0,1],['Sim','Real'])
   for i in range(2):
    for j in range(2):a.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>450 else 'black')
  ax.ravel()[-1].axis('off');fig.suptitle(f'{group} K{K}: true rows / predicted columns');fig.tight_layout();fig.savefig(O/f'confusion_matrices/{group}_K{K}.png',dpi=140);plt.close(fig)
# Complete trajectories include B inference-only supplement.
traj=pd.concat([pd.read_csv(O/'iteration_trajectory.csv'),pd.read_csv(O/'B_iteration_trajectory.csv')],ignore_index=True);traj.to_csv(O/'all_iteration_trajectories.csv',index=False);fig,ax=plt.subplots(1,3,figsize=(12,4))
for a,group in zip(ax,['A','B','C']):
 for K,g in traj[traj.group==group].groupby('K'):a.plot(g.time,g.state_norm_mean,'o-',label=f'K{K}')
 a.set(title=group,xlabel='Euler normalized time',ylabel='Mean target-state norm');a.legend(fontsize=8)
fig.tight_layout();fig.savefig(O/'iteration_trajectories.png',dpi=170);plt.close(fig)
# Honest decision: no acceptable migration improvement, A remains only a representation candidate.
decision=dict(status='COMPLETED_STOPPED_NO_P46',formal_acceptance=False,pass_counts={f"{r['group']}_K{r['K']}":r['joint_passes'] for r in summary},explicit_Mask_required_for_domain_detection=False,continuous_D_train_validation_AUC='~.992/.987, strong discrimination without explicit Mask',STE_gradient_repaired=True,STE_transfer_success=False,A_mean_AUC_distance_improved_vs_B=True,A_quality_gate_passed=False,C_amplitude_and_diversity_shrink=True,global_constant_collapse=False,recommendation='If separately continued, prioritize continuous-only A as representation candidate; first resolve output-tail/structure tradeoff, no further Mask/temperature search. No new architecture/training started.',limitations=['singleSeed historically exposed Validation','A first-layer parameter count lower; common weights mapped but not identical architecture size','R1 formula/weight fixed; effective gradient penalty magnitude differs byrepresentation','Real Decoder self-reconstruction still distinguishable; not a proof of an absolute ceiling'],existing_D090_release_replaced=False,no_more_models=True);save(O/'decision.json',decision)
arch='''# P45架构对照与受控范围

共同生成器始终为P44初始G：冻结Es→唯一时间条件共享field（原Translator初始化）→冻结Dr。Dr的硬Mask、训练期ST反向、Gamma幅值结构均未改变。h0仍为固定零状态，不是Sim latent。

|组|D输入|前向|G侧反向|输入层参数|
|---|---|---|---|---|
|A|原完整连续log1p触觉/固定scale，2848维|不计算Mask|连续路径|原P44首层仅保留幅值半列|
|B|硬Mask+连续，5696维|原P44|Mask分支为0|原P44，历史结果|
|C|硬Mask+连续，5696维|逐值等于B|Mask采用sigmoid代理导数|原P44全部权重|

A不是删去触觉零值，也不是删去生成器的Mask。零和正值均原样进入完整连续712D。

C的epsilon=0，tau为Real Train全部正值中位数；hard+(soft-soft.detach())保证前向精确硬Mask。只在G反传中启用Mask代理；Real侧R1用mask_backward=False，保持P44原正则化路径，避免另改正则项。

A/C均从同一P44训练前Generator加载。C与B的初始D全部参数相同；A的公共层相同、首层拷贝原幅值列，因此A有559809参数而B/C有1106625，不能声称总参数完全相同。

每组只训练一个G/D，四个K共享同一组权重，K不等于触觉四帧时间轴。
''';(O/'architecture_comparison.md').write_text(arch)
readout=pd.read_csv(O/'Real_reconstruction_D_readout.csv');resp=pd.read_csv(O/'input_response.csv');initgrad=grad[grad.phase=='initial'];finalgrad=grad[grad.phase=='final'];report=['# P45：连续触觉对抗迁移的显式Mask与梯度路径消融','', '**已完成A/C两个单Seed受控训练，B复用P44。没有任何组、任何K通过11/11原联合门槛。** A的平均AUC偏差有所改善，但联合通过仍为0–1/11且尾部异常；C恢复Mask梯度后出现明显幅值和样本差异收缩。不能宣布连续方案或STE方案迁移成功。本轮停止，不启动P46。','', '## 1. 前置审计与控制变量','', 'P44的generator_initial.pt和discriminator_initial.pt均存在，未使用终点冒充初始。A/C完整Generator初始state hash相同；原P44初始与终点field不同，双VAE相同。保存初始化源文件Hash、逐参数Hash和冻结范围。','', init.to_markdown(index=False),'', 'A的公共层精确拷贝P44初始D，首层只取连续幅值列。C的公共D权重全拷贝，Mask前向与B逐值一致。A参数量约减半，故本消融不能把A/B全部差异仅归于一个不影响容量的开关；没有补加隐藏层来追平参数数。','', 'Sim/Real原Train与Validation、原通道/时间顺序、Es/Dr、共享field、Translator初始化、h0、GAN损失、教师及保持/flow权重、优化器lr/betas、clip、K分配均继承P44。C只在G侧增加Mask代理梯度；Real R1特意沿用原硬路径，否则R1会同时包含新Mask导数，破坏控制。','', f"C固定epsilon={epsilon if False else cfg['epsilon']}，tau={cfg['tau']:.12g}，后者来自Real Train正值中位数，未用C0或Validation选温度。无输出阈值变化、填正值、裁剪或删窗。",'', '## 2. 梯度检查','', initgrad.to_markdown(index=False,floatfmt='.8g'),'', '相同初始G/D、相同Train前8窗、K8：A连续路径有效；B的Mask VJP到向量场为零；C的Mask VJP到输入和field均finite且非零。Decoder自身仍有硬Mask，但沿用P44已存在的硬前向/ST反向；未把Decoder阈值修改成新机制。','', finalgrad.to_markdown(index=False,floatfmt='.8g'),'', '最终权重的冻结复查显示，C的Mask梯度确实保留下来，且量级增大；梯度更大不等于分布更好。所有分支是独立VJP分解，不是人为加入常数梯度，也不参与验收打分。梯度统计仅为固定8窗，不冒充全数据梯度分布。','', '## 3. 实际训练','',pd.DataFrame(trainrows).to_markdown(index=False,floatfmt='.6g'),'', 'A/C各20轮，与P44一致：2轮D预热+18轮交替；每组G1152步、D1280步，每K对应G288步/D320步。Natural4049、Real5153每轮全部参与。相同种子4401、相同采样算法及逐轮order Hash、覆盖计数全部核验。B没有重训；采样源代码与种子可重建其顺序，原B未额外补造训练日志。','', '固定最后完整轮作为Checkpoint，未看C0挑epoch或K。每组预算20轮/30分钟，实际先到20轮，未延长。R1系数和公式未改，但A连续表示训练中R1数值量级明显更大；这是同一正则器在不同表示下的实际行为，需与表示/容量差异一起保留解释。','', '训练前已完成K8前反向，无OOM且所有梯度finite；显式峰值记录包含实际训练峰值和训练后无优化器的冻结权重Smoke，后者不伪称训练前测量。一次CuDNN eval模式Smoke错误在任何优化前修正；一次评价表格group列冲突在任何C0运行前修正，原日志保留；均未因此重训或换配置。','', '## 4. 原11个冻结C0横向比较','',summ.to_markdown(index=False,floatfmt='.7g'),'', 'mean_abs_AUC_minus_half仅为描述性汇总，不是验收判据。正式判定始终逐实例BA[.45,.55]、AUC[.475,.525]、D≥.90，11/11同时通过。所有组、K的D专项均11/11，但联合仅0或1个；不能用D替代联合成功。原C0/Scaler/log1p/0.5/标签/通道合同全部未变，Real参考召回在12种比较中逐实例相同。','', pivot[columns].to_markdown(floatfmt='.6f'),'', '132组原C0结果、完整预测分数、分组Bootstrap及逐Root/Task结果保留。B分数来自P44原文件，A/C全部训练完成后才统一评价；没有使用C0训练梯度或调整温度/权重。','', '## 5. 新判别器训练是否正常','',d[['group','role','K','balanced_accuracy','roc_auc','Loss_BCE']].to_markdown(index=False,floatfmt='.7g'),'', 'A即使没有显式Mask，Train约.992、Validation约.987，仍是强域判别器；没有证据把它当作无法训练的弱分类器。B约1，C约.9998/1。A有少量Train/Validation差距，但没有AUC随机化或严重泛化崩溃，C也不存在靠D失效制造通过。','', '这说明完整连续触觉足以在当前数据上学习明显域信息；显式Mask不是“能否识别域”的必要输入。但它是确定性结构先验/计算捷径，会改变可学习性和梯度条件，并非增加新的传感器观测。不能由此直接判定未来所有GAN都不需要Mask。','', '## 6. 完整712D质量与来源保持','',q.to_markdown(index=False,floatfmt='.7g'),'', '全部输出为905×4×712 float32、finite、非负，未全零或完全常数；每种输出880个数值窗，与P44相同。唯一数目不等于有效多样性；C窗口间标准差约.0052，远低于Real约.0464，且P99约1.07–1.15相对Real4.57明显收缩，时间相关降到约.47–.49。不能因为仍有880个不同窗就忽略近收缩/信息损失。','', 'A的P99约3.12–3.15更接近Real，但极端峰值约67.7–71.3，高于B约17.5和Real10.49；时间相关约.87–.90也低于B约.986。因此A的平均AUC改善伴随质量代价，尚不满足“Continuous-only成功”的结构要求。','',resp.to_markdown(index=False,floatfmt='.7g'),'', '同索引Mask IoU仅为诊断，物理Taxel方向仍UNKNOWN。总响应、激活量和时序变化相关是关联而非受控物理对应；未使用它们伪造配对。逐Taxel712通道统计、Patch空块/激活、每Root/Task、全部迭代状态轨迹另存，K增加没有稳定联合收益。','', '## 7. 冻结Decoder瓶颈的辅助证据','',readout.to_markdown(index=False,floatfmt='.7g'),'', '三组D也能一定程度区分“原Real”与“原C3 Real Encoder/Decoder自身重建”；两侧来源都是Real，这不是新Natural迁移验收。A约.959、B约.819、C约.758，提示目标域读出伪影仍存在。它不证明Decoder输出集合永远无法匹配Real，也不是0.90理论上界；更不能以此自动解冻Decoder。','', '## 8. 决策：哪些结论成立','', '1. **Mask是否必要？** 对当前数据上的域识别，显式Mask不是必要条件，连续D本身已经很强；对成功迁移，A/C都未给出完成证据。','2. **梯度断点是否影响优化？** 是。C修复了D→G的占用梯度并明显改变了训练结果；但没有获得联合收益，且产生幅值/多样性收缩，所以断点不是唯一障碍。','3. **连续712D能否替代显式Mask？** 作为判别器输入可行；作为已验证的成功迁移方案尚不成立。A有描述性AUC改善，质量仍未过关。','4. **哪条最值得保留？** 若以后继续，A是较简单、较有价值的表示候选，优先研究它的尾部/保持约束矛盾；这只是优先级，不升级默认模型、不立刻开P46。','5. **还不能解释什么？** 单Seed、历史Validation反复使用、A参数量/有效R1差异、任务/阶段/采样协议与Decoder伪影的独立贡献未分离。不能把所有失败归为Mask、容量或单一物理因素。','', '本轮不满足预设A/B/C成功情形；更接近“没有得到可接受迁移改善”的情形D，同时C有结构退化。它不是情形E的假成功，因为没有一组完整联合AUC通过。**停止Mask/温度搜索，不换GAN架构，不启动P46。**','', '## 9. 复现与保护','', 'A/generator.pt、A/discriminator.pt和C对应文件为唯一端点模型。B只引用P44原权重/结果，新增B轨迹为冻结推理补充。code/infer.py可用同一组G权重指定K，完整Generator仍包含原冻结双VAE，不需要D、Real或独立G0。','', '```bash','python code/infer.py --checkpoint A/generator.pt --input NATURAL_WINDOWS.npz --key values --K 8 --output NEW_OUTPUT.npz','```','', 'code/train.py为受控训练；config.yaml为合法YAML1.2的JSON形式。全部源模型/数据/C0/P44和D专项包Hash保持不变；无Test/Holdout、服务器修改、实机采集或额外训练。详细命令、代码Hash及完整性另存。']
(O/'REPORT.md').write_text('\n'.join(report)+'\n');assert all(sha(p)==h for p,h in json.loads((O/'protected_hashes_before.json').read_text()).items());save(O/'integrity.json',dict(all_protected_unchanged=True,files=json.loads((O/'protected_hashes_before.json').read_text()),P44_initial_used_not_final=True,VAE_and_initial_state_buffers_unchanged=True,A_C_sampling_orders_identical=True,steps_match_P44=True,only_D_representation_and_backward_changed=True,C_R1_matches_B_hard_path=True,C0_never_used_for_training_or_selection=True,no_Test_Holdout=True,no_server_or_hardware_operation=True,no_P46=True));save(O/'run_commands.json',dict(python='/home/zmy/miniconda3/envs/mujoco/bin/python',environment={'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},executed=[f'python {O}/code/{name}.py' for name in ['train','evaluate','diagnostics','report']],inference_example=f'python {O}/code/infer.py --checkpoint {O}/A/generator.pt --input NATURAL.npz --K 8 --output NEW.npz',training_requires='fresh result directory and registered original data; never overwrite historical outputs',code_hashes={str(p):sha(p) for p in (O/'code').glob('*.py')}));print('P45 REPORT COMPLETE STOP_NO_P46',flush=True)
