from audit_utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess,sys
assert all((O/f'{phase}_complete.json').exists() for phase in ['generate','swd','equivalence','quality','jsd_summary'])
def read(name):return list(csv.DictReader((O/name).open()))
js=read('metrics_jsd.csv');sw=read('metrics_swd.csv');eq=read('equivalence_metrics.csv');path=read('path_summary.csv');step=read('path_metrics.csv');quality=read('quality_metrics.csv');counts=json.loads((O/'actual_training_counts.json').read_text());reg=json.loads((O/'registration.json').read_text());cal=json.loads((O/'equivalence_calibration_summary.json').read_text());gate=read('baseline_validity_gate.csv');grid=read('integration_grid_sensitivity.csv');models=reg['models'];fig=O/'figures';fig.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'figure.dpi':140})
# Explicit metric rankings byestimator/seed, never selecting a preferred seed.
rankrows=[]
for seed in [11,12,13]:
 for K in [1,2,4,8]:
  values=[next(r for r in js if r['model']==f'{a}_K{K}' and r['instance']==f'gru_{seed}') for a in ['A','B','C']];ranked=sorted(values,key=lambda r:float(r['JSD_proxy']));rankrows.append(dict(metric='GRU JSD_proxy lowerbetter',seed=seed,K=K,order=json.dumps([r['model'] for r in ranked]),not_model_selection=True))
for a in ['A','B','C']:
 values=[r for r in sw if r['model'].startswith(a+'_')];rankrows.append(dict(metric='SWD lowerbetter',family=a,order=json.dumps([r['model'] for r in sorted(values,key=lambda r:float(r['SWD']))]),not_model_selection=True))
table(O/'metric_rankings.csv',rankrows)
# GroupCE decomposition makes overconfident source-baseline failures traceable.
groups=[];rr=real('validation');ss=natural('validation')
for model in models:
 for ident in reg['JSD']['instances']:
  folder=O/'probes'/model/ident
  if not (folder/'predictions.npz').exists():continue
  pp=np.load(folder/'predictions.npz');ll=pp['validation_logits'];yy=pp['validation_labels'];loss=np.logaddexp(0,ll)-yy*ll
  for domain,ix,gg in [('Real',np.arange(780),rr['g']),('Sim',np.arange(780,1685),ss['g'])]:
   for g in np.unique(gg):
    jj=ix[gg==g];groups.append(dict(model=model,instance=ident,domain=domain,group=g,windows=len(jj),CE=float(loss[jj].mean()),domain_totalweight=.5,group_totalweight=.5/len(np.unique(gg)),weighted_CE_contribution=float(loss[jj].mean()*.5/len(np.unique(gg))),accuracy=float(((ll[jj]>=0)==yy[jj]).mean())))
table(O/'probe_group_audit.csv',groups)

# One unified table keeps each estimator/seed, never averages a missing family into success.
unified=[]
for m in models:
 sr=next(r for r in sw if r['model']==m);er=next(r for r in eq if r['model']==m)
 for row in [r for r in js if r['model']==m]:unified.append(dict(row,SWD=sr['SWD'],R_W=sr['R_W'],R_W_CI_low=sr['R_W_CI_low'],R_W_CI_high=sr['R_W_CI_high'],MMD=er['MMD'],epsilon_RR=er['epsilon_RR'],Delta_eq_CI_low=er['Delta_CI_low'],Delta_eq_CI_high=er['Delta_CI_high'],marginal_equivalent=er['marginal_equivalent']))
table(O/'unified_metrics.csv',unified)
# Combine bootstrap route identities; fields remain route-specific.
bs=[]
for name in ['jsd_bootstrap.csv','swd_bootstrap.csv','equivalence_bootstrap.csv']:bs.extend(read(name))
table(O/'bootstrap_results.csv',bs)
# Correct planned scheduler counters using actual persisted model histories/status, preserve scheduler record.
completion=json.loads((O/'probes_complete.json').read_text());save(O/'probe_scheduler_planned_counts.json',completion);completion.update(actual_training_counts=counts,deep_instances=counts['actual_audit_models']-1,actual_audit_models=counts['actual_audit_models'],skipped_audit_models=counts['skipped_audit_models'],planned_deep_instances=78);save(O/'probes_complete.json',completion)
summary=dict(date='2026-11-19',scope='frozen development audit,notnewblindacceptance',migration_training_epochs=0,migration_optimizer_steps=0,actual_audit_training=counts,K='sharedfield Euler integration on same[0,1],nottrainingepochs',all_Validation_reinference_bit_equal=all(r['bit_equal_old']=='True' for r in read('K_semantics_audit.csv')),JSD=dict(all_family_baselines_valid=all(r['baseline_valid']=='True' for r in gate),robust_family90=False,reason='predeclaredLogistic/MLP baseline failure;validGRU results retained,no estimator cherry-pick'),SWD=dict(Original=float(sw[0]['SWD']),best_observed='A_K1',best_reduction=float(next(r['R_W'] for r in sw if r['model']=='A_K1')),any_candidate90=any(r['point90']=='True' for r in sw if r['model'] in models[1:])),equivalence=dict(epsilon=cal['epsilon_RR'],any_candidate_pass=any(r['marginal_equivalent']=='True' for r in eq if r['model'] in models[1:]),scope='fixedwindow marginal sampling only;notphysicalequivalence'),new090_achieved=False,C0_changed=False,P49_started=False);save(O/'summary.json',summary)
# Fullmodel SWD with groupbootstrap intervals.
f,ax=plt.subplots(figsize=(11,4));means=np.array([float(next(r['R_W'] for r in sw if r['model']==m)) for m in models]);lower=np.array([float(next(r['R_W_CI_low'] for r in sw if r['model']==m)) for m in models]);upper=np.array([float(next(r['R_W_CI_high'] for r in sw if r['model']==m)) for m in models]);ax.errorbar(range(13),means,yerr=[means-lower,upper-means],fmt='o',capsize=3);ax.axhline(.9,color='red',ls='--',label='90% reduction target');ax.axhline(0,color='gray');ax.set_xticks(range(13),models,rotation=55);ax.set_ylabel('SWD relative reduction');ax.legend();f.tight_layout();f.savefig(fig/'SWD_reductions.png');plt.close(f)
f,axs=plt.subplots(1,3,figsize=(13,4))
for group,ax in zip(['A','B','C'],axs):
 for K in [1,2,4,8]:
  rows=[r for r in step if r['model']==group and int(r['K'])==K];ax.plot([float(r['time']) for r in rows],[float(r['SWD']) for r in rows],marker='o',label=f'K={K}')
 ax.axhline(float(sw[0]['SWD']),ls='--',c='gray',label='Original Sim');ax.set_title(group+' frozen Euler path');ax.set_xlabel('Integration time t (not physical seconds)');ax.set_ylabel('SWD to Real');ax.legend(fontsize=7)
f.tight_layout();f.savefig(fig/'decoded_path_SWD.png');plt.close(f)
f,axs=plt.subplots(1,2,figsize=(11,4))
for a in ['A','B','C']:
 rows=[r for r in path if r['model']==a];grows=[r for r in grid if r['model']==a];axs[0].plot([1,2,4,8],[float(r['mean_PathRatio']) for r in rows],marker='o',label=a);axs[1].plot([1,2,4,8],[float(r['latent_mean_PathRatio']) for r in grows],marker='o',label=a)
axs[0].set_title('Decoded full2848 path ratio');axs[1].set_title('Same target latent path ratio');
for ax in axs:ax.set_xlabel('Euler K');ax.set_xticks([1,2,4,8]);ax.legend()
f.tight_layout();f.savefig(fig/'latent_vs_decoded_paths.png');plt.close(f)
f,ax=plt.subplots(figsize=(11,4));ers=[next(r for r in eq if r['model']==m) for m in models];vv=np.array([float(r['Delta_eq']) for r in ers]);ll=np.array([float(r['Delta_CI_low']) for r in ers]);hh=np.array([float(r['Delta_CI_high']) for r in ers]);ax.errorbar(range(13),vv,yerr=[vv-ll,hh-vv],fmt='o',capsize=3);ax.axhline(0,c='red',ls='--');ax.set_xticks(range(13),models,rotation=55);ax.set_ylabel('MMD - RealReal tolerance');ax.set_title('Equivalence requires upper95%CI <0');f.tight_layout();f.savefig(fig/'equivalence_intervals.png');plt.close(f)
f,axs=plt.subplots(1,2,figsize=(12,4));original=[r for r in js if r['model']=='Original'];axs[0].bar(range(7),[float(r['JSD_proxy']) for r in original]);axs[0].set_xticks(range(7),[r['instance'] for r in original],rotation=55);axs[0].axhline(0,color='gray');axs[0].set_title('Signed Original JSD proxies (not clipped)')
for a in ['A','B','C']:
 for seed in [11,12,13]:
  vals=[next(r for r in js if r['model']==f'{a}_K{k}' and r['instance']==f'gru_{seed}') for k in [1,2,4,8]];axs[1].plot([1,2,4,8],[float(r['R_JS']) if r.get('R_JS') else np.nan for r in vals],marker='o',label=f'{a} GRU{seed}')
axs[1].axhline(.9,color='red',ls='--');axs[1].set_title('Valid-baseline GRU reduction;allseeds');axs[1].legend(fontsize=6,ncol=3);axs[1].set_xlabel('K');f.tight_layout();f.savefig(fig/'JSD_proxy_audit.png');plt.close(f)
# Separate training-only objectives and untouched final Validation curves; no checkpoint selection.
f,axs=plt.subplots(1,3,figsize=(13,4))
for ax,group in zip(axs,['Original','A_K1','B_K1']):
 for seed in [11,12,13]:
  name=O/f'probes/{group}/gru_{seed}/history.csv'
  if not name.exists():continue
  h=list(csv.DictReader(name.open()));ax.plot([int(r['epoch']) for r in h],[float(r['train_CE']) for r in h],label=f'train{seed}');ax.plot([int(r['epoch']) for r in h],[float(r['validation_CE']) for r in h],ls='--',label=f'val{seed}')
 ax.set_title(group+' audit GRU');ax.set_xlabel('Epoch');ax.set_ylabel('Group-balanced CE');ax.legend(fontsize=6,ncol=2)
f.tight_layout();f.savefig(fig/'probe_training_curves.png');plt.close(f)
def swrow(m):return next(r for r in sw if r['model']==m)
def eqrow(m):return next(r for r in eq if r['model']==m)
rows=[]
for m in models:
 s=swrow(m);e=eqrow(m);gr=[r for r in js if r['model']==m and r['instance'].startswith('gru_') and r.get('JSD_proxy')];jr=[float(r['JSD_proxy']) for r in gr];rr=[float(r['R_JS']) for r in gr if r.get('R_JS')];ranges=lambda v:f'{min(v):.4f}–{max(v):.4f}' if v else 'NA';rows.append(f"| {m} | {ranges(jr)} | {ranges(rr)} | {float(s['SWD']):.4f} | {float(s['R_W']):.2%} | [{float(s['R_W_CI_low']):.2%}, {float(s['R_W_CI_high']):.2%}] | {e['marginal_equivalent']} |")
baselines=[]
for r in original:
 baselines.append(f"| {r['instance']} | {float(r['Train_LogLoss']):.4f} | {float(r['validation_LogLoss']):.4f} | {float(r['JSD_proxy']):.4f} | [{float(r['baseline_JSD_CI_low']):.4f}, {float(r['baseline_JSD_CI_high']):.4f}] | {r['Train_converged']} | {r['Train_Validation_mismatch']} | {r['denominator_valid']} |")
pathrows=[]
for r in path:pathrows.append(f"| {r['model']} | {r['K']} | {float(r['final_SWD']):.4f} | {float(r['mean_PathRatio']):.3f} | {r['distance_increase_steps']} | {float(r['maximum_rebound']):.4f} | {float(r['final_minus_best']):.4f} |")
eqrows=[]
for m in models+['Constant','RealDonor']:
 e=eqrow(m);eqrows.append(f"| {m} | {float(e['MMD']):.4f} | {float(e['Delta_eq']):.4f} | [{float(e['Delta_CI_low']):.4f}, {float(e['Delta_CI_high']):.4f}] | {e['marginal_equivalent']} |")
text='''# P48：冻结多步对抗迁移的四路线统一评价

日期：2026-11-19。实验性质：已有模型冻结推理、独立审计分类器训练、统计审计。不是新的迁移模型训练或未暴露盲测。

## 1. 结论

**没有得到新的90%差异消除或完整迁移通过结果。**A_K1在本轮组等权SWD中观察到约49.77%的缩减（95%组Bootstrap区间40.85%–56.67%），有实际改善，但距离90%仍明显。JSD代理评价族不完整有效：Logistic与MLP基线触发停止条件；基线有效的三个GRU实例仍完整保留候选结果，不冒充整个预注册家族通过。12个候选均没有通过本轮边际等价性置信区间规则。

K已核实为共享同一向量场、相同[0,1]时间区间的Euler积分步数。更多K不等于更多对抗训练或更强迁移。本轮没有替换历史C0协议，也没有改变原D-only封存版。

## 2. 数据、权重与保护

实际核验：Real Train5153窗/32Session、Validation780窗/4Session；Natural Train4049窗/50Root、Validation905窗/10Root，五任务各2Root。输入输出均为float32[N,4,712]。训练域先验各0.5，域内每Session/Root总权重相同，每组内窗口均匀；没有阶段再加权。分布路线也以组等权为主，不能与P46窗口边际距离的数值直接混为同一估计目标。

仅读取注册Train/Validation数值。P44/P45权重、配置、原结果与P46/P47产物Hash核验；不访问服务器或实机，不修改数据/原C0/Scaler，不读取Test/Holdout。

3个冻结Generator的K1/2/4/8分别生成Train4049窗和Validation905窗，共24份终点文件。新增Validation与原存档全部逐位相等，输入ID/Root/阶段完全一致；B仍是P44原模型，未算为独立P45模型。保存Validation每个实际积分状态的latent与完整712D解码。

主分析用P46固定log1p、RealTrain均值与max(std,0.05)，同一Scaler全部方法共享。约86.62%坐标触及下限，距离只在这个数值几何下解释，不是压力/力单位。审计神经网络使用float32；当前统计脚本将该固定变换转为float32再提升double计算距离，有约float32舍入，不是重新拟合或方法专属归一化。

## 3. K的真实含义与两条时间轴

原文件：P44/code/model.py（Generator.integrate/VectorField.forward）及同实现的P45/code/model.py。每步执行 `h <- h + F(h,z_s,t)/K`，t=j/K，h0为目标latent坐标零向量。F中原GRU基项与修正GRU只实例化一次。共享参数逐张量Hash、实际forward-hook调用次数/对象ID和时间输入见shared_parameter_hashes.csv、shared_parameter_call_trace.json。

必须区分：

- 每个触觉窗的4帧是采集时间轴；Real不均匀保留采样，Natural约20Hz。
- t∈[0,1]是模型积分参数，不是采集秒数。每个积分状态都包含完整4帧。

**Dr(h0=0)不是Original Sim。**它是合法内部状态经原冻结Real Decoder的额外诊断读出，所有输入初始相同。路径图将它与Original单独标记。Original→Dr(0)不算Euler更新，也不把K步累计成新增仿真时长。full_pipeline_path_metrics.csv额外保存包含该起始跳变的全链路长度。

## 4. 路线A：JSD代理与停止条件

[原GAN理论](https://arxiv.org/abs/1406.2661)在等先验和最优判别器下对应CE*=ln2−JSD。实际冻结生成器后的有限审计分类器只给**分类器受限、且受泛化/校准影响的held-out代理**，不能把它当真实JSD，也不能把负值裁零。

复用原Probe架构：MLP2848→192→64→1、ReLU/Dropout0.15；GRU712→128、LayerNorm/Dropout0.15；Logistic含L2/C=1等效项。全部重新初始化，未加载C0权重。MLP/GRU固定Seed11/12/13、AdamW lr1e-3/wd1e-3/batch64/clip5、固定40Epoch完整遍历，唯一Checkpoint是第40轮。Validation每轮仅记录，不作早停/选择。Logistic固定最多1000次float64 LBFGS，检查最大梯度≤1e-6且未到迭代上限。

训练前固定停止规则：最后5轮Train CE极差>.02记未收敛；Val CE−Train CE>.2标记泛化失配；J0≤.01或1000次组Bootstrap下界≤0，分母不可靠。它们是本轮审计规则，不是自然界通用常数，也没有根据结果修改。没有因未通过而延长训练、增加Seed或换分类器。

| Original审计实例 | Train CE | Val CE | 有符号J0 | J0组95%区间 | 训练收敛 | 泛化失配 | 分母/基线可用 |
|---|---:|---:|---:|---|---|---|---|
'''+ '\n'.join(baselines)+'''

Logistic的J0本身为正且优化收敛，但Train/Validation CE差约0.2825，触发预注册泛化差距检查；其余候选的Logistic分支停止。三个MLP的Validation CE>ln2，得到负J0且区间跨0；不是原始分布“负差异”，而是这个估计方案失效，其后续分支停止。**这四个实例×12候选未训练，记NOT_TRAINED_BASELINE_INVALID，没有伪造JSD或分类结果。**

三个GRU Original J0均为正、组区间排除0，继续完成同预算候选审计。它们的结果不能替代缺失的Logistic/MLP家族，因此不能宣称预注册家族整体支持90%。metrics_jsd.csv仍列出全部91个计划位置、真实训练/停止身份，完整逐Seed/CE/JSD/BA/AUC与区间没有只保留有利实例。

probe_group_audit.csv保留每个验证Session/Root的CE与加权贡献。少数高置信错误会使CE很大，即使AUC仍较高；负JSD代理是泛化/概率损失估计失效的信号，不能推断真实两域差异为零。

1000次Bootstrap按4Real Session和任务内10Sim Root重抽，同一Bootstrap身份在Original及所有候选共用，保留相对率的相关性。区间条件于已训练审计模型，不包含重新训练模型的不确定性；有限3Seed另行报告。负代理、无效分母和原始无效比值保留并标无效，不通过平移/截断制造90%。

## 5. 路线B与统一对照

固定Seed4803的1024个2848维单位方向，所有方法/负对照共享。每方向精确计算组等权一维经验CDF的Wasserstein-1，再平均为SWD。不是完整高维W1，也不是先选最好方向。每个方法首方向与scipy独立加权W1核对；全部方向保留swd_per_direction.csv。

Original SWD=0.411343，组Bootstrap区间[0.34635,0.55748]，分母有正支持。区间固定投影、Scaler和模型，只重抽真实Session/Root，不把1024方向当1024独立采集组。

下表JSD和R_JS列展示三个实际GRU Seed的范围，仅为紧凑显示，**不是代替完整7实例结果或家族通过判定**。SWD为组等权主分析。

| 方法 | GRU JSD代理范围 | GRU R_JS范围 | SWD | SWD缩减率 | 缩减率95%区间 | 边际等价性通过 |
|---|---:|---:|---:|---:|---|---|
'''+ '\n'.join(rows)+'''

SWD的观察排名A优于B优于C，三组内K1距离最低，但只是固定开发数据的观察，不据此重新挑Checkpoint或替换最终步。A约49.4%–49.8%的缩减有正区间支持；B/C的缩减区间可含0，不能对它们宣称稳健改善。Constant也能缩减SWD约19.1%、Real donor约46.5%，再次说明单一运输距离不是完整迁移质量。

K1下三个GRU Seed的JSD代理排名均为B优于A优于C，而SWD为A优于B优于C，实际不一致；其他K的逐Seed排名见metric_rankings.csv。unified_metrics.csv保留每个实例和Seed，不用平均值强行消除不一致。它反映“在这些有限审计分类器下的可分性”与“随机投影平均运输成本”不是同一个估计对象。没有任何90%差异缩减的联合证据。

## 6. 路线C：真实路径、反弹与读出

主路径在同一个目标latent坐标中积分，再用原Real Decoder解码；没有计算独立Real/Sim Encoder之间的latent欧氏距离。

| 组 | K | 终点SWD | 解码路径长度比均值 | 距离上升步数 | 最大反弹 | 终点−最佳中间点 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(pathrows)+'''

Dr(0)的SWD约0.4881，高于Original的0.4113。A_K8相对自身最佳中间点反弹约0.00213；B_K8约0.02216；C_K8路径中最大单步反弹约0.07067，部分相邻读出更新的平均方向余弦为负。没有用最佳中间步替换原终点。

**不能把解码路径绕行直接归因于latent积分振荡。**同一目标latent路径长度比均约1.0001，而解码后A/B/C的K8比值分别约2.192/1.616/4.911。K1/2/4的latent终点到K8终点距离逐渐减小，支持网格细化解释；K8不是精确ODE真解。Decoder包含硬Mask和非线性条件幅值，读出可以把平滑latent变化变成跳变。latent_path_diagnostics.csv同时记录原Mask激活变化，属于机制线索，不是已完成因果消融。

所以，已有证据是“输出空间路径存在反弹及绕行，潜空间相对直接”；没有证据称K8是8轮更强训练、材料物理振荡或必然过度迁移。

## 7. 路线D：匹配4-vs-10组的边际等价性

RealTrain32Session固定每组抽7窗，1000次互不重叠的4Session-vs-10Session划分，双方组等权，复用P46固定多尺度RBF/MMD V平方根合同。1000次划分仍只有32独立Session，不是1000独立校准实验。

epsilon_RR=0.368571；真实Session聚类重采样后重新校准的95%区间约[0.32049,0.40728]，相对宽度约23.55%。每个Bootstrap先重抽32Session身份，再从其经验频数支持中做1000次有权重无放回的互斥4+10唯一Session划分，保留不跨侧复用同一Session的要求。它是固定7窗下的经验组抽样校准，不包含所有采集协议/物理条件不确定性。

Validation同样固定每Session/Root7窗，计算整体MMD−epsilon，不计算RVC或Root通过率。1000次组Bootstrap同时重抽RealVal4Session、任务内SimVal10Root并重新估计RR容限。按原要求使用95%双侧百分位区间上界<0判边际等价。

| 方法 | MMD | MMD−epsilon | 差值95%组Bootstrap区间 | 边际等价 |
|---|---:|---:|---|---|
'''+ '\n'.join(eqrows)+'''

A_K1点估计与容限非常接近，但区间跨0且上界约0.246，**不能证明等价，也不能据此证明绝对不等价**。B/C的区间大多在容限上方。Constant被这项匹配设计拒绝；donor点估计低于容限，但上界仍为正。当前12候选均未满足指定通过规则。

RR校准在这批固定窗口/组抽样下可计算、1000聚类重复均有效；32Session、4RealVal Session仍不足以认证跨所有条件的通用自然容限。Session与Root构成、任务/采集协议差异依旧存在。百分位MMD V区间还受非负距离与小组数重采样影响；未事后改Bootstrap、核或容限以获得通过。

## 8. 跨路线质量与负对照

直接复用经Hash核验的P47完整PRDC四个k、HSIC三带宽、Root/Stage诊断；新增组等权幅值/零值/时序/逐Root多样性统计见quality_metrics.csv。P47的PRDC采样口径仍是其原合同，不伪称它也是本轮新组等权分布指标。**P47没有一个通用联合通过阈值，本轮也没有凭空增加。**

- Constant：逐Root仅一个不同输出，HSIC=0；P47完整空间Precision=1但Recall=0，不能因SWD下降就接受。
- Real donor：边际真数据来源不保证与RealVal同分布；HSIC Root内约0.1162，接近其置换基线0.1157，缺输入依赖。
- Pair Shuffle：同Root/Stage内置换不改变每Root边际多重集，因此组等权SWD也应完全不变；小样本MMD使用逆置换索引保留相同输出子集。P47对PRDC/MMD/Energy逐值核验；P48再次保存SWD多重集核验。它的配对损坏依靠HSIC及受限块诊断，而不是要求边际距离拒绝。
- P47时序摘要仍有整窗反转盲点；不能宣布联合评价已识别所有物理错误。

模型输出finite、非负并不保证质量。C的幅值/多样性收缩，A的尾部和时间变化异常，任务/Stage关联不等于实机语义，这些历史问题没有被一次新距离评价自动解决。

## 9. 最终十一问

1. **Original JSD基线可靠？**不是整个家族都可靠。三个GRU有正且可区分的J0；Logistic虽正但泛化检查失败；MLP负代理/分母区间不可靠，按规则停止。
2. **是否消除了原差异？**A缩小了组等权SWD，未消除全部差异；局部支持/输入保持和JSD不能由此推断通过。
3. **JSD消除率达到90%？**没有整体家族支持；有效GRU的完整候选结果见逐实例表，不能拿失败分支或负代理凑比值。
4. **SWD缩减90%？**没有；最佳观察A_K1约49.77%，区间上界约56.67%。
5. **两者排名一致？**不一致。K1下三个GRU Seed均观察JSD代理B优于A优于C，而SWD为A优于B优于C；其他K逐Seed完整列于metric_rankings.csv，不合并成人为总分。
6. **哪个K最好？**SWD中各组均K1为最低观察终点；不是模型选择或额外训练批准，不证明所有指标均K1最好。
7. **K增加含义？**同一共享场、固定区间的数值积分网格细化，不是增加对抗训练轮数或迁移强度。
8. **是否振荡/反弹？**解码后的距离和更新存在反弹/反向；latent路径相对直接，不能直接归因为流场失稳或物理振荡。
9. **进入RR范围？**按本轮指定的置信区间规则，没有候选获得边际等价证据；A点估计接近但证据不足。
10. **负对照识别？**Constant的多样性/依赖失败，donor依赖近零基线，pairshuffle边际不变而条件依赖下降；仍不能认证所有语义错误都能检测。
11. **明确障碍？**审计分类器跨Session泛化/校准、完整局部支持与平均距离的矛盾、Decoder硬Mask非线性读出、幅值时序及输入条件保持、缺共同物理语义与独立开发外验证。

已获支持：A的本轮SWD改善、K积分语义、部分读出反弹与负对照限制。初步线索：Decoder对平滑latent的非线性放大。尚不能判断：真实JSD、全物理条件等价、总体Root泛化。被负对照否定：把单一距离缩减/高Precision当成完整条件迁移成功。

## 10. 实际训练与复现

'''+f"迁移模型训练Epoch=0、Optimizer Step=0。独立审计分类器实际训练{counts['actual_audit_models']}个，按基线停止{counts['skipped_audit_models']}个；深度模型累计{counts['actual_deep_epochs_total']} Epoch、{counts['actual_deep_optimizer_steps']} Optimizer Step，Logistic累计{counts['actual_Logistic_LBFGS_steps']}次LBFGS迭代。每个实际深度实例固定40轮，未延长；详见actual_training_counts.json和probe_training_audit.csv。合成张量设备计时不算研究数据训练，单独记录。\n\n"+'''
复现代码与run_commands.json记录全部步骤。新目录复制code/后在原项目根目录运行，原注册数据/Checkpoint需存在。source Hash、统计投影、Bootstrap索引、共享参数Hash、训练日志、终点权重、预测分数、每步解码及输出索引均保留。被停止的分支写明NOT_TRAINED，不能当成实际完成模型。

P48不是全新盲测；当前Validation已被历史开发使用。组Bootstrap区间均条件于固定模型、数据注册和测量合同，不包含所有训练选择/历史暴露影响。没有修改历史验收标准，不commit/push。

**P48到此停止；未自动开始P49或其他模型训练。**
'''
(O/'REPORT.md').write_text(text);save(O/'report_complete.json',dict(figures=6,summary=str(O/'summary.json'),no_new_090_claim=True));print('P48 REPORT COMPLETE',flush=True)
