from support47 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess,sys,time,platform
assert all((O/f'{name}_complete.json').exists() for name in ['prdc','dependence','rr_bootstrap'])
def read(name):return list(csv.DictReader((O/name).open()))
pr=read('prdc_metrics.csv');hs=read('hsic_metrics.csv');wr=read('hsic_root_diagnostics.csv');ret=read('task_stage_retention.csv');tm=read('temporal_diagnostics.csv');rr=read('real_real_prdc.csv');boot=read('bootstrap_results.csv');neg=[];summary=[];rootpr=read('root_metrics.csv');config=json.loads((O/'experiment_config.json').read_text())
def pget(model,k=5,space='full2848'):return next(r for r in pr if r['model']==model and r['space']==space and int(r['k'])==k)
def hget(model,scope='withinRoot equalmacro',scale=1):return next(r for r in hs if r['model']==model and r['scope']==scope and float(r['bandwidth_scale'])==scale)
def tget(model):return next(r for r in tm if r['model']==model and r['Root']=='ALL' and r['stage']=='ALL')
def taskget(model):return next(r for r in ret if r['model']==model and r['information']=='Task')
def stagevalue(model):return float(np.mean([float(r['normalized_HSIC']) for r in ret if r['model']==model and r['information']=='Stage']))
models=['Original']+[f'{g}_K{k}' for g in ['A','B','C'] for k in [1,2,4,8]]+['Constant','RealDonor']+[f'{g}_K{k}_PairShuffle' for g in ['A','B','C'] for k in [1,2,4,8]]+[m+'_TemporalShuffle' for m in ['Original']+[f'{g}_K{k}' for g in ['A','B','C'] for k in [1,2,4,8]]]+['Original_TemporalReverse','A_K1_TemporalReverse']
for model in models:
 p=pget(model);h=hget(model);g=hget(model,'all905window global descriptive');st=hget(model,'withinRootStage rootmacro');t=tget(model);task=taskget(model);row=dict(model=model,PRDC_reference_k=5,k5_is_display_only=True,precision=float(p['precision']),recall=float(p['recall']),density=float(p['density']),coverage=float(p['coverage']),HSIC_global=float(g['normalized_HSIC']),HSIC_withinRoot=float(h['normalized_HSIC']),HSIC_withinRootStage=float(st['normalized_HSIC']),HSIC_Root_CI_low=float(h['CI_low']),HSIC_Root_CI_high=float(h['CI_high']),Task_Root_HSIC=float(task['normalized_HSIC']),Stage_withinRoot_HSIC=stagevalue(model),lag1_correlation=t['lag1_correlation'],absolute_frame_delta=float(t['mean_absolute_frame_delta']),physical_semantics_certified=False,formal_pass='NOT_DEFINED',new090_acceptance=False);summary.append(row)
 if model not in ['Original']+list(candidate_paths()):neg.append(dict(row,control='constant' if model=='Constant' else 'donor' if model=='RealDonor' else 'pair_shuffle' if model.endswith('PairShuffle') else 'temporal_reverse' if model.endswith('TemporalReverse') else 'temporal_shuffle',independent_physical_roots=False,interpretation='diagnostic controls derived from registered development samples'))
table(O/'evaluation_summary.csv',summary);table(O/'negative_control_metrics.csv',neg)
# Conditional bootstrap intervals are kept separate from MC draw spread and are not formal margins.
brows=[]
for file in ['bootstrap_results.csv','real_real_bootstrap.csv']:
 grouped={}
 for r in read(file):grouped.setdefault((r['model'],r['space'],r['k']),[]).append(r)
 for (model,space,k),rows in grouped.items():
  q=dict(model=model,space=space,k=k,replicates=len(rows),scope=rows[0]['scope'])
  for key in ['precision','recall','density','coverage']:
   v=np.array([float(r[key]) for r in rows]);q[key+'_CI_low']=float(np.quantile(v,.025));q[key+'_CI_high']=float(np.quantile(v,.975))
  brows.append(q)
table(O/'bootstrap_interval_summary.csv',brows)
# Keep ALL RR k/cohorts and sampling dispersion, not universal acceptance cutoffs.
rrsummary=[]
for space in ['full2848','descriptor52']:
 for cohort in [f'reference_block_{j}' for j in range(4)]:
  for k in KS:
   rows=[r for r in rr if r['space']==space and r['cohort']==cohort and int(r['k'])==k];q=dict(space=space,cohort=cohort,k=k,N_each=rows[0]['N_each'],draws=100,independent_Sessions=32)
   for key in ['precision','recall','density','coverage']:
    a=np.array([float(r[key]) for r in rows]);q[key]=float(a.mean());q[key+'_draw_std']=float(a.std())
   rrsummary.append(q)
table(O/'real_real_prdc_summary.csv',rrsummary)
# Explicit common-condition audit: no fabricated physical correspondences.
condition=[dict(dimension='Task',Real='hardness/slip/stable acquisition protocols',Sim='lift/pick_place/stack/push/nut_assembly',common_semantics='NOT_CERTIFIED',evidence='P46/session_configs.json; C3 natural_sim_validation/windows_manifest.csv',missing='same manipulation/contact protocol and independently registered correspondence'),dict(dimension='Stage',Real='CLOSE/HOLD/LIFT retained-frame labels',Sim='task-specific stage from actual recorded sourceEpisode',common_semantics='NAME_ONLY',evidence='registered real_stage and natural stage arrays; manifest',missing='same physical event definition and comparable control protocol'),dict(dimension='Intensity',Real='processed nonnegative sensor response',Sim='project processed MuJoCo touch response',common_semantics='NOT_CERTIFIED',evidence='P44REPORT input audit; P46condition_audit.csv',missing='traceable common force/response units and calibration'),dict(dimension='Time',Real='strictly increasing irregular retained samples',Sim='continuous source about20Hz',common_semantics='ORDER_ONLY',evidence='C24 train/validation_time_contract.npz;Natural timestamp_first/last',missing='matched physical sampling intervals; derivative-time interpretation unsupported'),dict(dimension='Sensor position',Real='Usb32 mapping referenced but original driver unavailable',Sim='canonical712 ordering from simulation processor',common_semantics='UNKNOWN',evidence='P46/session_configs.json mapping_file;P44 mapping audit',missing='certified physical Taxel position/orientation/channel correspondence'),dict(dimension='Output representation',Real='full2848 indexed log responses',Sim='same mathematical transform, no sharedphysicalsemantic certification',common_semantics='NUMERICAL_ONLY',evidence='P46/RealTrain_preprocessing.npz',missing='validated common output/condition embedding; dimensional equality insufficient'),dict(dimension='FJD decision',Real='none',Sim='none',common_semantics='NOT_APPLICABLE',evidence='prerequisites above unmet',missing='do not concatenate unrelated latent or run GMM/OT')];table(O/'condition_semantics_audit.csv',condition)
fig=O/'figures';fig.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'figure.dpi':140});display=['Original','A_K1','A_K2','A_K4','A_K8','B_K1','C_K1','Constant','RealDonor'];color=['#555','#148a82','#148a82','#148a82','#148a82','#cc7838','#7d60ad','#b33','#4674aa']
f,axs=plt.subplots(2,2,figsize=(12,8))
for ax,key in zip(axs.ravel(),['precision','recall','density','coverage']):
 for m,c in zip(display,color):ax.plot(KS,[float(pget(m,k)[key]) for k in KS],marker='o',label=m,color=c,alpha=.8)
 ax.set_xticks(KS);ax.set_title(key+' / full2848');ax.set_xlabel('k')
axs[0,0].legend(fontsize=7,ncol=2);f.tight_layout();f.savefig(fig/'prdc_all_k.png');plt.close(f)
f,axs=plt.subplots(1,2,figsize=(12,4))
for ax,space in zip(axs,['full2848','descriptor52']):
 for j,key in enumerate(['precision','recall','coverage']):ax.plot(range(len(display)),[float(pget(m,5,space)[key]) for m in display],marker='o',label=key)
 ax.set_xticks(range(len(display)),display,rotation=60);ax.set_ylim(-.03,1.05);ax.set_title(space+' k5 illustrative only');ax.legend()
f.tight_layout();f.savefig(fig/'representation_sensitivity.png');plt.close(f)
f,axs=plt.subplots(1,2,figsize=(12,4));depdisplay=['Original','A_K1','B_K1','C_K1','Constant','RealDonor','A_K1_PairShuffle','B_K1_PairShuffle','C_K1_PairShuffle']
for ax,scope in zip(axs,['all905window global descriptive','withinRoot equalmacro']):
 ax.bar(range(len(depdisplay)),[float(hget(m,scope)['normalized_HSIC']) for m in depdisplay]);ax.set_xticks(range(len(depdisplay)),depdisplay,rotation=65);ax.set_ylim(0,1.05);ax.set_title(scope)
f.tight_layout();f.savefig(fig/'input_dependence.png');plt.close(f)
f,axs=plt.subplots(1,2,figsize=(11,4))
for ax,key in zip(axs,['lag1_correlation','mean_absolute_frame_delta']):
 labels=['Original','Original_TemporalShuffle','Original_TemporalReverse','A_K1','A_K1_TemporalShuffle','A_K1_TemporalReverse'];ax.bar(range(len(labels)),[float(tget(m)[key]) for m in labels]);ax.set_xticks(range(len(labels)),labels,rotation=65);ax.set_title(key+' (retained-frame index)')
f.tight_layout();f.savefig(fig/'temporal_negative_controls.png');plt.close(f)
f,ax=plt.subplots(figsize=(9,4))
for j,m in enumerate(['Original','A_K1','B_K1','C_K1','RealDonor']):
 vals=[float(r['precision']) for r in rootpr if r['model']==m and r['space']=='full2848' and int(r['k'])==5];ax.scatter(np.arange(10)+.05*j,vals,label=m,s=20)
ax.set_xticks(range(10),[f'Root{i}' for i in range(10)]);ax.set_ylabel('Precision k5 (size-unequal Root diagnostic)');ax.legend();f.tight_layout();f.savefig(fig/'root_precision.png');plt.close(f)
f,ax=plt.subplots(figsize=(10,4))
for key in ['precision','recall','coverage']:
 vals=[float(r[key]) for r in rrsummary if r['space']=='full2848' and int(r['k'])==5];ax.plot(range(4),vals,marker='o',label=key)
ax.set_xticks(range(4),[f'Reference{i}' for i in range(4)]);ax.set_title('Real-Real Session-disjoint; k5 descriptive, not thresholds');ax.legend();f.tight_layout();f.savefig(fig/'real_real_reference.png');plt.close(f)
# Derived table snippets: no model selection or new acceptance threshold.
mainrows=[]
for m in models:
 if m.endswith(('PairShuffle','TemporalShuffle','TemporalReverse')):continue
 p=pget(m);h=hget(m);mainrows.append(f"| {m} | {float(p['precision']):.4f} | {float(p['recall']):.4f} | {float(p['density']):.4f} | {float(p['coverage']):.4f} | {float(h['normalized_HSIC']):.4f} | {float(taskget(m)['normalized_HSIC']):.4f} | {stagevalue(m):.4f} |")
negrows=[]
for m in ['Constant','RealDonor','A_K1','A_K1_PairShuffle','A_K1_TemporalShuffle','A_K1_TemporalReverse']:
 p=pget(m);h=hget(m);t=tget(m);negrows.append(f"| {m} | {float(p['precision']):.4f} | {float(p['recall']):.4f} | {float(p['coverage']):.4f} | {float(h['normalized_HSIC']):.4f} | {float(t['lag1_correlation']):.6f} | {float(t['mean_absolute_frame_delta']):.6g} |")
rrtab=[]
for q in rrsummary:
 if q['space']=='full2848' and q['k']==5:rrtab.append(f"| {q['cohort']} | {q['N_each']} | {q['precision']:.4f} | {q['recall']:.4f} | {q['density']:.4f} | {q['coverage']:.4f} |")
# Machine-readable method decision does not invent a combined acceptance score.
save(O/'decision.json',dict(decision='B',description='部分指标有价值；需要改进统计设计和语义条件证据，尚不具备新0.90验收基础',PRDC='CONDITIONAL_GO descriptive fidelity/coverage with strong k/representation limits',HSIC='CONDITIONAL_GO statistical dependence only; Root/Stage offsets and temporal null assumptions remain',TaskStage='DESCRIPTIVE_ONLY;2Roots/task;not semanticcorrectness',FJD='NOT_APPLICABLE',new_090_defined=False,generator_accepted=False,P48_started=False))
report='''# P47：条件触觉迁移评价体系可行性实验

日期：2026-11-18。性质：评价方法与负对照验证。没有生成模型训练、C0推理或新验收通过声明。

## 1. 决策

**B：部分指标有价值，但还需要改进统计设计和条件语义证据。当前不能重新定义或宣布新的 0.90 迁移验收。**

PRDC补充边际分布的局部支持与覆盖，HSIC补充输入依赖，任务/阶段统计补充Sim内部信息诊断。它们测量不同对象，不能简单加权成一个“90分”。能识别部分退化并不证明完整真实触觉语义正确；本轮没有新的统一通过阈值。

## 2. 数据及历史保护

实际重新加载并核验：Real Train 5153窗口/32Session，Validation780窗口/4Session；Natural Train4049窗口/50Root，Validation905窗口/10Root，五任务各2Root。Real仅加载原注册Train/Validation行；未读取Test/Holdout数值数组。12份已有A/B/C、K1/2/4/8输出逐文件SHA、shape、Root/Stage/窗口ID与源数组核对。

B继续引用P44原输出；没有把P45复用B当新实验。原始数据、模型、C0、Scaler和P44–P46产物只读。Hash校验不等于模型运行。新增结果全部在本目录。

P44/P45报告与实现显示其训练本来已有索引Patch份额/相邻相对响应保持损失，本轮没有把“加入保持”误称为尚未尝试的全新方法，也没有按这些评价反向更新模型。

## 3. 固定评价定义

### PRDC

使用[作者实现](https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py)的原定义，遵循[Naeem等，ICML2020](https://proceedings.mlr.press/v119/naeem20a.html)和[生成模型Precision/Recall论文](https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html)。

设r_i为Real第k近邻半径，s_j为生成数据第k近邻半径，d_ij为跨域欧氏距离：Precision=生成点满足某个d_ij<r_i的比例；Recall=Real点满足某个d_ij<s_j的比例；Density=平均每个生成点落入Real球的个数/k；Coverage=满足min_j d_ij<r_i的Real点比例。保留原严格小于号，不为重复点/常量另设epsilon。邻居半径按包含self的第k索引取值。Precision/Recall/Coverage在[0,1]，Density可以大于1。

k=3/5/10/20全部报告，k5仅用于下文紧凑展示，不是选定验收k。完整空间使用P46冻结log1p、RealTrain均值和std下限0.05，保留全部2848维。86.62%坐标触及下限，不能把该几何当单位无关或物理力差。

辅助52维固定描述子：每帧均值/std/正值比例/5个分位数共32维，3对相邻帧绝对差的均值/std/中位数/P90共12维，全窗均值/std/正值比例/5个正值分位数共8维。仅用RealTrain拟合其独立尺度。该对照同时改变表示和几何，**不能据此单独归因于标准化下限**，也不替代完整空间。

每次生成侧从905按Root原数量比例、最大余数定额抽780，无放回；保持原任务比例而非人为均衡任务。100次固定种子抽样共享于全部候选。保存均值、中位数、标准差及抽样P2.5/P97.5；这只是MC抽样离散程度，不是100次独立实验置信区间。Root逐项结果使用原全部窗口，对780Real参考，样本量不等，明确仅作诊断。

300次Bootstrap以4个Real Session、任务内10个Sim Root为单位。为避免重复采集组在kNN中产生人工零邻域，采用**draw0固定邻域下逐点贡献的聚类重加权**；它不重新估计kNN邻域，不能当完整PRDC估计量的无条件区间。区间与100抽样均值分开保存，且只有4个Real Session。RR基线也保存Session聚类的同类条件Bootstrap。

### HSIC

X/Y分别构造RBF核，使用各自正成对距离中位数，固定0.5/1/2倍带宽。计算centeredK与centeredL的内积/(n-1)^2，以及其归一化核对齐值[0,1]。常量核无可测变化，归一化值按0记录并附constant_kernel标记，不能解读成正确或错误语义。

核带宽按预先规定规则从本轮开发Validation各数据自身无监督计算，未用成绩搜索；它**不是Train拟合的新盲测评价**。没有改变P46原Scaler。

同时报告全905窗描述性依赖、10Root均值依赖、Root内宏平均、Root+Stage块内宏平均，避免仅测到Root身份。Root+Stage块至少4窗口，小块不用于该项，但全量数据仍用于其他指标；不是删除不利样本。Root内采用199次非零循环移位轨迹零假设对照，不假定905窗iid；边界回卷与轨迹非平稳性导致可交换性未认证，tail fraction只能作探索性诊断。Root均值另穷举每任务2Root的32种交换，分辨力很有限。归一化值高只说明依赖强，不证明对应方向或接触位置正确。

任务使用10个Root均值与类别核，并按Root置换标签；阶段只用Sim内部原标签做Root内类别核统计。均不训练神经网络。

## 4. 全部候选的主表

下表PRDC为完整2848维、k5的100次均值；HSIC为Root内归一化宏平均；Task为10Root均值类别关联；Stage为各Root内部阶段关联宏平均。所有其他k/带宽/描述子/逐Root结果在CSV，不能用此表单项判通过。

| 方法 | Precision | Recall | Density | Coverage | 输入依赖HSIC | Task关联 | Stage关联 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(mainrows)+'''

PRDC各指标可能排序不同：P46的平均MMD/Energy下降，并不推出局部支持、召回和覆盖会同时提高。Constant的生成内部半径为0，严格原公式下Recall为0；Precision或Density是否仍偏高要看实测，不能把高Precision当模式多样性。

Root数量不同于窗口数量。每Task只有2个Validation Root，任何Task保持结论都受Root/任务混杂限制；同Root内Stage也与时间顺序、接触幅值和动作过程耦合。

## 5. Real–Real局部支持基线

使用P46固定的四个8Session参考组，对其互斥24Session补集。每侧等量无放回抽样，100次，完整保留组身份；下表为k5均值，其余k/特征空间全部保存。

| RR参考组 | 每侧窗口数 | Precision | Recall | Density | Coverage |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rrtab)+'''

这些不是普适阈值。局部邻域对组构成、重复/近邻窗口、k、样本量和高维表示敏感；8对24Session与10Root对4Session不是可交换实验。不可通过选择最宽RR基线放宽迁移门槛。

## 6. 负对照与数学核验

| 对照/候选 | Precision | Recall | Coverage | Root内依赖HSIC | lag1 | 相邻帧绝对差 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(negrows)+'''

- **Constant**：每输入输出同一个RealTrain均值窗，非零静态响应不等于存在输入依赖。其PRDC和HSIC反应互补，不能只看Fidelity。
- **Real Donor**：每输入独立随机取一个RealTrain窗，保留边际真实来源而不保留源对应。它是检验“像Real不等于条件迁移”的必要反例。随机有限样本HSIC不必恰为0，需与置换零分布、Root内结果共同看。
- **Pair Shuffle**：每个候选在同Root、同Episode、同Stage内打乱窗口对应；singleton保留。alignment_audit.csv记录每块范围及固定点。每窗口内部四帧不变。全905输出多重集和PRDC逐值核验不变；100次子集评价采用逆置换后的索引以保留同一个输出子集。因此纯边际指标不应、也没有被用于识别配对损坏。它仍保留Root与Stage信息，若全局HSIC仍高，不说明对应正确。
- **Temporal Shuffle**：所有原候选采用固定[0,2,1,3]，不搜索最差顺序。另对Original和A_K1加入反转[3,2,1,0]，检验时序指标的盲点。改变完整2848维的帧位置可以改变PRDC，但不保证有足够功效。
- **时间方向盲点**：反转不改变相邻差绝对值之和、窗内极差；lag1 Pearson在交换前后向量时也可完全不变。即使三个数不变，也不能据此声称顺序保持。按保留帧计算的这些统计不能解释为物理每秒变化率；Real不均匀采样，时间间隔差异仍未消除。

所有负对照完整结果和全K指标保留。没有预设统一“合格”阈值，因此不能说某个控制通过全部门槛；可以明确指出某个指标无法区分特定退化。没有证明该有限控制集覆盖所有作弊或物理错误方式。

### 6.1 实测解释：不能只看表格中的大数字

**PRDC对Constant的证伪有效，但不是每一项都有效。**完整2848维下Constant在所有k的Precision均为1，Recall均为0，Coverage仅0.0026–0.0051。Density为0.2–0.667而非必然很大，因此“Density高就坍塌”也不是一个可靠规则。辅助描述子下Constant四项全为0，表明表示几何影响很大。

**A并未在完整空间获得一致的局部质量改善。**以K1为例，完整空间k=3/5/10/20的A Recall为0.0297/0.0399/0.0527/0.1009，Original为0.0748/0.1958/0.2819/0.2903；A均更低。A Coverage在这四个k稍高，Precision只在k20高于Original。辅助描述子下A的四项却均较Original改善。这说明全局统计量接近并不等于完整Taxel时序局部支持接近，也不能将表示敏感性单独归咎于0.05尺度下限。

B_K1完整空间各k Precision/Density/Coverage均为0，但Recall随k从0.0142升到0.6353：生成侧邻域变大可以覆盖Real点，这不是Fidelity改善。C_K1完整空间四项在全部k均为0。全部K的原始结果保留，未据此换模型。

**Real donor也不能当作必然有高Precision的“真阳性”。**k5完整空间Precision仅0.0383、Recall0.6379；它来自RealTrain，不是同一Validation Session。RR基线的k5 Precision也只有0.0608–0.2331。来源真实不保证其位于另外4个Session的窄局部邻域，显示Session支持与近邻几何限制，不能解释成donor“不真实”。

**HSIC能发现部分配对损坏，但全局/组均值明显不够。**K1的Root内依赖A/B/C分别0.5071/0.3926/0.4292，同RootStage打乱后降到0.3322/0.2427/0.2883；去掉Root+Stage块偏移后分别0.6126/0.5905/0.5409，打乱后0.2301/0.2036/0.1868。但这些归一化估计有有限样本偏置，不能直接按0.90或0.5定门槛。

Real donor Root内HSIC为0.1162，其循环移位零分布均值0.1157、tail fraction0.45；Constant为0。这支持依赖评价具有诊断价值。另一方面，Real donor的10Root均值X–Y核关联竟为0.6733；其Task关联0.5009，任务置换基线均值0.5536、tail fraction0.79。只看一个Root均值关联的大数会误判。

受限Pair Shuffle保留阶段布局。A打乱后Root内HSIC仍高于不限制阶段的循环移位零分布（0.3322对0.2381），这不是打乱后仍有正确细粒度对应，而是现有零分布与被保留条件不同。应以RootStage块诊断和对照差值解释，不能把该tail fraction当认证的条件独立检验。

**Task与Stage信息只得到部分统计保持证据。**A_K1 Task核关联0.7439，原始Sim0.6428，但A Stage关联0.3211，低于原始0.4335。B/C也有任务关联，却没有足够边际Fidelity。RootStage内Pair Shuffle完全保留Task/Stage关联，再次说明这些量不能认证逐输入对应。每任务2Root且未做多重比较校正，置换结果不是新任务验收。

**时序控制确有盲点。**A_K1交换中间两帧后lag1从0.901824降到0.850563，绝对差从0.005106升到0.006934；但整窗反转后两项约不变。相同首末帧的中间交换还使平均有符号变化因望远镜求和保持不变。没有一个单独摘要能可靠检查所有顺序错误。

## 7. FJD前提审计

[FJD原论文](https://arxiv.org/abs/1907.08175)研究输出与条件的联合分布，本项目尚无认证的共同条件/输出物理表示。condition_semantics_audit.csv逐项列明任务、阶段、强度、时间、Taxel位置及证据。

**NOT_APPLICABLE。**不能把独立Encoder潜变量拼接，也不能因两侧都是712D就称其相同物理坐标。未拟合GMM、未解OT、未训练共享特征。缺少的是经过验证的共同接触协议、条件对应和公共输出语义，不能用数学上可计算一个协方差矩阵代替它们。

## 8. 指标能力和限制

| 指标 | 范围/单位 | 负对照能力 | 不能证明 |
|---|---|---|---|
| PRDC | P/R/C为[0,1]，Density可>1；窗口邻域，外层Session/Root | 可以揭示部分集中化/缺覆盖；不能辨Pair Shuffle或无条件Real donor的输入丢失 | 条件语义、任务正确、真实物理对应 |
| HSIC | 非负raw；归一化[0,1]；Root内/间分开 | 可揭示常量/部分随机配对丢失；需看受限置换及RootStage残留 | 输出接触位置正确、物理单调性或因果保持 |
| Task/Stage类别核 | 归一化[0,1]；10Root、Root内块 | 检查关联是否被抹去；Root/Stage内shuffle可以保留这些量 | Real任务对应、机器人操作成功率 |
| lag1/绝对差/极差 | retained-frame统计，不是每秒率 | 对部分顺序扰动有反应，对反转存在解析盲点 | 时间箭头、同采样协议下的材料动态 |

尚无一个经认证的联合任务标签集，也没有足够独立开发外单位为综合阈值做确认。因此不把指标加权平均、把相关性乘100或把某个PRDC值命名为新Domain Confusion。

## 9. 直接回答十个问题

1. **P46为何接纳Constant/donor？**RR包络包含跨Session/条件异质性，边际距离不检查输入对应；某些无信息或无条件输出仍可落入宽包络。
2. **PRDC能识别Constant/坍塌吗？**它比单一距离能揭示某些缺召回/覆盖情形，Constant的生成半径0使Recall=0；但其他指标可表现不同，没有通用坍塌检测保证。
3. **能区分Original与迁移吗？**可以产生不同的支持/覆盖描述，但这不是“迁移一定更好”；所有候选/k的差异见主表和曲线。
4. **A的改善伴随覆盖变化吗？**完整空间A_K1 Coverage略增，但Recall在全部4个k均下降；辅助描述子却全面改善，故没有跨表示一致的整体改善。
5. **HSIC能分辨配对吗？**本批A/B/C的受限shuffle会降低Root内和RootStage内HSIC，Constant为0、donor接近自身零基线；但Root身份/Stage与有限样本偏置仍可能造成高值，不能仅凭全局HSIC判通过。
6. **任务阶段保留吗？**部分保留：A_K1任务关联较Original高，但阶段关联较低；Pair Shuffle保留两者却破坏细粒度对应。只有每Task2Root，不能给总体保持率或实机任务结论。
7. **能证明Real触觉语义正确吗？**不能。输入依赖和标签关联可以与错误位置/反向强度映射共存。
8. **能算FJD吗？**不具备本课题所需的可解释共同条件前提，NOT_APPLICABLE。
9. **仍有漏洞吗？**有：边际指标对配对置换不敏感，相关性不能认证语义，常用时序摘要对反转有盲点，有限控制集不能证明无其他退化。该体系还没有综合验收有效性证据。
10. **足以重定义0.90吗？**不足。可保留多轴评价作为开发诊断；下一步需先固定条件语义/目标总体、独立组和对照有效性，再前瞻注册联合标准，不能在这10Root上事后命名90%通过。

## 10. 复现与保护

代码：design.py→prdc_eval.py与dependence.py→rr_bootstrap.py→report.py→finalize.py。前两统计脚本互不依赖，可顺序运行。所有随机种子、索引、控制置换、原文件Hash与代码Hash保存。将code/复制到新输出目录后，在原项目根目录、同Python环境复现；references/原作者代码只供核验，未运行其sklearn依赖，也没有训练Probe。

Bootstrap区间不与100次采样范围混淆，raw HSIC与归一化值不混淆；Real物理通道映射未知，源配对ID已核验。历史旧C0联合定义及D-only封存包不变。

**P47完成后停止，未启动P48。**
'''
(O/'REPORT.md').write_text(report)
# Save diagnostics separately to keep original raw CSV (including explicitly undefined correlations) intact.
undefined=[r for r in ret if r.get('Spearman','').lower()=='nan'];save(O/'undefined_statistics.json',dict(Spearman_undefined_rows=len(undefined),reason='constant descriptor ties; correlation mathematically undefined, never interpreted as zero or evidence',rows=undefined))
print('P47 REPORT COMPLETE',flush=True)
