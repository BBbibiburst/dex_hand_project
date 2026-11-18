from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rs):
 with (O/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
r=read('metrics.csv');comp=read('historical_instance_comparison.csv');stat=read('output_statistics.csv');dec=read('reconstruction_error_decomposition.csv');rec=json.loads((O/'reconstruction.json').read_text());done=json.loads((O/'complete.json').read_text());d0,d1=stat
passed=[x['instance'] for x in r if x['passed_joint']=='True'];failed=[x['instance'] for x in r if x['passed_joint']!='True'];gain=[x['instance'] for x in comp if x['C11_A_old_Real_reconstruction_passed']=='False' and x['C23_repaired_Real_reconstruction_passed']=='True'];lost=[x['instance'] for x in comp if x['C11_A_old_Real_reconstruction_passed']=='True' and x['C23_repaired_Real_reconstruction_passed']=='False']
# Scores already computed; no model, calibration or source tactile access here.
score=[]
for x in r:
 z=np.load(O/'predictions'/f"{x['instance']}.npz");p=z['probabilities']
 for label,v in [('reconstruction',p[:780]),('Real_reference',p[780:])]:score.append(dict(instance=x['instance'],group=label,mean=float(v.mean()),std=float(v.std()),exact_score_1_fraction=float((v==1).mean()),unique_scores=len(np.unique(v)),P01=float(np.quantile(v,.01)),P50=float(np.quantile(v,.5)),P99=float(np.quantile(v,.99))))
table('score_statistics.csv',score)
ga=json.loads((O/'bootstrap_group_audit.json').read_text());ga['synthetic_prior_draws_in_C23']=False;ga['context_note']='No synthetic prior samples; only original/reconstructed shared Real Sessions. Legacy generic helper annotations do not describe an additional C23 source.';save('bootstrap_group_audit.json',ga)
text='''# C23：C13修复后Real自重建的原冻结C0读出对照

日期：2026-11-18。**6/11联合通过；属于中间状态，目标侧读出尚未完整通过。** 本轮不训练模型或Probe、不读取Natural触觉/Test/Holdout、不修改原数据或协议。C23结束后停止，未开始下一实验。

## ① C23完整结果

输入是C13已保存且与归档Hash一致的`validation_outputs.npz`，以及原Real Validation注册的780行；4 Session、120 Trial、每窗4×712。C13 Encoder/Decoder文件未更改，也未重新生成数据。为避免读取同一源NPZ中其他历史角色的数值，只按注册索引流式读取这780行，未展开其他数值行或Sim键。

只执行原11个C0。使用原Scaler、log1p、通道顺序和0.5阈值，权重/参数全冻结。kNN仅使用原冻结模型内的训练参照库，没有打开或添加Natural Sim数据文件。原Real参考组=1、重建对照组=0，沿用C11 A的读出诊断；两者实际来源都仍是Real，不伪称重建来自Natural。

判据为BA[.45,.55]、AUC[.475,.525]、D≥.90。BA与D等价，保留双字段而不当成两份独立证据。

|实例|BA|AUC|D|联合通过|
|---|---:|---:|---:|---|
'''
for x in r:text+=f"|{x['instance']}|{float(x['balanced_accuracy']):.6f}|{float(x['roc_auc']):.6f}|{float(x['domain_confusion']):.6f}|{'是' if x['passed_joint']=='True' else '否'}|\n"
text+=f'''
**汇总：BA 11/11，D 11/11，AUC及联合门槛6/11。** 通过的是{', '.join(passed)}；其余{', '.join(failed)}均因AUC未过。BA/D高主要对应两个对照组都大多被判Real，不能据此忽略排序差异。正式AUC没有翻转、移动阈值或挑Seed。

300次Bootstrap将原值和重建值按**同一来源Session联合重采样**，不是两套独立组；只有4 Session，置信区间很有限。逐实例BA/AUC/D区间在`metrics.csv`，阶段结果在`stage_metrics.csv`，完整概率及混淆矩阵均保存。点估计过线不等于总体等价；CI包含0.5也不能把未过的点估计改成通过。例如Transformer13 AUC约.3448，区间约[.1926,.4487]，仍存在明确读出分数偏差。

主评价计算实际约{done['wall_seconds']:.2f}秒，新增网络Epoch/Optimizer Step/Probe拟合均0；报告整理时间另计。本轮是Real来源读出诊断，**不是Natural Sim的PRIMARY新成绩**。

## ② 与C11/C13逐实例对照

- C11-A：原Decoder对相同Real Validation的旧自重建，4/11通过，是最直接可比对照。
- C11-F：原GRU完整Natural迁移的历史Validation成绩，1/11。
- C13-Natural：幅值修复后Natural迁移的历史Validation成绩，1/11。
- 后两列是不同负类来源/样本数，不能把C23的6/11解释成Natural迁移由1提高到6。本轮只读历史指标CSV，没有读取它们的Natural数组或预测文件。

下表列正式AUC，“✓”表示该历史实例BA/AUC/D均通过；全部BA/D与通过字段在`historical_instance_comparison.csv`。

|实例|C11-A 旧Real重建|C23 修复Real重建|C11-F 旧Natural|C13 修复Natural|
|---|---:|---:|---:|---:|
'''
for x in comp:
 vals=[]
 for key in ['C11_A_old_Real_reconstruction','C23_repaired_Real_reconstruction','C11_F_old_Natural_transfer','C13_Natural_transfer']:
  vals.append(f"{float(x[key+'_AUC']):.6f} {'✓' if x[key+'_passed']=='True' else '×'}")
 text+='|'+x['instance']+'|'+'|'.join(vals)+'|\n'
text+=f'''
从4/11到6/11是净增2项，但不是单调改善：{', '.join(gain)}由未过变通过；{', '.join(lost)}由通过变未过；kNN保持通过。尤其Transformer13由约.4816变为.3448。不能只报平均改善或总数。

- 若只有3/11，会低于C11-A的4/11，不能说修复改善了整体读出；具体仍需看哪些实例变化。
- **当前6/11**表示部分读出伪影改善，但时序网络仍有残差；不能将剩余问题完全归到Natural latent/condition。
- 即使9/11，也只表示很接近的局部结果，剩余实例仍必须分析；不能替代11/11。
- 即使11/11，这项Real自重建诊断也不能冒充完整Sim-to-Real通过。

## ③ Decoder/读出链是否基本可用

**作为保留的实验基线，部分可用；作为“目标域读出问题已解决”的冻结终点，尚不可用。** 全部MLP、Logistic、kNN及GRU13通过，但GRU11/12与全部Transformer仍未过AUC。不能声称0.90完整闭环，也不能从此只研究latent映射。

### 输出层诊断

|项目|Real|C13 reconstructed Real|
|---|---:|---:|
'''
for key,title in [('zero_fraction','零比例'),('positive_mean','正值均值'),('positive_P1','正值P1'),('positive_P10','正值P10'),('positive_P50','正值P50'),('positive_P90','正值P90'),('positive_P95','正值P95'),('positive_P99','正值P99'),('maximum','最大值'),('transition_01','Mask 0→1'),('transition_10','Mask 1→0'),('mean_abs_frame_delta','相邻帧平均绝对变化'),('lag1_correlation','相邻帧相关系数'),('both_positive_mean_abs_frame_delta','双方帧均正值处平均绝对变化')]:
 text+=f'|{title}|{float(d0[key]):.7g}|{float(d1[key]):.7g}|\n'
text+=f'''
- 780/780输出finite、非负，780个数值窗口各不相同；形状和配对来源一致，没有裁剪、填正数或重排。
- 配对MSE={rec['MSE']:.8f}，Mask IoU={rec['Mask_IoU']:.8f}，复现C13保存结果。相对C11-A原MSE约.07787确有改善，但Mask未改变，空间错配仍在。
- 相邻帧平均绝对变化由约.000631变为.005529，约{float(d1['mean_abs_frame_delta'])/float(d0['mean_abs_frame_delta']):.2f}倍；双方帧均正值处也约{float(d1['both_positive_mean_abs_frame_delta'])/float(d0['both_positive_mean_abs_frame_delta']):.2f}倍。这里是相同窗口的帧索引统计，不宣称物理秒变化率。
- Mask转移本身略慢于Real，不是C18/C22那种Mask过快消失。此处更需关注**正值幅值的跨帧读出**，不能直接照搬先前Mask归因。
- 弱值结构也不同：Real正值中约0.87%≤1e-6，重建中为0；正值P90偏高、P99偏低。正值均值和零比例接近不代表完整条件分布一致。
- 逐帧MSE约.02271→.03054、IoU约.52593→.50016；只是趋势，不足以证明反馈误差是原因。按Session的MSE/IoU另存，不能只用一个平均数掩盖Session差异。

全部弱响应阈值仅用于诊断表，从未施加到C0或输出。绝对幅值不是已标定力阈值。

### 能否把失败归到某一个分支

本轮配对平方误差分解：

|部分|全MSE占比|解释|
|---|---:|---|
'''
labels={'both_positive':('双方均正值','相同Taxel/帧均激活处的数值误差'),'missed_positive':('漏掉Real正值','Mask漏检与目标幅值共同影响'),'generated_only_positive':('仅重建为正值','Mask误激活与生成幅值共同影响'),'both_zero':('双方均零','无误差')}
for x in dec:
 title,desc=labels[x['component']];text+=f"|{title}|{100*float(x['share_of_full_squared_error']):.2f}%|{desc}|\n"
text+='''
约82.68%的配对平方误差位于双方均激活的位置，且幅值时间变化明显偏大，所以**幅值/时间读出是当前最值得做最小受控修复的分支**。但这不是“82.68%的域差异由幅值头导致”：MSE分解不是C0因果归因，GRU/Transformer也可利用空间信息，Mask IoU约.512仍不能忽略。此次没有训练Mask-only Probe或构造Oracle混合输出来人为定位。

能够支持的判断：修复后的目标域Encoder/Decoder链仍有C0可见的重建伪影，幅值及时间统计提供了具体线索。不能证明Encoder完全无损、Decoder容量不足、Sim-to-Real整体失败，或某单一模块已经是唯一根因。

## ④ 下一步唯一推荐路线（仅提案，未执行）

**仍留在C13目标域读出链，做一次“冻结Mask的幅值跨帧一致性最小修复对照”，暂不进入新latent对应、GMM或Translator。**

理由：C23只有6/11；虽然比旧Real重建改善，但没有达到可以把问题完全收缩为Natural→正确Real latent的前提。当前有明确的正值误差和动态幅值偏差，优先级高于继续盲调Mask。

若另获确认，只允许沿用现有C13幅值分支：冻结Encoder、循环骨干、Mask、所有原C0及数据划分；保留原Gamma混合似然与自由运行条件，加入一项对齐真实相邻帧幅值差分的约束，不是把差分压到0。不要换网络、搜索分位数端点或修改输出阈值。训练权重和尺度规则须在开始前固定，不能根据C0反复调整；只用Real Train，Validation仅按预登记损失选一次Checkpoint，最后一次性报告全部11C0、幅值边际和动态结构。

这项建议不把轻微平滑当成功：必须检查样本多样性、输入响应、边际尾部及真实动态，防止变常数/过度平滑。只有11/11才能说该读出诊断通过，部分改善仍是中间态。预算建议单Seed、最多20轮或15分钟先到即停，不自动加Seed或加轮；**本轮没有编写/启动其训练**。

## ⑤ 唯一备选路线

若决定不再批准修复，或上述一次最小读出修复没有明确收益：**冻结当前0.90模型搜索，按已审计的论文收束路线，将触觉部分写成差异分析、统计伪配对与跨域重建探索，清楚保留未达标事实。** 不把6/11改成成功，不自动换GAN、Diffusion或扩大预算。

C23结果处于C类中间状态，不是A类近乎11/11，也不证明迁移原理不可行。该备选是期限/投入决策，不是GMM或触觉迁移理论被否定。

## 文件与执行保护

- `metrics.csv`：全部点估计、混淆矩阵、联合通过及共享Session Bootstrap区间。
- `historical_instance_comparison.csv`：C11 A/F、C13 Natural与C23全部BA/AUC/D及通过状态。
- `output_statistics.csv`、`reconstruction.json`、误差分解、弱响应、逐帧及逐SessionCSV：轻量输出诊断。
- `predictions/`及`score_statistics.csv`：原标签方向的完整分数；比较组与实际来源域分开保存。
- `registration.json`、`data_contract.json`、`bootstrap_group_audit.json`、`integrity.json`、`code/`及`run.log`：合同、实际访问范围与Hash。

代码只抽取原Probe/DeepProbe类、原行读取及Bootstrap定义，未导入GAN/VAE/Translator训练调度；使用原P37指标函数，冻结预处理逐项核对。没有运行新的Encoder/Decoder推理，也没有重新拟合分类器或统计Scaler。

所有受保护文件Hash未变；旧C0的Real参考召回与C13历史记录逐实例一致。本轮已停止，不自动执行提案。
'''
(O/'REPORT.md').write_text(text)
save('decision.json',dict(result_type='C_INTERMEDIATE',C23_readout_joint_passes=6,required_instances=11,readout_fully_established=False,usable_as_partial_research_baseline=True,primary_Sim_to_Real_new_evaluation='NOT_EVALUATED (no Natural input permitted)',passed_instances=passed,failed_instances=failed,gained_vs_C11_A=gain,lost_vs_C11_A=lost,most_actionable_branch='conditional positive amplitude and temporal readout, not a proved sole causal source',next_proposal='single fixed-configuration C13 amplitude temporal-consistency repair; frozen Mask; requires separate confirmation',next_experiment_executed=False,fallback='freeze model search and write honest difference/reconstruction exploration results',stop_after_C23=True))
# Standalone figures: all instances kept, and scopes explicitly labelled.
fig,ax=plt.subplots(figsize=(10,4.5));xs=np.arange(len(comp))
for key,label,offset in [('C11_A_old_Real_reconstruction','C11 A: old Real reconstruction',-.18),('C23_repaired_Real_reconstruction','C23: repaired Real reconstruction',0),('C13_Natural_transfer','Historical C13: Natural transfer',.18)]:ax.plot(xs+offset,[float(x[key+'_AUC']) for x in comp],'o',label=label,markersize=5)
ax.axhspan(.475,.525,color='green',alpha=.12,label='AUC gate');ax.axhline(.5,color='grey',linewidth=.7);ax.set_xticks(xs,[x['instance'] for x in comp],rotation=45,ha='right');ax.set(ylabel='AUC (official direction, no reversal)',title='Frozen C0: Real readout vs historical Natural transfer (different scopes)',ylim=(0,1));ax.legend(fontsize=8);fig.tight_layout();fig.savefig(O/'auc_comparison.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.5));keys=['positive_P1','positive_P10','positive_P50','positive_P90','positive_P95','positive_P99'];xs=np.arange(len(keys))
for j,s in enumerate(stat):axes[0].plot(xs,[float(s[k]) for k in keys],'o-',label=s['domain'])
axes[0].set_xticks(xs,['P1','P10','P50','P90','P95','P99']);axes[0].set(yscale='log',ylabel='Positive amplitude',title='Marginal quantiles (no output clipping)');axes[0].legend(fontsize=6)
axes[1].bar(['Real','Reconstruction'],[float(s['mean_abs_frame_delta']) for s in stat]);axes[1].set(ylabel='Mean absolute adjacent-frame change',title='Temporal amplitude diagnostic');fig.tight_layout();fig.savefig(O/'output_diagnostics.png',dpi=180);plt.close(fig)
protected=json.loads((O/'protected_hashes_before.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items());save('run_commands.json',dict(cwd='/home/zmy/dex_hand_project',python='/home/zmy/miniconda3/envs/mujoco/bin/python',environment=dict(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),scripts=['code/evaluate.py','code/report.py'],no_training=True,no_next_experiment=True))
files=[p for p in O.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'];save('artifact_manifest.json',{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
print('REPORT COMPLETE: 6/11, partial readout only; C23 stopped; no next experiment')
