# 第五章5.3 R1定点技术复核

状态：核验与定点修订完成，等待用户及ChatGPT审核。原V1不覆盖。本轮没有训练、拟合、推理、MuJoCo运行、新伪配对或Test/Holdout访问。

## 1. 关键判断

|核查点|结论|正文处理|
|---|---|---|
|训练偏移与逆构造固定坐标|存在原点语义不一致，代码没有省略在别处的目标中心化|保留实际q公式，明确局限；不虚构变换|
|分量中心|目标响应在原网格按责任度重新加权，不是μ加回目标质心|补明绝对索引加权中心的来源|
|混合均值|使用冻结π，不使用目标分量m|补明权重来源，准确公式见METRIC_DEFINITIONS.md|
|混合协方差|包含分量内协方差及分量间均值项|明确不同于单分量协方差或简单平均分量协方差|
|e的分母|没有相对误差分母；仅加权矩存在响应量归一化与ε下限|改用“差值范数”，不人为添加相对归一化|
|空帧|s/e仍平均全部4帧；d只平均双方活跃帧|补充差异与无有效d时拒绝登记|
|提前停止|统计gate一旦通过即停止，未执行候选不进集合；重复检查在最后|补明已执行集合及非全局最优；边界见指标报告|

## 2. 实际调用链与代码定位

根G=`outputs/tactile_gmm_dual_vae_2026-11-16/code/`；根C=`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/`。

1. G/fit_gmm.py第23–40行读取Real Train、逐Patch/阶段抽样并拟合；G/gmm.py第5–15行实现索引坐标、密度及EM。GMM参数随后复制进入集成P43和C3，三处文件Hash相同，见artifact_presence.json。不是本轮重新拟合。
2. C/dictionary.py第10–24行规定字典条件、调用physics.run并保存固定窗口。C/physics.py::reset、run规定复位与物理执行。
3. C/expand.py第41–60行计算责任度、目标分量、Top-5和字典检索。第14–16行的objective直接调用matching.inherited_metrics；第77行实际传入(y生成,x真实)。不是把Real、Sim顺序颠倒。
4. C/inverse_physics.py::run第43–79行从site得到实际位置/法线、施加球心命令、保存触觉与轨迹。只静态阅读，未导入/调用该脚本。
5. C/matching.py第4–9行定义s/d/e及加和代价；C/expand.py第64–87行规定候选、早停、argmin、去重和保存。pilot inverse.py第67–87行有对应逻辑。
6. C/verify_metrics.py把新函数与旧六例指标比较，已有验证登记不等于本轮重新执行。本轮只核对其调用与字段，不运行历史验证脚本。

## 3. 九项其他事实逐项核对

### 3.1 480行组合

C/dictionary.py第10–15行：16Patch×中心及上下左右5位置×0.25/0.5/1mm三深度×posture0/1两设置=480。球半径固定4mm、tilt_deg=0。实际dictionary_manifest.csv重计480行，不能说有480种独立机械手初态。

### 3.2 160指纹定义

C/dictionary.py第19行：SHA256(round(qpos[0],9).tobytes() || round(mocap_position[0],9).tobytes())，不是只Hash任务ID或模型文件。qpos[0]是首个已保存控制步推进后的完整qpos，包含手和自由球等模型状态；mocap_position[0]是此步目标球心数组，不是actual_probe_position。

C/physics.py第63–69行先放置球并推进一个控制步，再保存qpos/mocap。因此“放置后初态”是登记名称，不是放置前reset状态，也不是严格t=0。它不包含全部qvel、控制量或模型参数，不能认证独立完整物理状态。Manifest的160个不同指纹各对应3行；三种深度的初始未加载放置可共享指纹。指纹的9位小数量化和字节拼接是该历史版本定义。

### 3.3 驱动指令0

C/physics.py第23–28行：mj_resetData后`d.ctrl[:]=.0006*posture`，推进200个物理步再重置信号处理器。posture=0表示所有模型控制槽为0目标，posture=1为0.0006目标，不是直接把所有关节qpos设为0，也不能据此声称没有被动弹簧、外力或物理接触。伪配对检索只取posture=0，字典中另一设置未参与该检索路径。

### 3.4 离散字典深度与连续反馈

C/expand.py第55–60行：位置检索只在posture0、深度0.25且target_peak>0的条目中进行；选定位置后用同位置posture0三条深度响应量作`np.interp(co.mass,masses,depths)`，所以初始深度可为两档之间的连续值，并非只能三选一。代码没有额外按masses排序或检验单调性的步骤，不能保证所有曲线为已认证单调标定。插值越界沿NumPy端点行为取端点。

第70行的后续反馈倍率为`clip((RealPatchTotal/max(SimPatchTotal,1e-10))**(1/.7),.5,2)`，乘初始spec的深度并clip到[.1,1]mm。它不受“必须属于原三档”的限制；第66行每次从初始specs复制，不能写成深度在连续试次中直接累乘。命令量不是实际材料压陷真值。

### 3.5 四帧加载因子

C/expand.py第60行，令P_t=Σ_j x_tpj：f_t=clip(P_t/max(mean(P_0..P_3),1e−12),0,2)。不是按最大值归一化，也不是min-max映射或四帧总和归一化。

C/inverse_physics.py第51–62行在实际深度命令上乘因子；取因子索引clip(frame−12,0,3)，在12–15帧对应四个值，窗口外沿用首/末因子，并叠加原进入—保持—退出加载包络。没有把四帧幅值直接写入触觉数组，也不能称两域物理时间同步。

### 3.6 分阶段与ALL

G/gmm.py第4行定义ALL/CLOSE/HOLD/LIFT。G/fit_gmm.py第26–40行各自拟合、样本不足用同Patch ALL，保存4套参数；早期GMMRenderer.transform第23–29行可按输入Stage选择并以anchor平移核。C/expand.py第44行和matching.py第6行固定取索引0的ALL，C3逆构造/评分没有用Stage切换GMM。早期渲染分支不能拿来证明C3已完成目标中心化。

### 3.7 失败候选是否实际保存

C/inverse_physics.py第77–79行在返回评分前保存episode.npz、condition.json、contact_records.json；C/expand.py第77–79行把每次尝试的路径与评分写入expansion_attempts.jsonl，不只保留最终最优者。pilot写inverse_attempts.csv。

本轮只读取这些Train/Validation元数据、检查路径存在并对一个失败实例计算文件Hash，**没有解码Episode数组或重算触觉指标**。这两份日志共3593条记录（pilot124、扩展3469）中，2153条统计gate未通过记录的上述三类文件均存在；详见artifact_presence.json。该日志范围不冒充全部历史物理执行：C4历史汇总另有3597次口径，本轮没有逐条对齐其多出的4条记录，差异来源UNKNOWN，未覆盖或更改历史汇总。

未构造任何候选时NO_NONZERO_DICTIONARY_SUPPORT不代表有物理Episode应存在。这里只认证文件存在与一个样本Hash，不声称全部失败文件都已重新回放或完成物理正确性认证。

### 3.8 GMM拟合角色

G/fit_gmm.py第23行只从P37/train.npz读取Real进行参数拟合。第41–44行后续虽读取Sim Train/Validation作早期统计渲染输出，但位于参数保存之后，不参与GMM拟合；不能把“脚本读过Validation”混同“用Validation拟合先验”。C3逆构造读取Real Validation目标只是按冻结先验构造对应验证样本，不重新拟合GMM。

### 3.9 Paired1200/240范围

C/expand.py第33–37行先接入pilot已接受窗口，再扩展到1200Train/240Validation；因此总数**包含pilot，不应再加pilot32/10**。第89–92行最终写paired_train/validation和expansion_complete.json；本轮核对该登记中的1200/240，沿用上一轮对指定4×712数组的L2核验，不重新构造数据。它们是有效伪配对窗口，不是所有尝试数，也不是独立Root数。

## 4. UNKNOWN与未执行

- 坐标不一致对配对、加载饱和及迁移指标的独立影响：UNKNOWN；未运行对照。
- 以已执行集合取最小的边界情况实际造成多少拒收：UNKNOWN；没有新评分或改选择。
- Real硬件到Sim物理Taxel映射：未认证；不以归一化数值坐标替代。
- 全体失败Episode可回放性：本轮未复验，仅查路径与一例Hash。

## 5. 正文改动范围

只明确训练/责任度原点问题、绝对索引加权中心、混合π及完整协方差、s/d/e空帧与分母、初态指纹和零指令含义、提前停止与已执行集合最优性。三个核心公式主体保持忠实于原代码，不增加不存在的中心化、e分母或额外门槛。完整精确定义放在专项报告，不把方法节扩写为审计记录。

修订稿单独写入本revision_r1；原section_5_3.tex及原阅读稿均保留。最终字数、差异和保护Hash另见交付文件。不编译、不冻结、不开始5.4、不commit/push。
