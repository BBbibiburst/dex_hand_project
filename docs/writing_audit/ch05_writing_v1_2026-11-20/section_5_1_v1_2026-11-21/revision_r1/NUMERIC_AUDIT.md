# 数字专项核验

范围：只读取P37旧开发数据train.npz及validation.npz的real/sim键；无模型/Probe推理、无训练/仿真，无Test/Holdout数组。本轮是既有数组的轻量静态计数/分位数核验，不产生新的实验或转换结果。复核脚本check_numeric.py、完整值numeric_check.json、来源Hash在numeric_source_hashes.json。

## 1. 逐项数字、形状与分子/分母

所有数组dtype为float32；统计不按Session/Root宏平均，不包含生成或校准后的响应。旧CSV候选名raw在下面仅作为查找键，正文不继承该别名。

|正文数字|原始文件/key与shape|角色/版本|原始统计字段与算法|分子/分母或样本池|一致性|
|---|---|---|---|---|---|
|激活率17.60%|`outputs/tactile_p37_full_data_revalidation_2026-11-15/train.npz` / real，5153×4×712|P37 Train|mask_statistics.activation_fraction；mean(X>0)|2582877 / 14675744 = N×4×712分母|与CSV及正文舍入值一致|
|激活率13.31%|`outputs/tactile_p37_full_data_revalidation_2026-11-15/train.npz` / sim，1824×4×712|P37 Train|mask_statistics.activation_fraction；mean(X>0)|691261 / 5194752 = N×4×712分母|与CSV及正文舍入值一致|
|P50=0.0050025014206767082；P95=1.5010055303573537|`outputs/tactile_p37_full_data_revalidation_2026-11-15/train.npz` / real，5153×4×712|P37 Train|amplitude_statistics.positive_P50/P95；np.quantile(X[X>0],[.5,.95])|严格正值池2582877项；分位数非比率，无分子分母|一致|
|P50=1.039736616803566e-05；P95=0.45948466658592224|`outputs/tactile_p37_full_data_revalidation_2026-11-15/train.npz` / sim，1824×4×712|P37 Train|amplitude_statistics.positive_P50/P95；np.quantile(X[X>0],[.5,.95])|严格正值池691261项；分位数非比率，无分子分母|一致|
|Mask变化率1.7154%|`outputs/tactile_p37_full_data_revalidation_2026-11-15/validation.npz` / real，1408×4×712|P37 Validation|temporal_statistics.change_fraction；mean((X[:,0]>0)!=(X[:,1]>0))|17197 / 1002496，分母N×712|一致|
|Mask变化率0.0958%|`outputs/tactile_p37_full_data_revalidation_2026-11-15/validation.npz` / sim，406×4×712|P37 Validation|temporal_statistics.change_fraction；mean((X[:,0]>0)!=(X[:,1]>0))|277 / 289072，分母N×712|一致|

CSV根为`outputs/tactile_p39_domain_decomposition_2026-11-15/`，激活率和幅值取stage=ALL、patch=ALL、role=train、candidate=Real/raw；变化率取stage=ALL、role=validation、transition=0、candidate=Real/raw。计算代码describe.py:13–28直接读取P37对应role的real/sim。

激活率分母不是只有激活位置的数量，而是各域N×4×712。正值统计先筛X>0后汇总全部窗口、帧和通道，不按Patch求分位数再平均，不在log1p空间统计P50/P95，也不除以最大值。np.quantile采用代码默认插值；正文所列精度只为显示舍入，没有改原数值。

变化率仅零基0→1，即第1→2个保留帧，分子计所有发生0↔1变化的通道位置。不是4帧三次变化的平均，也不是01/10的条件概率。01和10在历史CSV同样以N×712作分母。没有除真实时间间隔。

相邻幅值分析仍对应describe.py:28的mean(abs(X[:,1]-X[:,0]))（全通道，含零值，非正值池均值），历史Validation记录Real=0.0005749664269387722、未转换Sim=0.0004527934070210904。R1仅说明应观察此对象，没有新增幅值变化数值或物理频率比较。

## 2. 时间戳证据

`outputs/tactile_p42_dual_vae_diagnostics_2026-11-16/window_time_contract.csv`仅包含Train/Validation；mask_diagnostics.py:48–53说明从P37/data_inventory.csv既存timestamps按已选窗口ID恢复，再计算dt_min/max/mean。该表是时间戳派生记录，不是依据模型输出估计频率。

本轮核对其角色与窗口数量，并检查每窗dt_min、dt_max；Real Train5153窗、Real Validation1408窗均存在大于1e−6秒的窗内间隔差。例：Train第一窗dt_min=0.06001615524291992，dt_max=0.07886576652526855，dt_mean=0.06861392656962077。完整样例、来源窗口ID与合同见numeric_check.json。该例已足以反对统一固定间隔假设，全部记录也支持非均匀表述。

本轮未重新访问实机日志或重采样，证据等级为既有时间戳派生CSV及提取代码直接核对。其原始记录时间戳来源仍以历史采集合同为限，不证明硬件时钟跨设备校准或两域物理同步。不存在本轮根据差分推断真实频率的结论。

## 3. 三个D值及Probe语义

原V1已核最小值保持不变：mask_after=0.02718456112852663，amplitude_after=0.14935764666368123，full_after=0.05725551388266892，正文显示0.0272/0.1494/0.0573。取P39/probe_metrics.csv中role=validation对应表示的11行最小D，非只挑某个Seed来声称达标。

11个实例=Logistic1+kNN1+MLP3+GRU3+Transformer3（各Seed11/12/13）。Full复用P38同校准后输出、同划分/预处理的11个已有独立Probe拟合；不是把迁移前冻结C0当后训Probe，也不是新训练11个模型。Mask校准前后逐值相同，mask_after是已有mask_before结果的别名，并非另一次独立实验。代码run.py:41–49、72–80，历史报告与P38复现记录说明此复用关系。

正文补D=1−2|BA−0.5|，它只反映BA接近0.5，不能替代AUC或完整分布质量。有效Amplitude依旧4×5；Train各自拟合均值/标准差（标准差下限0.05），Full与分位数先log1p，Mask由精确x>0构造，定义不变。

## 4. 数据边界

P37旧开发Real Train/Validation=5153/1408，Sim=1824/406。P43主线合同Real=5153/780、Natural=4049/905。R1明确P37统计不能自动外推P43，后者数量只描述合同，不冒充本轮重新分析后的P43差异结果。未读取P43原始数组。

未出现无法确认的所请求数字；需要保持UNKNOWN的是认证物理Taxel对应、共同力学尺度和跨设备物理同步，而不是以这些缺口抹去可验证的数值统计。
