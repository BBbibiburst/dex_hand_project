# C25：Real-Sim潜空间与Conditional GMM-OT可行性审计

**结论：NO-GO——停止在当前独立Encoder潜空间中直接构造GMM-Wasserstein/OT对应。** 这不是触觉迁移或OT方法被证明不可行，而是当前代价坐标系未被认证。本轮不训练网络、不运行C0、不生成712D迁移数据、不读取Test/Holdout；完成后停止，没有进入C26。

## 1. 数据与实际执行

| dataset            |   windows |   groups |   trials_or_episode_units | group_type   |   unique_numeric_windows | latent_shape   | cached_encoder_checked   | common_coordinate   |
|:-------------------|----------:|---------:|--------------------------:|:-------------|-------------------------:|:---------------|:-------------------------|:--------------------|
| Real_train         |      5153 |       32 |                       755 | Session      |                     5153 | [5153, 4, 64]  | True                     | False               |
| Natural_train      |      4049 |       50 |                        50 | Root         |                     4049 | [4049, 4, 64]  | True                     | False               |
| Real_validation    |       780 |        4 |                       120 | Session      |                      780 | [780, 4, 64]   | True                     | False               |
| Natural_validation |       905 |       10 |                        10 | Root         |                      905 | [905, 4, 64]   | True                     | False               |

Real Train为32Session/755Trial，Validation为4Session/120Trial；Natural为五任务Train50Root/50源Episode、Validation10Root/10源Episode。4049/905是当前Natural采集去重窗口，不是旧P37的1824/406，也不是服务器新增小时数。旧GMM/Dual-VAE和P40的数据来源另见模型清单，未把旧Sim与Natural混合。

仅复用经过冻结Encoder首批64窗CPU数值重放核对的C4后验均值/logvar，逐角色核验全部groups、stage、units一致。保存当前Train/Validation副本和来源Hash。只读模型：C13 Real、C3 Sim及历史模型合同；未执行Decoder或Translator。

## 2. Stage A：公共坐标系未成立

C13 Encoder与C3原Real Encoder逐位相同；幅值修复没有改变编码器。Real和Sim是两个独立GRU VAE，均为4×64，但使用不同权重和各自训练scale。输入为exact-positive Mask拼接log1p(x)/encoder.scale；输出posterior mean与clamp[-8,4]的logvar。其训练没有Real/Sim共同参数或潜空间联合对齐损失，Translator是在两者冻结后以统计伪配对训练。

即使两个VAE的先验都是标准正态，也不能据此认定第j维有相同语义。潜变量旋转/置换与相应Decoder补偿可保持自身重建，却改变跨域欧氏或Gaussian W2代价。直接把两个256维向量放在一起会把这种任意坐标选择当成接触对应。

因此本轮没有计算native Real–Sim距离、交集率、联合PCA或跨域W2，也没有拟合GMM来规避这个前提。各自标准化/白化不能建立共同语义；未经配对/共同函数验证的标准化仍然不够。

### 历史上存在的其他模型不能被忽略

- 旧GMM+Joint Dual-VAE：Real/Sim为独立ER/ES，4×32，经过MMD/CORAL及共享代理头联合约束；它是名义共享表征候选，并非“从未对齐过”。但没有认证Natural数据对应，且不能直接接收/输出C13的4×64坐标。
- P40：确实用同一个共享Encoder处理两域，4×64是同一函数坐标。但训练Sim为P37转换数据，不是当前五任务Natural；其feature Z不是C13 VAE后验，也没有与C13 Decoder的接口映射。不能因形状相同而直接接入。此次未重新跑P40或训练新网络。
- 历史空间GMM：是固定Patch索引坐标上的响应先验，不是神经Encoder；物理Taxel行列映射未认证。
- C3 GRU、C10、C10R1：是映射候选，复用C3独立Encoder；不能用待验证的映射本身证明坐标已对齐。分层Mask的global8/patch8随机变量也不是这套4×64 VAE潜变量。

完整模型合同和Hash见encoder_inventory.csv、historical_mapper_hashes.json。

## 3. Stage B：各域内部结构与稳定性

| dataset            |   windows |   effective_rank |   top10_variance |   PCs_for90pct |   norm_P10 |   norm_P50 |   norm_P90 |   norm_max |   total_variance |
|:-------------------|----------:|-----------------:|-----------------:|---------------:|-----------:|-----------:|-----------:|-----------:|-----------------:|
| Real_train         |      5153 |          6.24395 |         0.975277 |              6 |    20.6045 |    24.0958 |    30.143  |    33.8499 |          435.852 |
| Natural_train      |      4049 |          4.61554 |         0.979907 |              5 |    21.0439 |    24.8829 |    32.063  |    42.6247 |          371.062 |
| Real_validation    |       780 |          3.41461 |         0.983525 |              3 |    19.1305 |    25.7976 |    30.1498 |    32.3145 |          202.571 |
| Natural_validation |       905 |          4.2559  |         0.97953  |              4 |    20.3238 |    23.9623 |    30.6159 |    40.3218 |          312.047 |

结果表明两边自身都呈低有效秩结构，但相近有效秩、范数或PCA曲线不能证明它们共享同一个低维流形。PCA图分面、分别使用各自Train拟合的基底，不共享坐标轴语义。256维逐维均值/标准差、协方差、特征谱和64维每帧logvar已保存。

| domain   | metric                  |   Train_cross_group_NN_P95 |   validation_fraction_within_Train_cross_group_P95 | interpretation                                                                           |
|:---------|:------------------------|---------------------------:|---------------------------------------------------:|:-----------------------------------------------------------------------------------------|
| Real     | raw_native_Euclidean    |                 14.1204    |                                           1        | within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee |
| Real     | Train_only_standardized |                 11.1836    |                                           1        | within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee |
| Real     | Train_only_whitened     |                 13.3542    |                                           0.926923 | within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee |
| Real     | native_cosine           |                  0.14168   |                                           1        | within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee |
| Natural  | raw_native_Euclidean    |                  7.91281   |                                           0.941436 | within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee |
| Natural  | Train_only_standardized |                  7.29322   |                                           0.954696 | within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee |
| Natural  | Train_only_whitened     |                 11.1766    |                                           0.883978 | within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee |
| Natural  | native_cosine           |                  0.0610906 |                                           0.949171 | within-domain descriptive reference envelope, NOT Real-Sim support or coverage guarantee |

上表“域内参考包络”是各域Train跨Session/Root最近邻距离P95，统计Validation有多少落在该阈值内；阈值只用相应Train拟合。它不是Real–Sim覆盖率，也不证明源/目标语义可转移。排除自身与排除同组两种Train参考均保存；高比例可能受训练组异质性影响。

| role       |   independent_Roots |   task_categories |   task_explained_Root_centroid_variance |   label_permutation_mean |   label_permutation_P95 | descriptive                                                                                          |
|:-----------|--------------------:|------------------:|----------------------------------------:|-------------------------:|------------------------:|:-----------------------------------------------------------------------------------------------------|
| train      |                  50 |                 5 |                                0.715168 |                0.0785864 |                0.139061 | native Sim coordinates, group-level association; task not randomized, not causal/Real correspondence |
| validation |                  10 |                 5 |                                0.923859 |                0.443329  |                0.800412 | native Sim coordinates, group-level association; task not randomized, not causal/Real correspondence |

五任务的原生Sim latent可按任务/阶段分层，Root均值的任务关联统计见上表；任务并非随机分派，阶段构成/物体/控制方式也不同，因此不是因果任务效应或Real对应证据。Real记录未具有同一五任务标签体系，不能完成跨域任务匹配。所有逐Root、Session、stage、task结果见latent_group_stage_task.csv和condition_group_stage_task.csv。

## 4. Stage D：共同condition能支持到什么程度

| condition                                     | formula                                                        | shared_observable_semantics                                   | limitations                                                                                                        | decision                                      | OT_now   |
|:----------------------------------------------|:---------------------------------------------------------------|:--------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|:---------|
| total_response                                | mean_t sum_j x[t,j]                                            | sum of processed sensor response                              | units/gain not cross-calibrated; cannot call force or energy                                                       | REJECT_PHYSICAL_OT_COST                       | False    |
| positive_log_mean                             | sum log1p(x)/count(x>0)                                        | nonzero processed response intensity                          | depends on domain units, weak positives and processing                                                             | REJECT_UNCALIBRATED_COST                      | False    |
| taxel_participation                           | mean_t (sum x)^2/(712 sum x^2); zero-frame=0                   | scale/permutation-invariant response concentration            | processing/background alters concentration; not contact area; no certified semantic alignment                      | WEAK_STATISTICAL_PROXY_ONLY                   | False    |
| patch_response_entropy/effective_patch_number | p_patch=sum_patch x/sum_all x; H(p),1/sum(p^2)                 | response concentration among fixed16 blocks                   | blockwise aggregation certified in array only; physical identities/orientations unverified; not number of contacts | EXPLORATORY_WEAK_PROXY                        | False    |
| scale_free_frame_delta                        | mean_t sum|x[t]-x[t-1]|/(sum x[t]+sum x[t-1]); 0 denominator=0 | scale/permutation-invariant retained-frame relative variation | Real uneven intervals and Natural20Hz; neither derivative nor matched motion stage                                 | EXPLORATORY_WEAK_PROXY_REQUIRES_TIME_CONTRACT | False    |
| exact_active_patch/transition01/transition10  | x>0 existence and conditional positive transitions             | exact-zero pattern of processed values                        | threshold-sensitive, tails/background/filters differ; never infer physical contact count                           | REJECT_PHYSICAL_CONTACT_CONDITION             | False    |
| patch_index_centroid                          | weighted mean of fixed taxel index coordinates                 | index-space center only                                       | Real/Sim row-column physical map not certified                                                                     | REJECT_COMMON_PHYSICAL_POSITION               | False    |
| task/stage                                    | metadata literal values                                        | recorded labels                                               | Real task not5task classification; same name stage not equal physical state                                        | STRATIFY_ONLY_NOT_MATCH                       | False    |

存在弱统计共同量，不应把它们说成完全没有语义：taxel participation是对712个响应通道的集中程度，整体正增益与通道置换不改变它；relative frame delta也是相同定义的相对响应变化。这些性质经数值测试核验。但它们不是共同的力、接触面积、物体位置或机器人状态。时间跨度、噪声、门控和滤波的差异仍然改变数值分布。

Patch energy shares/entropy只在固定数组分块上成立；物理指节对应不能默认认证。Real缺原驱动源码、处理开关运行效应未知，Sim存在串扰/时间滤波，因此exact positive不能当作实际接触。总响应的单位与增益也未跨域标定。

| dataset            | threshold_type      |   threshold |   mean_active_patch |   equal_group_mean | no_output_threshold_change   |
|:-------------------|:--------------------|------------:|--------------------:|-------------------:|:-----------------------------|
| Real_train         | relative_frame_peak |       0     |            15.4624  |           15.3978  | True                         |
| Real_train         | relative_frame_peak |       0.001 |            12.5139  |           11.409   | True                         |
| Real_train         | relative_frame_peak |       0.01  |             7.77009 |            7.35065 | True                         |
| Real_train         | relative_frame_peak |       0.05  |             5.79289 |            5.55907 | True                         |
| Natural_train      | relative_frame_peak |       0     |             7.83817 |            7.21367 | True                         |
| Natural_train      | relative_frame_peak |       0.001 |             4.55273 |            4.35271 | True                         |
| Natural_train      | relative_frame_peak |       0.01  |             4.39201 |            4.20896 | True                         |
| Natural_train      | relative_frame_peak |       0.05  |             4.23771 |            4.06836 | True                         |
| Real_validation    | relative_frame_peak |       0     |            15.126   |           15.1331  | True                         |
| Real_validation    | relative_frame_peak |       0.001 |            12.817   |           13.066   | True                         |
| Real_validation    | relative_frame_peak |       0.01  |             7.3516  |            7.7038  | True                         |
| Real_validation    | relative_frame_peak |       0.05  |             5.58429 |            5.6438  | True                         |
| Natural_validation | relative_frame_peak |       0     |             6.64503 |            5.98183 | True                         |
| Natural_validation | relative_frame_peak |       0.001 |             3.8768  |            3.7058  | True                         |
| Natural_validation | relative_frame_peak |       0.01  |             3.68481 |            3.53415 | True                         |
| Natural_validation | relative_frame_peak |       0.05  |             3.53978 |            3.41171 | True                         |

Real Train精确正值Patch均值15.46，在5%帧峰值诊断阈值下为5.79；Natural对应7.84→4.24。这说明不能拿15.46直接当物理接触数量或OT目标。阈值只用于本表，未改任何输出或C0。

| dataset            | features                                                          |   fraction_in_all_Real_Train_marginal_P05_P95 | interpretation                                                                                 |
|:-------------------|:------------------------------------------------------------------|----------------------------------------------:|:-----------------------------------------------------------------------------------------------|
| Real_train         | taxel_participation|effective_patch_number|scale_free_frame_delta |                                      0.770813 | descriptive axis-aligned weak-feature overlap; not common latent/physical correspondence proof |
| Natural_train      | taxel_participation|effective_patch_number|scale_free_frame_delta |                                      0.509015 | descriptive axis-aligned weak-feature overlap; not common latent/physical correspondence proof |
| Real_validation    | taxel_participation|effective_patch_number|scale_free_frame_delta |                                      0.828205 | descriptive axis-aligned weak-feature overlap; not common latent/physical correspondence proof |
| Natural_validation | taxel_participation|effective_patch_number|scale_free_frame_delta |                                      0.435359 | descriptive axis-aligned weak-feature overlap; not common latent/physical correspondence proof |

弱描述子边际区间存在部分交叠，但这个表只回答可观测统计是否有共同数值区域，既不证明条件等价，也不等于潜空间交集或迁移成功。没有为提高交叠筛样本。每域分组均值、300次完整Session/Root等权Bootstrap区间、逐任务/阶段统计均保留；没有用相邻窗口数作为独立单位。

## 5. Stage C/E/F：按停止条件未执行

|阶段|状态|原因|
|---|---|---|
|GMM K=2/4/8/16|未拟合|公共代价坐标未成立；不能报告AIC/BIC或稳定性|
|Gaussian W2、条件OT|未求解|跨独立原生坐标的代价缺依据；弱condition不能自动修复|
|OT条件打乱/epsilon/K稳定性|未运行|没有合法可解释的基础plan|
|四种随机/硬匹配对照|未运行|没有合格GMM/OT；随机性优势UNKNOWN|

未运行不是GMM数值失败，也不证明随机映射无效。当前无法证明OT概率采样优于直接目标GMM抽样；不能把后者的Real-like随机生成当作输入响应保持。

## 6. 八项最终回答

1. **可信共同latent坐标？** 默认C13 Real/C3 Sim没有。P40有共同函数特征，但不是C13兼容坐标；旧联合32维也不是当前接口。
2. **共同支持？** 跨native空间UNKNOWN，未进行无依据的距离计算；各域内部有低秩且部分Validation落在Train参考包络内。
3. **GMM稳定？** UNKNOWN，按前提停止，未拟合。低秩提示即使未来坐标成立，也应使用Train确定的降维/正则化，不能直接用不受约束256维全协方差宣称稳定。
4. **共同condition？** 有定义一致、尺度/置换不变的弱响应统计候选；没有认证为共同物理接触状态，不能使用未验证质心或exact positive数量。
5. **OT多对多与稳定性？** UNKNOWN，未求解。
6. **随机对应有优势？** UNKNOWN，未模拟；不能凭多样性或finite就认定保留源输入。
7. **值得继续当前Conditional GMM-OT？** 当前直接native→native方案NO-GO，不批准直接进入C26。
8. **具体缺失？** 首要缺失是与目标C13接口兼容、并保留源输入信息的共同代价坐标/对应依据。弱条件只能辅助，不能定义两个独立VAE轴之间的关系。

## 7. 唯一下一步建议与边界

**如要重新评估这条路线，先只验证一个固定公共代价表示，而不是拟合新的Transport。** 可把“同一个冻结C13 Real Encoder用于两域”作为候选假设：它在数学上定义共同轴，但Natural是否落在有效区域、是否保留输入响应与空间/时间结构仍须核验，不能直接假定可用。该建议不要求相同物理状态的精确配对，也不承诺能够通过C0。

另一种历史共享Encoder不能仅靠4×64同形接入C13 Decoder；那需要额外桥接，会超出本轮及最小下一步。此次不训练新Encoder、不重新修Decoder、不替换C0，也不生成正式迁移数据。没有共同表示证据之前不继续GMM-OT。C25完成后停止，等待人工决定。

## 文件与复现

code/audit.py与finish.py为本轮代码；registration.json固定允许范围。native_latents/covariance、latent_summary/dimensions/eigenspectrum、逐组分层统计、域内距离、条件敏感性与Bootstrap、encoder_inventory及全部Hash均保留。run.log记录实际执行，integrity.json核验原权重/数据/C0文件未变；没有读取Test/Holdout文件或任何新C0分数。
