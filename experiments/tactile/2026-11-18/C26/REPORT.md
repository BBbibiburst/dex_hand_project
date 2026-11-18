# C26：共享冻结Encoder与零样本转换可行性

**结论：CONDITIONAL GO仅限输入分布外问题分析；当前零样本转换NO-GO，不允许直接进入GMM-OT或C27。** 同一个C13 Encoder成功建立了共同数值坐标，但部分支持区域失配，Decoder输出的尾部和时间结构明显异常。Stage D未通过，因此原11个C0按预设顺序未运行，不写成0/11，也不引用旧分数冒充C26。

## 1. Stage 0：输入合同与物理映射

| dataset            | shape          | dtype   | finite   | nonnegative   |   zero_fraction |   minimum |   maximum |   positive_P01 |   positive_P50 |   positive_P99 |   windows |   groups | group_unit   |   exact_unique_numeric_windows |
|:-------------------|:---------------|:--------|:---------|:--------------|----------------:|----------:|----------:|---------------:|---------------:|---------------:|----------:|---------:|:-------------|-------------------------------:|
| Real_train         | [5153, 4, 712] | float32 | True     | True          |       0.8240037 |         0 |  12.26911 |   6.938894e-18 |   0.005002501  |      4.10888   |      5153 |       32 | Session      |                           5153 |
| Natural_train      | [4049, 4, 712] | float32 | True     | True          |       0.8407124 |         0 |   1       |   5.605194e-45 |   3.783447e-11 |      0.8824956 |      4049 |       50 | Root         |                           4049 |
| Real_validation    | [780, 4, 712]  | float32 | True     | True          |       0.8342868 |         0 |  10.49105 |   2.501251e-05 |   0.005852927  |      4.570212  |       780 |        4 | Session      |                            780 |
| Natural_validation | [905, 4, 712]  | float32 | True     | True          |       0.8605566 |         0 |   1       |   4.203895e-45 |   3.701769e-12 |      0.8808682 |       905 |       10 | Root         |                            905 |

所有Train/Validation数组为非负finite float32 [N,4,712]；16块大小一致，现有Real canonical置换可追溯，未作任何新置换、反转、归一化、补值或阈值改动。四帧同一Trial/Root且时间递增。Real按历史greedy20Hz保留帧、间隔不均匀（见C24逐帧audit），Natural为连续20Hz源帧。

物理合同仍为UNKNOWN：原驱动/Calibration代码缺失；P41中10块320Taxel的中段/根段疑点是“若历史用户解剖表正确才构成冲突”，并非已证明数据放错。块内旋转/行列方向和实机物理单位未认证。Real/Sim processed也不是已证明相同的传递函数，Sim力经deadzone/power/lowpass/crosstalk不等于实机压阻标定。

这里区分“已证实的阻止执行的数值合同错误”和“物理对应未知”。本轮未发现前者，因此按用户Stage D对物理未知的分支，仅进行保持原索引的统计零样本实验；绝不把同索引Mask IoU解释成真实接触位置保持。若要求认证物理触觉转换，Stage0仍未满足认证条件。

## 2. Stage A：同一模型、同一预处理、同一后验路径

两域都使用C13同一个冻结Real Encoder，CPU batch64，取posterior mean并保留logvar。精确正值Mask拼接log1p(x)/原encoder.scale，不重新拟合输入Scaler，不使用Sim Encoder，不做域内独立归一化。全部latent、原索引、stage、Root/Session和task保存。Real latent与已有缓存核验一致。

| dataset            |   windows |   effective_rank |   PCs_90pct |   top10_explained_variance |   total_variance |   norm_P50 |   norm_P95 |   norm_max |   mean_coordinate_std |   posterior_variance_P50 |   posterior_variance_P95 | same_Encoder   |
|:-------------------|----------:|-----------------:|------------:|---------------------------:|-----------------:|-----------:|-----------:|-----------:|----------------------:|-------------------------:|-------------------------:|:---------------|
| Real_train         |      5153 |         6.243945 |           6 |                  0.9752773 |         435.8521 |   24.09579 |   30.96587 |   33.84986 |             1.229154  |               0.02070736 |               0.08069418 | True           |
| Natural_train      |      4049 |         4.957295 |           6 |                  0.9629637 |         306.3464 |   22.51001 |   25.3941  |   28.96709 |             1.056465  |               0.05900207 |               1.162226   | True           |
| Real_validation    |       780 |         3.414607 |           3 |                  0.9835253 |         202.571  |   25.79755 |   31.0836  |   32.3145  |             0.8197126 |               0.01261282 |               0.04808622 | True           |
| Natural_validation |       905 |         4.652928 |           5 |                  0.9661548 |         301.5008 |   22.30772 |   25.33848 |   28.91106 |             1.03868   |               0.07195945 |               2.48975    | True           |

Natural没有发生整体latent collapse：Train/Validation有效秩约4.96/4.65，均有显著样本差异。但这不证明接触语义被保留；相比Real Validation，Natural后验方差分布也明显偏大，记录在latent_summary.csv。逐维、协方差、逐组及逐任务结果均保存。

## 3. Stage B：数值支持区域仅部分重叠

| metric                 | dataset            |   windows |   within_RealTrain_crossSession_P95 |   equal_group_fraction |   unique_target_nearest_Real_Sessions | interpretation                                                                        |   stratum |   value |
|:-----------------------|:-------------------|----------:|------------------------------------:|-----------------------:|--------------------------------------:|:--------------------------------------------------------------------------------------|----------:|--------:|
| raw_Euclidean          | Real_validation    |       780 |                           1         |              1         |                                    17 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| raw_Euclidean          | Natural_train      |      4049 |                           0.8409484 |              0.8367384 |                                    32 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| raw_Euclidean          | Natural_validation |       905 |                           0.7447514 |              0.8000639 |                                    25 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| RealTrain_standardized | Real_validation    |       780 |                           1         |              1         |                                    18 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| RealTrain_standardized | Natural_train      |      4049 |                           0.7819215 |              0.7729323 |                                    30 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| RealTrain_standardized | Natural_validation |       905 |                           0.6939227 |              0.7532929 |                                    24 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| RealTrain_whitened     | Real_validation    |       780 |                           0.9269231 |              0.913125  |                                    21 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| RealTrain_whitened     | Natural_train      |      4049 |                           0.2729069 |              0.2587804 |                                    31 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| RealTrain_whitened     | Natural_validation |       905 |                           0.2629834 |              0.2431982 |                                    29 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| cosine                 | Real_validation    |       780 |                           1         |              1         |                                    16 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| cosine                 | Natural_train      |      4049 |                           0.7399358 |              0.7289897 |                                    32 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |
| cosine                 | Natural_validation |       905 |                           0.679558  |              0.7202519 |                                    24 | numeric Train reference envelope, not semantic coverage or calibrated OOD probability |       nan |     nan |

参考包络是Real Train跨Session最近邻距离P95；仅由Real Train统计确定，非训练Scaler或输入变换。标准化/白化只用于分析距离，不改变Encoder、Decoder或C0的预处理。Raw距离使74.48%的Natural Validation落在参考包络内，而Train-only白化后仅26.30%；Real Validation白化后为92.69%。这说明共同支持判断对距离度量敏感，不能用原始欧氏距离或二维PCA重叠直接宣布可转移。

每个窗口保留，不挑选“支持内”样本生成；分Root、Task、stage比例见support_by_group_stage_task.csv。边际分位数外点比例只是高维描述，不是校准的OOD概率。所有PCA使用同一个Real Train基底，与C25独立潜空间的不可比PCA不同。

## 4. Stage C：原Decoder零样本输出

使用C13原deterministic_decode自由运行接口：保留Mask、循环反馈、Gamma幅值分支和固定positive_scale。对Real/Natural Validation使用完全相同的设备和batch64，不引入Teacher Forcing、不裁剪、不改阈值。保存780与905条完整4×712输出。

Real重建复现C13：Mask逐位相同，数值allclose(rtol=1e-5,atol=1e-5)，最大浮点差约2.86e-5；不是新训练后的模型。输出数值差来自推理batch浮点顺序，不进行修正来追求指标。

| source              |   N | finite   | nonnegative   |   zero_fraction |   positive_P1 |   positive_P10 |   positive_P50 |   positive_P90 |   positive_P95 |   positive_P99 |    maximum |   total_response_mean |   concentration |   absolute_frame_delta |   log_frame_delta |   lag1_correlation |   transition01 |   transition10 |   unique_numeric_windows |   mean_window_std |   all_zero_windows |
|:--------------------|----:|:---------|:--------------|----------------:|--------------:|---------------:|---------------:|---------------:|---------------:|---------------:|-----------:|----------------------:|----------------:|-----------------------:|------------------:|-------------------:|---------------:|---------------:|-------------------------:|------------------:|-------------------:|
| Real_original       | 780 | True     | True          |       0.8342868 |  2.501251e-05 |   0.0004252126 |   0.005852927  |     0.3011431  |      1.389087  |      4.570212  |   10.49105 |             26.80134  |     0.01646001  |           0.0006311995 |       0.000395399 |          0.9993105 |   0.01106803   |    0.05401985  |                      780 |        0.0464398  |                  0 |
| Real_reconstruction | 780 | True     | True          |       0.8353392 |  0.0002296131 |   0.0008137504 |   0.006694925  |     0.5533424  |      1.390353  |      3.544257  |   12.61104 |             25.30078  |     0.01953983  |           0.005529205  |       0.002567672 |          0.981649  |   0.009795269  |    0.04157604  |                      780 |        0.03657763 |                  0 |
| Natural_original    | 905 | True     | True          |       0.8605566 |  4.203895e-45 |   6.662549e-38 |   3.701769e-12 |     0.024      |      0.1824405 |      0.8808682 |    1       |              3.515626 |     0.009528424 |           0.0005248079 |       0.000407264 |          0.9845644 |   0.0007411357 |    0.003622637 |                      905 |        0.02729425 |                  1 |
| Natural_zero_shot   | 905 | True     | True          |       0.7396005 |  0.0001237508 |   0.0007082116 |   0.00701157   |     0.09871851 |      0.2783424 |      1.475737  | 1207.838   |             14.66653  |     0.02201826  |           0.01074203   |       0.006025322 |          0.454248  |   0.01658811   |    0.2101523   |                      880 |        0.06664363 |                  0 |

## 5. Stage D：输出保持失败，停止C0

| feature                 |     Spearman |     Pearson | interpretation                                                                    |
|:------------------------|-------------:|------------:|:----------------------------------------------------------------------------------|
| total_response          | -0.03865045  | -0.0336473  | association across unchanged windows, NOT intervention or physical correspondence |
| response_max            | -0.1125866   | -0.03702325 | association across unchanged windows, NOT intervention or physical correspondence |
| index_positive_fraction | -0.495894    | -0.4995304  | association across unchanged windows, NOT intervention or physical correspondence |
| concentration           |  0.119928    |  0.09292026 | association across unchanged windows, NOT intervention or physical correspondence |
| absolute_frame_delta    | -0.007198076 | -0.07769422 | association across unchanged windows, NOT intervention or physical correspondence |

输出finite、非负、905窗中880个数值输出不同，因此不是全常数坍缩。但输出峰值1207.8380，Real Train峰值12.2691；有38窗超出Real Train观测最大值。没有据此截断或删除任何样本。异常窗口保留在window_output_audit.csv，最大异常来自NutAssembly。

输入/输出总响应Spearman约-0.0387，Mask密度约-0.4959，时间变化量约-0.0072；它们是跨原始窗口的关联，不是受控干预、也不能因物理单位未知而被解释为力保持。单看相关性不足以证明完全没有任何信息，但与严重尾部放大、时间结构失真共同构成停止理由。逐Root/Task与300次Root Bootstrap保留，避免把总体均值当作所有任务同样失败。

Natural输入lag1相关约0.9846，输出约0.4542；绝对相邻变化从0.0005248到0.010742。整体精确零比例从86.06%变为73.96%。同索引Mask IoU约0.0901仅作为数组响应结构诊断，不能称为物理接触位置保持。

| value        |   N |    maximum |   zero_fraction |   absolute_frame_delta |   lag1_correlation |   unique_numeric_windows |
|:-------------|----:|-----------:|----------------:|-----------------------:|-------------------:|-------------------------:|
| push         |  84 |   10.41984 |       0.724159  |            0.008765295 |          0.7389091 |                       84 |
| lift         | 111 |   13.92037 |       0.7875766 |            0.007007006 |          0.8954071 |                      111 |
| stack        | 165 |   14.92745 |       0.8542603 |            0.006082723 |          0.9443879 |                      150 |
| nut_assembly | 302 | 1207.838   |       0.6376439 |            0.0187626   |          0.4620203 |                      302 |
| pick_place   | 243 |   10.99948 |       0.7718795 |            0.006327227 |          0.8659211 |                      233 |

不同任务表现不一致，因此不把全局异常简化成所有样本失效，也不把某个任务相对稳定当成五任务迁移可行。当前不足以保留输入响应并形成可靠零样本转换。

### 为什么没有11个C0成绩

原指令明确“只有Stage A-D未触发停止条件时才执行”。数值finite/非负/非常数只是初筛，不能掩盖后续尾部与响应保持失败；因此StageD_screen.json将结构停止原因写明，保留早先numeric_screen。C0_evaluation_status.csv逐实例标NOT_RUN，BA/AUC/D均UNKNOWN，未重新训练或推理任何C0。evaluate.py实现仅保存，未执行；即使它可能给出高D，也不能绕过输出保持要求。

## 6. Stage F：P40独立候选审计

P40 selected_model/checkpoint.pt有共享Mask/Amplitude Encoder与两层Temporal Transformer，输入4×712、输出4×64 feature Z；两域确实使用同一数学函数和共同预处理。训练是P37 Real5153窗与P37转换Sim1824窗，不是当前Natural五任务Train4049窗。

它不输出VAE mean/logvar，没有712D Decoder；Stage Head和48维Structure Head不等同712D触觉重建。因而只能作为独立表示候选，不能把4×64直接送入C13 Decoder。此次只核对代码和权重Hash，没有P40推理或训练，也未与C26结果混合。详见P40_shared_encoder_audit.json。

## 7. 决策与唯一建议

**共同数值坐标成立；部分支持重叠；语义对应未建立；完整零样本迁移未成立。** 四者分别报告，不互相替代。

本轮归为CONDITIONAL GO仅限继续分析输入分布与Encoder泛化，而不是批准GMM/OT或C27。当前“共享C13 Encoder直接接原Decoder”的零样本候选停止。不能仅因为共享坐标、finite或局部PCA重叠就训练Transport。

**唯一建议：若继续，先只定位Natural的输入/处理差异与共享Encoder分布外读出失败之间的关系，使用现有冻结模型和原分组；不继续修Decoder，不拟合GMM，不开启新神经网络训练。** 物理通道认证仍为UNKNOWN，后续解释必须保留这个限制。没有自动进入C27。

## 8. 完整性与产物

输入合同、物理映射证据、共享Encoder latent、Train-only分析几何、支持统计、两份完整Validation输出、逐窗异常、分Root/Task统计、P40审计、代码与Hash均位于本目录。原始数据/C13/C0/P40文件Hash未变；没有Test/Holdout访问、训练、GMM、OT、服务器操作或commit/push。运行耗时在inference_freeze.json；训练Epoch/Optimizer Step均为0。
