# R1逐段证据与改动映射

|段落/对象|依据|R1处理|
|---|---|---|
|观测语义|P40 sensor_semantics_report；源signal_processing处理合同|统一“传感器原生raw、仿真processed、Real processed、未转换Sim”；“串扰处理”改“邻域扩散处理”，不暗示真实串扰标定|
|版本范围|P39/data_counts.csv；C3/model_training_registration.json|P37统计不外推P43，P43仅合同描述，不新做差异实验|
|激活支持、幅值分位数|P37 train.npz real/sim；P39 describe.py与mask/amplitude_statistics.csv|只读静态核验分子分母与正值池，完整精度在numeric_check.json|
|表格与D|P39 run.py、probe_metrics.csv、registration、p39_report|保留三个值及Amplitude4×5；补D定义、11实例组成及Full复用意义|
|时序数据|P37 validation.npz real/sim；P39 temporal_statistics、P42 window_time_contract|明确P37 Validation，第1→2帧N×712，Real非均匀采样；不作频率解释|
|历史采样机制|P42 p42_report及已有源码链|从5.1移出到MATERIAL_FOR_5_4_MASK_SAMPLING.md，未改5.4|

源TeX中表引用保留规范\ref{tab:ch05_domain_feature_difference}，正式表号应由LaTeX在整章生成；阅读稿按5.1首表显示“表5-1”，不把解释性括号当正式表号。未来整章接入本表后，5.3及以后已有预览表号会自动顺延，不能将当前独立预览编号硬编码进正文。本轮不编译、不修改后续章节。

原始别名raw只在审计来源选择器中用于追溯CSV，不进入R1正文解释；本节没有新增文献、外部检索或参考文献条目。
