# 三类数据及阶段角色

范围为C3基础路线；C13/C27的后续适配不能回填为原始训练合同。

## 数据划分与统计单位

|数据|Train|Validation|单位与来源|
|---|---:|---:|---|
|Real|5153窗/32 Session|780窗/4 Session|P43当前划分；C3/common.py:23–24从BASE相应role索引读取；Session不是窗口|
|Natural Sim|4049窗/50 Root|905窗/10 Root|自然任务仿真；common.py:25–26的相应role windows；Root为合同登记分组|
|Paired Sim及其Real目标|1200组|240组|每组sim/real均4×712；目标来自各自Real划分，不是额外1440次真实采集|

本轮另直接读取C3/c0/group_coverage.csv并按role与real:/sim:前缀汇总，四组窗口/分组数与上表完全一致（L2静态计数，不运行C0）。C24/stage0.json登记Real四帧为同Trial/Stage内连续保留的完整帧，而非每个硬件采样，且非均匀20Hz；这是已有时序审计记录，不是本轮重新验证时间戳。

窗口形状来自既有数组核验、训练断言和推理接口，本轮不重开NPZ。Paired数量、来源与分组由本轮manifest静态复计：Train 30个Real Session、1200个唯一Real来源；Validation 4个Session、240个唯一来源。既有数组逐行Hash核验见5.5专项pairing_recount.json，属于既有L2结果，本轮仅读取。

## 实际使用矩阵

|阶段|Real Train/Validation|Natural Sim Train/Validation|Paired Train/Validation|代码/证据|
|---|---|---|---|---|
|GMM先验|Train拟合；Validation只作冻结先验下的指定目标|不用于该先验拟合|生成结果，不反向拟合先验|冻结5.3:21–40、84；既有GMM审计|
|Real VAE|Train自重建与尺度拟合；Validation选权重|不使用|对应Real目标不另开一套VAE训练池|pipeline_cpu.py:16、27–55|
|Sim VAE|不以对应Real为Sim重建标签|Train自重建；Validation参与选择|Sim端Train参与混采；Sim端Validation参与选择|同文件:27–55|
|Sim批次/尺度|无|每步32窗|每步32个Sim窗；较小池重复访问|训练等量混合；尺度按两Train池合并元素拟合，非同一加权口径|
|原Translator|Real Encoder提供Paired Real后验均值；Real Decoder提供条件NLL|不直接参与原配对训练或其选权重|Train更新；Validation选取第30–45轮中的权重|pipeline_cpu.py:54–77；最终选第30轮|
|原冻结C0训练/选择|Real Train训练；Real Validation开发选择/诊断|Natural Train训练；Natural Validation开发选择/诊断|不用于原C0训练|c0.py:14–22；冻结后不再拟合|
|冻结跨域评价|原Real Validation作正类|同划分Natural Validation转换输出作负类|不作为Natural主验收的替代|原协议与冻结5.6:5–20|
|部署|不需提供Real窗口|processed Natural输入|不需在线GMM、探针或配对目标|inference.py:1–19|

原C0形成过程包含既有Validation使用，不应将后续相同Validation称为新盲测。VAE分别拟合本域训练尺度；C0的pooled Train Scaler是另一对象，不能混称共享VAE Scaler。

## 关联、时序及独立性

Paired manifest以同一行关联Sim与Real；训练使用同一索引同时取Sim均值、Real均值和Real窗口。仅证明登记监督关系，非同步物理接触。四帧按各自保留顺序组织；没有跨域等时间戳、同频率或同物理状态的保证。Real保留帧间隔不恒定，不把差分写作物理导数。

本轮复计：Paired Train/Validation登记放置指纹分别461/106种，交叉19种，涉及97个Validation窗口。定义为首记录帧qpos与mocap_position四舍五入9位后联合Hash（C4/code/pairs.py:6–13、27–38及既有审计）。未包含速度等完整积分状态，不等于认证相同物理初态；仍明确存在登记状态跨划分复用风险。Real来源/Session隔离或窗口内容不重复，不能推出物理Root完全独立或全局无泄漏。

字典的480行/160种指纹为另一范围；不能与配对461/106混算。P37旧Real 5153/1408、Sim 1824/406不替换当前合同。C27增加Natural相关适配属于后续改进，不改写原Translator仅Paired监督事实。
