# 伪配对来源、数组索引和独立性

## 1. 直接数组—manifest核对（L2）

读取C3/paired_train.npz、paired_validation.npz的sim/real数组，仅Train/Validation。逐行重算两种窗口的内容SHA256，与paired_manifest.csv中对应role的行顺序比对：两组均全部一致。本轮没有运行模型或重新构造配对。

|项目|Train|Validation|
|---|---:|---:|
|sim与real形状|1200×4×712|240×4×712|
|dtype|float32|float32|
|唯一Real source window|1200|240|
|唯一Real Session|30|4|
|唯一Real Trial（既有C4审计）|566|106|
|唯一登记放置状态指纹|461|106|
|同一放置指纹最多窗口数|90|21|
|唯一Sim/Real内容Hash|各1200|各240|
|唯一登记Episode路径|1200|240|
|CLOSE/HOLD/LIFT窗|462/384/354|116/52/72|

manifest两域均无完全重复内容；同一Real来源在已接受数据中只出现一次。不能由此声称没有相邻窗口相关、没有物理初态复用或全局无泄漏。

代码verify_pairs.py:6–21验证配对来源及拼接顺序；expand.py:85–90追加已选Sim和对应Real。pipeline_cpu.py:16,59以相同位置配对编码，:61用同一排列索引b同时索引输入、目标及Real触觉，不重新做最近邻匹配。

## 2. 跨划分关系

本轮从manifest直接复核：Real source、Session、内容Hash、Episode路径无交叉；sim_root_fingerprint交叉19种。C4/paired_cross_split.json进一步记录这些共享指纹涉及97/240个Validation窗口；相同接触拓扑17种、涉及165窗；近配置诊断160窗（拓扑一致且四帧有效压入RMS≤0.05mm）。后二者为既有审计结果读取，不是本轮重新执行物理状态复核。

C4/code/pairs.py:6–13,27–38给出定义：登记Root指纹来自放置后首个记录帧qpos与mocap_position四舍五入9位的Hash，不是原始reset；full reset与robot reset各只有1种且两组共享。故有效的Real Session隔离不等于Sim独立物理初态隔离。来源路径不同/内容Hash不同也不等于物理Root独立。

这些是已量化的复用和潜在泛化风险；是否在目标任务语义下构成致命泄漏、对最终误差的因果贡献仍UNKNOWN，不能从该数量断言全部验证无效。现有训练/验证分组保持原样。

## 3. 从统计软信息到单一监督

C3/code/expand.py:37–53作GMM责任度统计，按分量mass排序取Top-5；这表示最多5个高质量目标分量构造球组，**不是选5条完整配对候选**。后续:66–82最多执行3次物理候选，遇门槛通过提前停止，再在已执行候选中按structural_loss取min，并重新检查通过/重复条件。可能找到通过候选却因min选另一不通过者而未登记；详见冻结5.3，不能称保证全局最优。

训练与验证采用同一扩展生成/准入程序，不是两套任意匹配规则。最终每个被接受Real窗口仅一个选定Sim，失败候选虽保留Episode/评分，也不会传入Translator的输入或损失。

Translator实际只读取两个4×64均值及对应Real目标数组，未输入GMM责任度、完整分量概率、多候选分布、采样不确定性、stage标签、字典配置或匹配质量权重。因此，多分量信息先压为Top-5物理构型，再压为单条已选触觉配对，最后回归posterior mean；不能称当前T在学习完整多模态p(z_r|z_s,c)。

## 4. 支持与风险

C4/paired_mapping_ambiguity_summary.csv已有诊断：Validation近Sim邻距离中位数约0.00783，对应Real目标距离中位数约14.57984。代码input_ambiguity.py限定为各自坐标内近邻/目标差异，不是直接比较两独立latent坐标的绝对距离。这支持“相近Sim输入对应差异Real目标”的风险，不能从未经标准化距离比值推导覆盖率、不可约误差或必然失败。

Natural Sim参与Sim VAE训练，但不在C3 Translator配对监督中；没有Natural的同步对应Real真值。Paired验证支持仅限当前生成规则和来源覆盖，不能外推Natural任务输入的完整映射正确性。

独立性直接复算与既有审计的边界记录于pairing_recount.json。未读取任何Test/Holdout数组、预测或服务器数据。
