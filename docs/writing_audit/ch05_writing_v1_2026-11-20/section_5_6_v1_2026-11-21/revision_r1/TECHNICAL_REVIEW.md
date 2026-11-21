# 5.6 R1定点技术复核

日期：2026-11-21。只读历史代码、CSV和登记信息；未执行项目代码、模型、C0或仿真。仅修订重叠定义、C24统计对象及汇总表头，保留V1和原有实验结论。

## 1. 19种状态与97窗：准确对象与统计条件

准确组名为**Paired Train（1200窗）与Paired Validation（240窗）**。来源：`outputs/tactile_p43_c4_root_cause_audit_2026-11-16/paired_source_inventory.csv`与`paired_cross_split.json`的`exact.reported_root`。

本轮只重新归并已有source inventory的标识字段：Train登记461种、Validation登记106种，集合交集19种；97是满足“本Validation行的reported_root属于Train reported_root集合”的Validation行数，不是97种状态或97个独立Root。已保存19个指纹及97个target ID于placement_overlap_check.json，没有打开Episode数组重新运行物理状态提取。

原始定义见C4/code/pairs.py:11、27–29：

`SHA256(ep['qpos'][0].round(9).tobytes() + ep['mocap_position'][0].round(9).tobytes())`

这是放置后首个记录帧的qpos及探针mocap位置字段，不是完整reset状态，更未包含qvel、控制命令、探针方向/半径、接触与求解器内部状态等全部物理条件。相同Hash表示所登记、已舍入字段的字节一致性，**不能认证物理初态完全相同**。九位小数舍入阈值不应被称为真实物理分辨率。

据此能够确认登记放置状态层面的跨划分复用，说明不能仅凭Real来源隔离宣称物理初态独立；不能确认全部物理初态相同、所有Validation都泄漏，或量化复用对最终分类指标的因果贡献。历史JSON的完整reset与robot reset各共享1种是另一字段统计，不能和19种放置后指纹混用。

正文已改为明确Paired组名、19与97的统计单位、九位小数字段及未记录完整状态的边界。其余独立性结论不扩大。

## 2. C24四项指标：原始实现与比较范围

直接依据：`outputs/tactile_c24_temporal_amplitude_repair_2026-11-18/code/report.py:14–23`的`summary(a,ref)`，结果为`Real_C13_C24_structure.csv`。同一组780个Real Validation窗、4个Session、120个Trial，每窗4×712；参考ref是原Real，a分别为C13/C24已保存重建。数据角色另由data_contract.json确认。这里只读代码与既有CSV，不加载重建数组、不重算模型或统计。

记N=780、J=712，Y为某版本输出、X为原Real。

### temporal error：原字段L_temporal

$$L_{\rm temporal}=\frac1{N\cdot3\cdot J}\sum_{n=1}^N\sum_{t=1}^3\sum_{j=1}^J
\left|[\log(1+Y_{n,t+1,j})-\log(1+Y_{n,t,j})]-[\log(1+X_{n,t+1,j})-\log(1+X_{n,t,j})]\right|.$$

代码为沿axis1作np.diff后，abs(dy-dt).mean()。既不是输出自身mean_abs_log1p_frame_delta，也不是把两个平均变化量再相减；包括所有Taxel与零/正值位置，不仅统计共同正值。对保留相邻帧作差，没有除实际时间间隔；Real采样不均匀，不能解释成速度或动力学误差。

### reconstruction MSE

$$\mathrm{MSE}=\frac1{N\cdot4\cdot J}\sum_{n,t,j}(Y_{ntj}-X_{ntj})^2.$$

代码((a-ref)**2).mean()，包括零值、所有四帧和所有通道，不是正值条件MSE或latent MSE。报告用float64统计，不等于重新生成了float64触觉。

### positive P99

先取全部780×4×712输出中a>0的元素，再计算np.quantile(pos,0.99)。不是逐窗或逐Patch的P99再平均，不是Real/输出共同正值交集，不是归一化响应/误差分位数。

### window total response standard deviation

原字段名`mean_per_window_total_std`，实际实现为`a.sum((1,2)).std()`：

$$S_n=\sum_{t=1}^4\sum_{j=1}^{712}Y_{ntj},\qquad
\sigma_S=\sqrt{\frac1N\sum_n(S_n-\overline S)^2}.$$

这是**四帧全通道总响应在780窗之间的总体标准差（ddof=0）**，不是窗内时间标准差的均值，也不是`mean_sample_std=a.std(0).mean()`。原字段前缀mean并不代表这里对逐窗std又取平均。

四项均是未作力学标定的processed响应描述，不称真实接触力、压力或物理多样性真值。总量离散度下降可以描述为输出总量差异收缩，不能单独证明语义模式坍塌；唯一窗口数与空间结构仍需另看。

|字段|C13|C24|
|---|---:|---:|
|L_temporal|0.00281496740036965|0.0016136413835846951|
|reconstruction_MSE|0.026319601646693502|0.02715744920401311|
|positive_P99|3.544258334636667|3.2897100281715383|
|mean_per_window_total_std|67.58366986476672|57.522235290821406|

正文保留原舍入精度及数值，补足780窗比较范围、log1p相邻增量、全窗MSE、汇总正值池和总响应跨窗std定义，未改变指标。

## 3. C13冻结Validation全部实例

唯一读取结果：`outputs/tactile_c13_frozen_c0_validation_2026-11-18/metrics.csv`，905个迁移Natural窗口与780个Real窗口，原冻结C0，不是Real自重建CSV或新Probe。下表按原标量判定，显示六位小数；未运行C0或重算AUC。

|实例|BA|AUC|D|联合满足|
|---|---:|---:|---:|---|
|mlp_11|0.497436|0.571536|0.994872|否|
|mlp_12|0.498718|0.589597|0.997436|否|
|mlp_13|0.498077|0.601097|0.996154|否|
|gru_11|0.498718|0.187208|0.997436|否|
|gru_12|0.498718|0.191583|0.997436|否|
|gru_13|0.498718|0.308620|0.997436|否|
|transformer_11|0.497436|0.266441|0.994872|否|
|transformer_12|0.496154|0.236963|0.992308|否|
|transformer_13|0.498077|0.675462|0.996154|否|
|logistic|0.498718|0.658306|0.997436|否|
|knn|0.498629|0.497752|0.997259|是|

11/11满足BA/D门槛，仅kNN同时满足AUC。三个MLP、Logistic、Transformer13的AUC高于0.525；三个GRU、Transformer11/12低于0.475。正文“MLP与Logistic高于上限，GRU与部分Transformer低于下限，仅kNN通过”准确，无需扩大正文或增加完整表格。Transformer13虽未单独点名，仍在“仅kNN通过”的结论范围，完整反例在本报告保留。

## 4. BA与D的等价性及汇总表修订

$$D\ge0.90\iff 1-2|\mathrm{BA}-0.5|\ge0.90\iff |\mathrm{BA}-0.5|\le0.05\iff\mathrm{BA}\in[0.45,0.55].$$

因此在本历史定义下，两门槛代数等价，不是两份独立证据。历史BA和D逐值记录保留，AUC仍须独立满足[0.475,0.525]。实际CSV标量与公式在数值容差内一致，未改变原判据。

汇总表列名由“联合满足实例数／D专项满足实例数”改为“BA/D及AUC联合／BA/D等价门槛”，11/11的第二列现在明确表示同一等价门槛，不暗示新增证据。所有实例结果、显示小数、数据角色与数值均未更改。

## 5. 范围与待审事项

仅R1副本有三处定点变更（两段统计语义及一行表头）。V1、5.3/5.4/5.5、主文件、参考文献、代码/数据/权重/合同不变。TeXcount报告随附；本轮不编译、不冻结。表头宽度待后续授权PDF检查，不据此调整模板。等待人工审核。
