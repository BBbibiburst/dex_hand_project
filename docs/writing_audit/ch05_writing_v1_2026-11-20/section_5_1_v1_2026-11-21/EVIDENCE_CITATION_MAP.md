# 5.1逐段论点与证据

本轮只读既有CSV、合同和代码，不打开Train/Validation/Test/Holdout原始数组，不运行模型。本节没有新外部引文，证据来自项目已核材料；reference.bib不变。

|段落|论点/数字|来源与选择器|限制|
|---|---|---|---|
|1|16Patch/712标量、五指掌心、native touch与processed|P40/sensor_semantics_report.md第1、3、4节；已有第二章处理链|MuJoCo原生法向力累加≠processed压力；Real自身处理链不能假设同物理尺度|
|2|旧5153/1408、1824/406；当前5153/780、4049/905|P39/data_counts.csv四行；C3/model_training_registration.json|旧/新合同分开，非服务器全量、非独立窗口主体|
|3|精确x>0；旧Train Real17.60%、原Sim13.31%|P39/code/run.py:15–25；mask_statistics.csv role=train,stage=ALL,patch=ALL,candidate=Real/raw|数值Mask非物理接触真值；raw为Sim processed别名|
|4|正值P50、P95；有效4×5幅值|P39/amplitude_statistics.csv上述同选择器；run.py:18–22|P50:0.005002501420676708/1.039736616803566e−5，P95:1.5010055303573537/0.45948466658592224；取各域汇总正值，不是Patch平均|
|5、表|0.0272/0.1494/0.0573|P39/probe_metrics.csv role=validation；candidate=mask_after/amplitude_after/full_after分别11行的domain_confusion最小值|校准后的旧转换输出，不是原始Sim，也不是后文冻结C0|
|6|分解特征/预处理不同、不能相减|P39/run.py:15–33,41–49,72–80；registration与representation_counts|Full与Amplitude log1p，Mask直接二值；各自Train均值/std（下限0.05）标准化|
|7|第一相邻帧Mask变化1.7154%/0.0958%|P39/temporal_statistics.csv role=validation,stage=ALL,transition=0,candidate=Real/raw；describe.py:28|change_fraction是窗口×Taxel无条件均值，非P(01|0)等条件概率|
|7|历史独立逐帧采样的机制风险|P42 p42_report.md第5节及引用的采样源码链|早期历史实现，不指后续C3 AR Mask；不引用新生成结果|

## D值精确记录与实例

- mask_after：0.02718456112852663，MLP Seed12；为mask_before精确相同表示的别名，非另一次实验。
- amplitude_after：0.14935764666368123，Transformer Seed11。
- full_after：0.05725551388266892，Transformer Seed11；复用P38同条件的完整输入Probe拟合。

共五种架构：Logistic/kNN各1，MLP/GRU/Transformer各Seed11/12/13。Mask不log1p；幅值P10/P25/P50/P75/P90按每帧全部正值求，前5列放有效值、其余707列补零维持接口；any帧无正值则按预登记规则排除。实际纳入计数另由representation_counts.csv保存，不把接口4×712误写为有效幅值712D。

## 数据与图表取舍

未直接复用PNG：旧图常包含多个转换状态，当前小节只需明确限定的对照。使用一张由原CSV标量整理的表，无新曲线、热图或物理叠加图；不为图数量重复制作。没有声称逐Taxel对应或将跨域可分性归因为纯传感器差异。

原始路径/Hash见source_hashes.json；标量核对见verified_numbers.json、selected_statistics.json。本轮统计仅选取既有行/最小D及百分比显示，不重跑Probe或算新概率/AUC。
