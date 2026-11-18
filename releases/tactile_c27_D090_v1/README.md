# 712D触觉迁移D≥0.90专项验收材料

**版本：C27-D090-v1｜日期：2026-11-18｜状态：11/11冻结分类器D专项达标。**

D最小值：**0.976045**；最大值：**0.999469**。输入905个Natural Sim Validation窗口，参考780个Real Validation窗口。所有实例使用原迁移前冻结C0，不重新训练或修改权重、Scaler、log1p、标签方向及0.5阈值。

**口径说明：本包只按D≥0.90专项整理。原BA/AUC/D联合门槛为2/11，尚未通过；完整AUC和输出保持诊断保留在附录。本包不作为全新盲测、物理Taxel对应或来源接触信息保持已充分验证的证明。**

## 模型和输入输出

Natural Sim processed float32 [N,4,712] → 原C3 Sim Encoder → GRU Translator → 原C3 Real Decoder → Real-like processed float32 [N,4,712]。采用后验均值、原硬MAP Mask及Gamma混合均值，保留原通道和四帧顺序，无裁剪/补值/阈值修改。

GMM为训练阶段构造统计物理伪配对的工具，部署无需GMM、Real、探针或域标签。Real/Sim VAE均为原C3独立模型，Translator是本轮单Seed候选第14轮Checkpoint。不是C13幅值修复或C26共享Encoder。

## 文件清单

- `models/`：Real VAE、Sim VAE、GRU Translator三个完整Checkpoint。
- `code/infer.py`：独立推理入口，不依赖原项目目录。
- `code/evaluate.py`：原11C0冻结评价入口。
- `c0/`：11个原冻结分类器、原Scaler和kNN参考库。
- `data/`：原始Natural Validation、原始Real Validation及完整迁移输出。
- `results/D090_results.csv`：D专项表和混淆矩阵。
- `results/full_BA_AUC_D.csv`：包含AUC和原联合门槛的完整结果。
- `training/`：训练配置、日志、选Checkpoint记录及原研究代码。
- `provenance/`：来源Hash、原注册表、完整研究报告及原不替换默认模型的决定。
- `acceptance_scope.json`：专项范围；`SHA256SUMS.json`：全部文件校验。

## 复现方法

在本目录执行，输出使用不存在的新路径：

```bash
python code/verify.py
python code/infer.py --input data/natural_sim_validation.npz --output /tmp/c27_d090_replay.npz
python code/evaluate.py --generated data/translated_natural_validation.npz --real data/real_validation.npz --output /tmp/c27_d090_metrics.csv
```

推理只需`models/`与`code/models.py`，评价另需`c0/`和`code/probes.py`。CPU固定2线程；环境版本见environment.json。训练复现需要原项目训练数据，`training/research_code`保留其路径和代码，不能称为脱离原数据即可重新训练。

## D专项结果

|实例|BA|D|D≥0.90|
|---|---:|---:|:---:|
|mlp_11|0.498541|0.997082|是|
|mlp_12|0.498718|0.997436|是|
|mlp_13|0.499734|0.999469|是|
|gru_11|0.498718|0.997436|是|
|gru_12|0.511978|0.976045|是|
|gru_13|0.511425|0.977150|是|
|transformer_11|0.497988|0.995977|是|
|transformer_12|0.499469|0.998938|是|
|transformer_13|0.503602|0.992796|是|
|logistic|0.501480|0.997039|是|
|knn|0.506364|0.987272|是|

## 使用与解释边界

D与BA数学等价。本包多数迁移样本被原C0判为Real，BA接近0.5；这足以满足这里的D公式，但不等于所有残余域差异消失。AUC排序差异仍存在，所以不能把“D专项11/11”改写为“原联合标准11/11”或“真实分布已完全一致”。

如果最终采用D专项口径，引用本包名称、版本、数据划分和公式；不要删去适用范围或将历史Validation称为全新独立Test。此次只封存备用材料，不改变原研究报告、不自动部署、不启动新训练。
