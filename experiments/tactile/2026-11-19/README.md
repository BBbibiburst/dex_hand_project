# P44–P51 触觉迁移研究归档（2026-11-19）

本快照延续C23–C27归档，保存P44–P51的原实验脚本、报告、配置、关键指标、图表及选定研究模型。原始`outputs/`未修改；源码按实际实验保留，没有改写成一套新的生产系统。

|阶段|内容|实际结论|
|---|---|---|
|P44|Dual-VAE与共享参数迁移场|建立完整4×712输出与多步迁移实验|
|P45|显式Mask及梯度路径消融|Continuous-only A_K1作为后续F0基线|
|P46|Real自然变异与迁移等价性审计|记录自然差异、分组评价与局限|
|P47|条件触觉评价可行性|增加覆盖、输入依赖及退化对照诊断|
|P48|四路线统一冻结评价|统一比较多步路径，未得到完整质量成功|
|P49|逐轮更新并冻结判别器，三轮|终点FR100%，质量失败，B类部分成功|
|P50|零初始化输出Refiner与三视角D，单轮|Joint FR100%，Real接受68.08%，分布严重退化，C类|
|P51|只读D50可信度及分数包络|NO-GO，不启动P52/P53；停止本条收尾实验扩展|

优先阅读[P51最终决策](P51/REPORT.md)、[P50失败分析](P50/REPORT.md)、[P49逐轮实验](P49/REPORT.md)。所有结果属于已有开发数据，不能称为新盲测。

## 当前质量基线与失败对照

|模型|SWD↓|MMD↓|定位|
|---|---:|---:|---|
|F0（P45 A_K1）|0.206609|0.343678|当前已比较候选中的分布质量基线|
|P49 F3|0.411330|0.676806|高FR但质量失败对照|
|P50 F50|4.247227|0.882140|多视角+Refiner失败对照|
|P50关闭终点Refiner|0.340365|0.365503|仅离线结构诊断，不替换主模型|

F0不等于已完成完整0.90验收。FR、旧D指标、AUC和质量护栏分别报告；不以FR100%代替完整迁移成功。

## 模型与独立推理

已纳入原文件，包含原优化器状态，Hash见source_inventory.json：

- `P49/checkpoints/F0.pt`：从P45 A_K1保存的原初始化状态。
- `P49/checkpoints/F3.pt`、`D3.pt`：P49最终生成器与判别器。
- `P50/checkpoints/F50.pt`、`D50.pt`：P50最终失败候选及冻结判别器。

从仓库根目录使用既有Python/PyTorch环境：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/tactile/2026-11-19/P49/code/infer.py \
  --checkpoint experiments/tactile/2026-11-19/P49/checkpoints/F0.pt \
  --input INPUT.npz --output OUTPUT.npz --device cuda
```

P50使用同目录下的`P50/code/infer.py`及`P50/checkpoints/F50.pt`。输入须为非负finite float32 `values[N,4,712]` processed tactile；不能先自行标准化。输出文件必须不存在。组、阶段、窗口ID及Task等提供的元数据原样保留。

F0历史CPU输出与CUDA自回归Decoder存在后端差异；P50重载响应在容差内一致，但近零处精确Mask不完全一致。见各报告及inference_audit.json，不保证跨后端逐位复现。P50是失败研究产物，不用于替代验收包。

## 完整性与归档边界

`source_inventory.json`记录1170个原文件的路径、大小、SHA256和是否归档；`archive_hashes.json`校验实际Git快照。各实验的原integrity/manifest文件仍是**原outputs目录**的证据，不能直接当作精简归档的完整文件列表。

原始Real/Sim数组、生成数组、逐步/逐EpochCheckpoint、较大的Bootstrap和距离明细不重复放进Git。它们仍保存在本机原outputs目录；本快照没有声称仅clone即可重训所有历史实验。历史脚本的输入及输出常量仍指向原`outputs/...`，完整复跑需要对应原数据和依赖。需要重跑时先建立新的实验目录，不能覆盖本次归档或历史结果。

旧[C27 D专项0.90验收版](../../../releases/tactile_c27_D090_v1/README.md)保持原样；它的限定验收口径见包内说明，不因本次研究结果而改写。

本次仅整理归档并提交，不启动训练、不改服务器采集程序。
