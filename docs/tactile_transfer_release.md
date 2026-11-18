# 712D触觉迁移版本说明

日期：2026-11-18。封存版本：**C27-D090-v1**。

## 内容

- [独立模型与验收包](../releases/tactile_c27_D090_v1/README.md)
- [D专项验收报告](../releases/tactile_c27_D090_v1/验收报告.md)
- [原BA/AUC/D完整结果](../releases/tactile_c27_D090_v1/results/full_BA_AUC_D.csv)
- [C23–C27研究代码与报告](../experiments/tactile/2026-11-18/README.md)

模型路线为原C3独立Real/Sim VAE、GRU Translator、Real Decoder，完整输入输出均为float32 `[N,4,712]`。GMM只用于训练阶段统计物理伪配对；部署无需Real、GMM、探针或域标签。

## 验收范围

**原11个冻结C0的D专项11/11通过，最低D=0.976045。原BA/AUC/D联合门槛仅2/11通过。** 本版按用户要求作为D专项备用材料封存；没有改写原研究报告，也不宣称物理对应或来源结构保持已充分认证。

## 运行

在项目根目录，使用已有numpy、torch、scipy环境：

```bash
python releases/tactile_c27_D090_v1/code/verify.py
python releases/tactile_c27_D090_v1/code/infer.py \
  --input releases/tactile_c27_D090_v1/data/natural_sim_validation.npz \
  --output /tmp/tactile_c27_replay.npz
python releases/tactile_c27_D090_v1/code/evaluate.py \
  --generated releases/tactile_c27_D090_v1/data/translated_natural_validation.npz \
  --real releases/tactile_c27_D090_v1/data/real_validation.npz \
  --output /tmp/tactile_c27_metrics.csv
```

输出路径必须尚不存在，避免覆盖结果。推理固定CPU两线程，环境和权重Hash在包内。`SHA256SUMS.json`包含71个原封存文件；模型/数据按二进制原样保存，日志也必须保留，否则校验会失败。

本机outputs中的42MB压缩包与受版本管理的release目录内容对应；Git只保存展开版本，避免重复存储同一批模型与数据。可自行打包`releases/tactile_c27_D090_v1/`用于迁移。
