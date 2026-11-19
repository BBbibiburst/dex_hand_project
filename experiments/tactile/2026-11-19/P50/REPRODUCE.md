# P50复现说明

本目录包含已完成的单Seed、单轮实验，不能直接重新运行register.py或train.py覆盖现有终点。训练从P45 A_K1开始；Dual-VAE和Real Decoder冻结。固定Seed5001、D10/G20 Epoch、batch64、K1，主结果是G_epoch_20/F50，不选Validation最优轮。

执行环境：项目根目录，`/home/zmy/miniconda3/envs/mujoco/bin/python`；`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，Torch内部2线程。离线质量统计使用BLAS/OMP各2线程。

实际脚本顺序：register.py → train.py → assess.py → audit.py → figures.py → report.py → finalize.py。assess初次完成全部质量统计后，额外统计组标签拼接出错；随后复用已经保存的quality_summary完成报告，没有重复训练或改变指标。audit先检测到近零Mask重载差异，保留数值诊断，不调整输出或门槛。loss_invariance_audit.json是首64个Train窗口的固定2倍解析检查，不是新的迁移候选搜索。

训练恢复：resume.pt逐优化步原子保存完整G/D模型、两个Adam、RNG、累计优化时间和下一步phase/epoch/step。原目录仅在未完成且确需恢复时运行train.py；完成后不要重新执行。main模型是checkpoints/F50.pt；D50.pt保存G训练前的冻结判别器，F50同时包含其完整状态。全20个G Epoch Checkpoint保留。

## 独立推理

输入是原始非负float32 processed `[N,4,712]`，不要自行标准化。NPZ键为values，groups/stage/units/task如存在则原样输出。输出文件必须尚不存在。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zmy/miniconda3/envs/mujoco/bin/python outputs/tactile_p50_residual_multiview_2026-11-19/code/infer.py \
  --checkpoint outputs/tactile_p50_residual_multiview_2026-11-19/checkpoints/F50.pt \
  --input INPUT.npz --output OUTPUT.npz --device cuda
```

注册训练推理后端是CPU Sim Encoder缓存、CUDA共享field和Real Decoder/Refiner，batch64。CPU路径可用于诊断，但不认证跨后端精确零Mask一致。CUDA独立重载幅值满足记录容差，近零Mask有差异，详见inference_audit.json；不修改阈值来掩盖。

该Checkpoint为C类失败研究产物，不能替换旧0.90验收版。全量结果在REPORT.md，原任务在TASK.md，原门槛在registration.json。没有Test/Holdout、C0、生产程序或P51。
