# P49复现及恢复说明

本目录是已完成实验，不要直接在这里重新执行训练或注册脚本。历史原件、初次CE门槛停止和G1完成后的日志恢复记录均保留。复现实验应先制作独立工作副本并修改代码中的输出目录常量；不能覆盖本目录。

实际顺序为：code/register.py → code/train.py（D0/D1完成，额外CE门槛停止）→ code/review_D1.py（冻结推理，复核分组与判别能力）→ 用户明确授权的protocol_amendment_2026-11-19 → code/resume_after_review.py → code/train.py（继续G1到G3）。过程中日志故障从完整权重和Adam状态恢复，已完成优化没有重复。baseline_audit.py在独立stopped_audit目录补充基线；snapshot_audit.py、final_tables.py、report.py及finalize.py仅统计、作图或校验，不训练。

Seed4901，K1，D0为20 Epoch，每轮D10/G20 Epoch，最多三轮。所有历史输入路径、原预算和初始Hash见registration.json。原registration保持不变，CE门槛的授权变更单独存档，不能把旧硬门槛覆盖成原本不存在。

## 冻结F3推理

在项目根目录、原mujoco环境运行，INPUT.npz须包含float32非负processed `values[N,4,712]`；OUTPUT.npz必须不存在：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python outputs/tactile_p49_round_frozen_adversarial_2026-11-19/code/infer.py --device cuda --checkpoint outputs/tactile_p49_round_frozen_adversarial_2026-11-19/checkpoints/F3.pt --input INPUT.npz --output OUTPUT.npz
```

注册后端是CPU Encoder缓存、CUDA field/Decoder，batch64。该路径复验Mask完全相同，幅值在记录容差内一致。纯CPU路径在少量Mask上存在数值分岔，不能宣称跨后端逐位复现；详见inference_backend_check.json。F3只是B类部分成功研究模型，不是完整质量合格验收版。

完整结果见REPORT.md；原始优化日志见training_log.csv；逐组指标见group_metrics.csv；原始预测分数在outputs/round*_predictions.npz。所有已记录操作及恢复见run_commands.json和io_recovery_note_2026-11-19.json。
