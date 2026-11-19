# P48 复现

在原项目根目录运行，Python环境需numpy/scipy/torch/matplotlib。将本次code/和TASK.md复制到一个全新的结果目录，不覆盖本结果。common.py按脚本所在目录推导输出位置；历史数据/Checkpoint引用原项目路径。

设置PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。swd_paths.py实际用BLAS/OMP2线程，其他统计1线程；torch审计/冻结推理用2线程。

顺序运行register.py、historical_manifest_audit.py、generate.py、probes.py、gate_baselines.py、swd_paths.py、equivalence.py、quality.py、shared_audit.py、latent_path_audit.py、checks.py、jsd_summary.py、report.py、finalize.py。可将独立统计脚本与probes.py并行，但不要重复启动同一训练脚本。

日志依次保存run.log、historical_audit.log、generate.log、probes.log、baseline_gate.log、swd.log、equivalence.log、quality.log、shared_audit.log、latent_path.log、checks.log、jsd_summary.log、report.log、finalize.log。最后一个脚本合并日志并保存integrity.json。

probes.py已包含Original基线有效性检查，失败实例在后续候选中记录NOT_TRAINED_BASELINE_INVALID。实际首次运行时，停止标记由gate_baselines.py在同一顺序训练器完成Original基线时创建，训练器跳过这些未训练实例；集成版本等价执行同一规则。不得删除停止状态后追训。

审计分类器只复用原架构定义，不加载C0权重，使用固定P46Scaler和组等权。40轮端点固定，不根据Validation选择。Validation曲线只是日志，不能调参。

初次分类器启动在生成输出完成前因依赖检查退出，未进行研究数据训练；合成设备计时重复一次，单独统计。不会影响迁移Epoch/Optimizer Step均0的事实。

统计区间固定模型、投影和窗口库，不能作为全物理总体或新盲测置信保证。JSD代理保留负值，失效基线不计算有效相对率。SWD是1024投影平均，不是完整高维W1。所有旧C0定义保持原样。
