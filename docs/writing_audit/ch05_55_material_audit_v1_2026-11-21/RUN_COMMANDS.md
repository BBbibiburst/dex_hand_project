# 执行与审计范围

`PYTHONDONTWRITEBYTECODE=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch05_55_material_audit_v1_2026-11-21/collect_evidence.py`

收集器仅静态读代码/CSV/JSON，以weights_only=True安全载入Checkpoint张量并提取形状、numel和标量；读取Train/Validation配对数组逐行比对既有Hash。没有import项目模块、实例化模型、forward/backward、训练、物理仿真或C0评价。报告生成用Python标准库；不改历史文件。

已有源代码正文含Test访问路径，本轮只读这些代码，不执行，也不打开其指向的Test/Holdout数组或预测文件。历史决策/汇总中的指标不作为新实验、不回算C0。
