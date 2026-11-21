# 执行记录与范围

日期：2026-11-21（Asia/Shanghai）。

1. 通过rg、sed、nl、cat只读定位和读取指定规划、既有审计、冻结5.3及C3/C13/C23/C24源代码与合同。
2. 静态收集命令：
```sh
PYTHONDONTWRITEBYTECODE=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch05_54_material_audit_v1_2026-11-21/collect_evidence.py
```
只解析AST、加载已有训练计数及CSV/JSON；Checkpoint通过torch.load(weights_only=True,map_location='cpu')读取张量形状、计数和差异，未实例化网络或执行forward。
3. 以Python标准库生成专项审计文件，补记C4 IoU实现来源Hash。首次文档生成命令使用不可用的python别名返回127；随后改用python3成功生成，无训练/推理进程。
4. 对baseline.json的保护文件及audited_inputs.json全部来源重算SHA256，仅作完整性检查。

未读取Test/Holdout原始样本或预测文件；未执行C0，未改TeX/Bib/生产代码/模型/历史结果。未编译、commit、push。读取的既有汇总报告可能含历史Test结论，本报告未提取其预测或据此选择模型。
