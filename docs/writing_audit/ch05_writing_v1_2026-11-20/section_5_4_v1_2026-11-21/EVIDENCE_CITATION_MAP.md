# 逐段论点与来源

代码根C3：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`。本轮使用已通过审核的专项审计，并再次核对models.py、pipeline_cpu.py、complete.json、历史训练日志与冻结权重元数据；没有运行模型。

|位置|论点|原始来源/审核索引|边界|
|---|---|---|---|
|第1段与架构图|独立双VAE；Sim两类输入自重建|code/pipeline_cpu.py:27–56；专项V01/V15|无跨域共享参数|
|第2段、式1|输入拼接、尺度、单层GRU、后验采样|code/models.py:7–12,29–33；scale_fit|4×64是每窗表示，非共同坐标|
|第3段、式2|时间AR Mask、双Gamma及Mask条件|models.py:13–28；专项V04–V08|同帧条件似然因子化；Gamma分量非物理标签|
|第4段、式3|BCE/正值NLL/平均KL；训练与自由推理|models.py:22–33；pipeline_cpu.py:21–39|保留KL的1/256、尺度Jacobian和TF差异|
|第5段|数据、尺度、batch、轮次、选定合同|model_training_registration.json；两域complete.json/history.csv；checkpoint_metadata.json|总Step不等于所存权重Step；不重新选择|
|第6段|C13/C23/C24与潜空间局限|专项CH05_54_VERSION_COMPARISON.md及原始parameter_change_audit.json|无C0通过数、无版本拼接|

外部文献仅复用`kingma2014vae`（Kingma与Welling，Auto-Encoding Variational Bayes，ICLR 2014），支持VAE及重参数化背景。项目网络、Gamma读出和训练合同来自实际代码，不归给该论文。未新增或修改BibTeX。

图为现有网络的数据流示意，不是实验截图、物理Taxel图或推理结果。未使用Test/Holdout样本或预测。
