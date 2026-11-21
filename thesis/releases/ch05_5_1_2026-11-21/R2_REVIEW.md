# R2最后定点修订

1. 保留D定义，删除“低D说明残余域信息”的过强推断，改为至少一个Probe不满足固定阈值BA等价门槛；排序能力须另看AUC，不能认定摘要已实现域混淆。
2. 明确P43 Real Train/Validation=5153/780、Natural Sim Train/Validation=4049/905；数据规模仅描述当前划分，非新差异复现。P37规模未改。
3. 明确包含至少一个全零响应帧的**整个窗口**排除，不是只删帧。Amplitude4×5、原三个D及所有既核数字不变。
4. 复用现有mujoco2026xml，在法向接触力累加句后加cite，reference.bib未改。官方来源专项说明见OFFICIAL_REFERENCE_CHECK.md。

仅在上述相关段落压缩重复说明，保留Probe组成、Full同条件复用、精度与统计限制；未改论证顺序。TeXcount1239，不重新审计统计、不运行实验。V1/R1及旧核验文件保留。

根据本轮“R2后直接PDF检查和冻结”授权，完成后形成FROZEN_DRAFT；不声称用户已经逐页查看R2 PDF，不自动进入5.2。
