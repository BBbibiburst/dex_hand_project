# 候选截图与图3-6布局

6张截图均由VERIFIED_DIRECT原视频提取，1920×1080原比例，无裁剪、内容填充、重绘。帧号零基；时间为视频PTS。逐张视频/manifest/截图SHA256及Episode标识见candidate_frames.csv和.json。L2仅认证来源、现有阶段记录及本轮解码，不升级为新物理实验。

|案例|帧|秒|阶段|文件|
|---|---:|---:|---|---|
|lift_sugar_rgb_00|246|12.30|hold|candidate_frames/lift_sugar_rgb_00_frame_0246_hold.png|
|lift_sugar_rgb_00|311|15.55|lift|candidate_frames/lift_sugar_rgb_00_frame_0311_lift.png|
|lift_sugar_rgb_00|401|20.05|verify|candidate_frames/lift_sugar_rgb_00_frame_0401_verify.png|
|nut_retain_lower_rgb_00|289|14.45|hold|candidate_frames/nut_retain_lower_rgb_00_frame_0289_hold.png|
|nut_retain_lower_rgb_00|428|21.40|transport|candidate_frames/nut_retain_lower_rgb_00_frame_0428_transport.png|
|nut_retain_lower_rgb_00|548|27.40|task_verify|candidate_frames/nut_retain_lower_rgb_00_frame_0548_task_verify.png|

Lift阶段可由manifest字典及stage数组对应；Nut旧manifest无显式stage字典，名称按现有executor的STAGE_CODES解释，保留历史版本限制，不凭视觉命名。原始stage整数一并登记。

## 方案A（优先推荐）

candidate_layouts/fig3_6_scheme_a.pdf：两个成功原画面与两个失败结果文字卡片。完整保留四案例意图且不虚构失败影像；适合论文简洁呈现。截图只能展示原Episode状态，不能单凭静帧证明最终成功。成功来自对应历史任务记录。

## 方案B

candidate_layouts/fig3_6_scheme_b.pdf：两条各自独立成功Episode的三阶段序列，加失败结果分类说明。可表达阶段推进，但手部在整幅原图中较小；不能从图中认证接触细节。失败分类属于历史任务终止记录，不作因果归因。

两份scheme_*.diff只是正式figure块替换建议，未应用。采用前应将资产复制到经授权的论文资产路径并修正includegraphics相对thesis工作目录的路径，同时将相邻“保留图位”文字改为准确描述；这些均需人工授权。两个方案不同时接入，避免重复图号。当前正式PDF继续保留原占位。
