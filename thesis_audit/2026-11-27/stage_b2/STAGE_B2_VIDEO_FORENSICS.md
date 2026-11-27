# 图3-6原始素材溯源

本轮只读取既有文件，用PyAV完整顺序解码两个原视频；未运行模型、仿真或历史重渲染。详细元数据、数组形状、阶段区间见video_metadata.json。

|案例|最终视频分类|本轮解码|来源及限制|
|---|---|---|---|
|lift_sugar_rgb_00|VERIFIED_DIRECT|447|source_episode/manifest.json；Root未登记，UNKNOWN|
|nut_retain_lower_rgb_00|VERIFIED_DIRECT|599|source_episode/manifest.json；Root未登记，UNKNOWN|
|nut_original_handle_00|MISSING|无视频|source_episode/manifest.json；Root未登记，UNKNOWN|
|nut_handle_lower_00|MISSING|无视频|source_episode/manifest.json；Root未登记，UNKNOWN|

## 两个成功视频

- `lift_sugar_rgb_00`：SHA256 `29f3a6a17eaaa3f7db2038e41866ff5a923239ef348a84cabc9e2a41766b93fc`；h264，1920×1080，20fps，447帧，容器时长22.35秒。全部帧解码无异常，PTS严格递增，尺寸一致。与原记录相对时间最大误差4.27e-12秒。
- `nut_retain_lower_rgb_00`：SHA256 `a95fbe6c81cdcee0a494cea5566555f9c8dcda5995ca65c85a0f764927e4403c`；h264，1920×1080，20fps，599帧，容器时长29.95秒。全部帧解码无异常，PTS严格递增，尺寸一致。与原记录相对时间最大误差1.36e-11秒。

Hash均与任务书登记一致。黑帧筛查使用RGB均值≤3且标准差≤2的启发式阈值，未命中；这不是逐像素场景真实性认证，也不能排除一切视觉异常。候选帧人工查看未见黑屏、截断或变形。

历史DECODE_VERIFIED与本轮解码检查分别记录；本轮结论来自实际解码，不冒用历史标记。视频帧数与记录行数一致，20Hz的PTS与timestamp逐行吻合，现有生产代码先记录当前行RGB再追加该行。旧manifest缺少后来新增的rgb_control_indices，故不能声称历史已有逐帧索引签名合同。

## 失败视频定向检索

检索范围为项目outputs中的媒体路径清单，以及P32原目录、Nut来源目录、相关历史报告和第三章证据索引中的准确案例引用；未扫描20T外置盘。命令结果见failure_search.txt、media_search_paths.txt、related_media.json。两个目标目录均无agentview.mp4，结果明确NOT_YET_IMPLEMENTED。原始和降低抓点失败均保留941行记录，success=false，原因字段为“grasped bolt was not stably inserted in the fixed nut”。这只是历史终止原因，不证明唯一物理成因。

其他Nut视频的目录/来源信息不足以认证为这两个目标Episode，列UNVERIFIED；retain-lower视频与它们共享部分来源，但配置/结果不同，对失败目标仅VERIFIED_RELATED。未找到可认证的直接失败视频，最终MISSING。缺失仅指本次限定检索范围，非声称世界范围不存在。

## Lift447/322控制索引

原stage数组：settle0–11、transit12–81、pregrasp82–136、approach137–181、close182–226、hold227–266、lift267–356、verify357–446（均零基）。

`source/verification/strict_replay.py:209–229`的Episode回退路径调用start_stage=approach；`source/grasp_pipeline/reference.py:137–145,182–195`选start=137、初始状态索引136、后续ctrl[137:]；`strict_replay.py:39,390–400`额外保持12步。因此447−137+12=322。按当前静态实现，回放局部控制j=0…309映射原记录137+j；j=310…321是额外保持，无对应原视频帧。

这是可复核的计数/控制索引解释，不是历史逐帧状态等价证明。历史Strict wrapper未保存每步映射轨迹，完整历史代码/XML版本链仍UNKNOWN；原录像对应执行前观测，回放测量发生于执行后。精确物理状态及时间一致性保留UNKNOWN，不将Strict帧号直接当录像帧号。截图仅依原记录stage选取，图注不称Strict回放截图。
