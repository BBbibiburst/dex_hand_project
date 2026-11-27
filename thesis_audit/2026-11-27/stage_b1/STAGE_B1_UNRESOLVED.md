# Stage B1尚未解决与人工确认事项

## 本轮范围之外

- A01图3-6保持原占位，不插图、不删除、不生成截图。已有素材可用性见下表，后续仅在Stage B2授权后处理。
- A02历史模型完整版本链UNKNOWN和A03 v4独立Root UNKNOWN均原样保留。它们限制结论外推，不被本次文字修订自动解决。
- 不新增物理配对、实机收益或全量生产数据认证；不把旧Probe、C0、逐轮D及包络混为同一验收。

## 图3-6既有素材只读检查

以下目录均位于outputs/five_task_p32_b_2026-11-03/。只检查文件存在、字节大小、结果及manifest，不提帧、不解码重验、不启动仿真。

|案例|视频存在/大小字节|历史RGB状态|历史任务success|source manifest存在|视频SHA256|
|---|---|---|---|---|---|
|`lift_sugar_rgb_00`|True/1184558|DECODE_VERIFIED|True|True|`29f3a6a17eaaa3f7db2038e41866ff5a923239ef348a84cabc9e2a41766b93fc`|
|`nut_original_handle_00`|False/0|NOT_YET_IMPLEMENTED|False|True|`MISSING`|
|`nut_handle_lower_00`|False/0|NOT_YET_IMPLEMENTED|False|True|`MISSING`|
|`nut_retain_lower_rgb_00`|True/1837682|DECODE_VERIFIED|True|True|`a95fbe6c81cdcee0a494cea5566555f9c8dcda5995ca65c85a0f764927e4403c`|

Lift成功与Nut retain-lower成功案例视频可定位，历史结果称DECODE_VERIFIED；这是历史记录，不是本轮完成解码验收。Nut原始失败和降低抓点仍失败的两个目录均无agentview.mp4，结果标NOT_YET_IMPLEMENTED。尚不能声称图3-6失败画面已有完整可用素材。

`lift_sugar_strict.json`指向Lift同一source manifest，历史Strict结果为FINAL_VERIFIED；其回放结果frames=322，原记录/视频frames=447，不能直接把Strict结果帧号当作视频帧号。Stage B2须结合stage/控制索引确认要展示的时刻和判定范围，不能用同对象但不同运行的图片充当该轨迹证明。Nut两个失败目录保留源轨迹和result，可进一步寻找与这些轨迹直接绑定的已有视频；本轮不重新渲染或以成功视频冒充失败。

## 仍需人工确认

1. 审阅新增的三主线/四工作映射和数据桥接，确认行文简洁程度。
2. 审阅7.1/7.2对分层检查及修订后开发协议的概括；数值与原结论未变。
3. 部分小节补定义后超过早期建议篇幅，本轮未为压字数另删技术说明；由全文审核统一取舍。
4. 已有第一章两处表格Overfull、文献元数据警告与章末留白留待对应排版/文献阶段，不擅自修模板、参考文献或其他段落。
5. 本次仅新预览对应当前修订源；2026-11-27既有发行包仍是修订前版本，未经授权未更新或冻结。

未发现新的跨章节事实矛盾，A04–A10范围内修订完成。没有任何实验结果、公式、参数或验收门槛变化。本轮停止，不自动进入Stage B2、不冻结、不commit/push。
