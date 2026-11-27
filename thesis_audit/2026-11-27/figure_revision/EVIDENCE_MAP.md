# 图表及伪代码证据映射

|内容|依据|边界|
|---|---|---|
|机械臂、手及触觉分布MuJoCo图|assets/robots/rm75b/rm75b.xml；assets/grippers/dex_hand/dex_hand.xml；source/robots/builder.py；configs/robot_profiles/rm75b_dex_hand.json|仅静态零位，mj_forward后渲染，无mj_step；非实机照片或任务成功实验|
|712感知区域|source/sensors/tactile/dex_hand.py:294–343及配置，编译模型实际解析712个touch site|颜色为可视化标记；box有体积，不是独立碰撞Taxel|
|算法2-1曲面阵列|dex_hand.py::augment_spec、_taxel_frame、_orient_frame_outward、_add_taxel及fitting/surfaces.py|现有构建流程归纳，不新增采样算法|
|算法3-1候选与示教|graspqp_adapter.py、dexevolve.py、executor.py；tools/grasping/batch_grasp_edit.py；verification/strict_replay.py|跨模块流程归纳，任务专属验证不合并成同一通用调用|
|算法5-1 GMM伪配对|outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/expand.py:37–90及当前5.3|固定网格责任度、Top-5、至多3候选、执行集合argmin后重检查，未修复坐标合同或改数据|
|算法5-2 Translator训练与自由读出|同C3/code/models.py、inference.py、当前5.5损失合同|冻结Decoder仍允许对输入梯度；部署用预测Mask和尺度恢复均值|
|算法6-1共享场积分|outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/code/model.py::integrate,forward|各积分步复用参数，单次GRU沿4帧，教师不参与部署|
|19处流程图|对应当前审核正文图块及上述模型/数据代码|仅替换视觉表达，不引入实验指标|
|历史案例画面|stage_b2/candidate_frames.json、video_metadata.json及原agentview.mp4|直接源视频帧；不把历史Strict帧当录像帧，失败无视频|

两个视频及截图的逐项Hash已在Stage B2登记。本轮原视频不重新生成。
