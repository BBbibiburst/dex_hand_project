# 论点—原始证据矩阵

完整Hash如下；行号由AST提取。L2仅限列明范围，XML默认为语义解析而非运行时验证。

|小节|论点|文件|函数/字段/行号|SHA256|等级|
|---|---|---|---|---|---|
|2.1|模块构建与任务factory|source/robots/builder.py|build_robot|82815dd6580c69366fb7c580afc28b92a01925c00b7f2a87821ef0ab28357356|L2函数|
|2.1|五任务入口|source/envs/manipulation/factory.py|make_manipulation_env / make_*_env；make_manipulation_env:12–21|39472444756c64c6298b215b3066695d2c4b5846a78279808dd91981a88e2148|L2函数|
|2.2|arm7关节位置执行器|assets/robots/rm75b/rm75b.xml|worldbody/joint; actuator/position|366d90de2897e0a5ffd45f5ccdcbb1b728063be0310c9578070c400cae9616fc|L2 XML|
|2.2|hand12关节6腱；无equality/flex|assets/grippers/dex_hand/dex_hand.xml|worldbody/joint;tendon/fixed;actuator|74399f5110ed8880bd8c21795e8b3431a8068a5d90aa8f68aeb6a46bb5d6277c|L2 XML|
|2.2|附加安装旋转与末端site|source/robots/arms/rm75b.py|RM75B_ARM|a54231d7290689834acd633333aa1bf15e9fc0e450995aa6e216426b87918c56|L2配置代码|
|2.3|DLS、加权误差与零空间|source/control/arm.py|_solve_ik;_adaptive_damping;_smooth_joint_target；_solve_ik:326–422, _adaptive_damping:424–430, _smooth_joint_target:432–454|f511f6f3ed5ec4bed81e791c3dbf1bd7059256a8a12fff127cc5187cc00793b3|L2函数|
|2.3|手部目标与传动坐标|source/control/end_effectors.py|bind;apply_action;current_position；bind:81–114, apply_action:153–166, current_position:174–184, bind:248–256, apply_action:277–293|5f0e68dff7663dcb1a1b6cc7b834fc546d18ee222c61cd2e7687cb130339193a|L2函数|
|2.3|每周期物理子步及先动作后观测|source/envs/rl_env.py|step;step_physics；step:207–224, step_physics:226–229|c2dc63454ca94df8983e03ca0e65923fcd8a77cd6f33725736fef0027fbd9b97|L2函数|
|2.4|16patch/712行主序|source/sensors/tactile/fitting/layout.py|dex_hand_patch_layout；dex_hand_patch_layout:14–22|314935d11cbd45a62b2280b24f42708e3adfa4c3ff6f32213b52785886798a4d|L2函数|
|2.4|外壳+Bezier|source/sensors/tactile/fitting/segment.py|_finger_segment_regular_surface_grid_points；_finger_segment_regular_surface_grid_points:226–245|7141ae6c4d598140f69a75119568e3752d572ca563938f5185faddf816d81ffd|L2函数|
|2.4|非理想椭球实现|source/sensors/tactile/fitting/fingertip.py|_fingertip_regular_surface_grid_points；_fingertip_regular_surface_grid_points:311–352|d7c46b0ed88fb39acba17d5ddea529d8eedd058ee1749c62fbe81cd30bfcfa3b|L2函数|
|2.4|掌心PCA投影及重心插值|source/sensors/tactile/fitting/projection.py|_mesh_uv_grid_points_from_triangles；_mesh_uv_grid_points_from_triangles:70–170|b23fdaddcd1225a097d2ef32ff644ceae7bf960216f65fef5808cdbb72bec90f|L2函数|
|2.4|Bezier正则拟合|source/sensors/tactile/fitting/bezier.py|fit_bezier_surface;evaluate_bezier_surface；fit_bezier_surface:20–58, evaluate_bezier_surface:61–71|1f34cfee10743e9c207b3a0ccd7720cb4f422ca2c5c7e73f64f9b4ea86ef2377|L2函数|
|2.4/2.5|site构建、地址绑定、read_raw|source/sensors/tactile/dex_hand.py|augment_spec;bind;read_raw;_add_taxel；augment_spec:294–342, _add_taxel:345–358, bind:379–400, read_raw:420–424, _add_taxel:511–533|a107c535e39490d4b605c8568c884cb353652588a951a7a4ffe667272ffcee0c|L2函数|
|2.5|processed顺序及邻域实现|source/sensors/tactile/signal_processing.py|process;_neighbor_crosstalk;_convolve_axis_reflect；_convolve_axis_reflect:28–42, _neighbor_crosstalk:51–91, process:162–187|1252a12322e4e808be931531b3dfdd304ba0ce2564843a61720ed764e87f9a7a|L2函数|
|2.3/2.5|特定profile参数，不冒充所有实验参数|configs/robot_profiles/rm75b_dex_hand.json|control_dt;tactile_sensor_options|4e48460129c9d8a0a2bc6285c8d770bc42656b2095fcb35e60d70be07cd86399|L2文件|
|2.6|合同字段与时间间隔规则|source/data/production_recorder.py|validate_online_arrays；validate_online_arrays:56–96|9a1e4f53e0f377f81ed91f9ce75b93760ad563d84a52c57ab57d2865294f6e0e|L2函数|

## 指定历史数组

|文件|字段|SHA256|等级|
|---|---|---|---|
|outputs/five_task_p32_b_2026-11-03/lift_sugar_rgb_00/source_episode/episode.npz|sensor_tactile / timestamp / shape|5acd12c1b7f7ea396c2358e6972042c84c550503693fa9db2075de2de70de5eb|L2指定数组|
|outputs/five_task_p32_b_2026-11-03/pick_tomato_rgb_00/source_episode/episode.npz|sensor_tactile / timestamp / shape|52c5f208995f37914ce429cddf6076315ca439f6922217af89b71c1110f872bc|L2指定数组|
|outputs/five_task_p32_b_2026-11-03/stack_ring_rgb_01/source_episode/episode.npz|sensor_tactile / timestamp / shape|bda0eca559e1d9d2b2ccd2645c83f7e6ba9e1cc8901fa13c6b97fe62ca86f54c|L2指定数组|
|outputs/five_task_p32_b_2026-11-03/push_tomato_rgb_00/source_episode/episode.npz|sensor_tactile / timestamp / shape|bae59537a18f8804f8bd475e21834dc90cce57bf6e0b15f9c49e7b1eae8b47f2|L2指定数组|
|outputs/five_task_p32_b_2026-11-03/nut_retain_lower_rgb_00/source_episode/episode.npz|sensor_tactile / timestamp / shape|cf4a4ce8a9d9545f0853a7321b5069a498e5f30d4c2fdcaac37b5c9f3eced789|L2指定数组|

result/manifest路径及Hash见source_inventory.json；success/视频/Strict历史记录只作L1。未核验当前运行模型、CAD来源认证、真实Taxel电气映射，均UNKNOWN。
