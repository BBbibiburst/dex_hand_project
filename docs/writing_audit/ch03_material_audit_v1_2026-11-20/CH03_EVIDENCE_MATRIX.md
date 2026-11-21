# 第三章论点—证据矩阵

函数行号和完整SHA256由source_index.json给出；下列L2只对所列函数/字段有效，不代表重放或全量验收。实验路线编号独立，不与触觉迁移P编号混用。

|ID|论点|代码|函数|报告/结果|范围|等级|
|---|---|---|---|---|---|---|
|E01|候选合同与六驱动表示|source/grasping/contracts.py|GraspCandidate|—|指定shape、旋转、fraction检查|L2函数|
|E02|官方指标适配，不是原创GraspQP|source/grasping/graspqp_adapter.py|refine_candidates_with_graspqp|docs/PIPELINE.md|官方metric及本地正则目标|L2函数|
|E03|进化变异/交叉与仿真评价|source/grasping/dexevolve.py|mutate_candidate;crossover_candidates;_evaluate_batch|docs/PIPELINE.md|需对应历史档案；非同预算消融|L2指定函数|
|E04|局部腕部编辑及可达排序|source/rl/grasp_edit/templates.py|local_wrist_lattice;build_grasp_edit_templates|outputs/dex_hand_top100_v2/summary.json|当前函数与历史路由清单分开|L2实现+L2清单重计|
|E05|预抓取和抓取两个点IK排序|source/grasping/executor.py|candidate_world_pose;rank_candidates_for_scene|—|两目标、加权残差、状态恢复；非全轨迹碰撞保证|L2函数|
|E06|实际分段连续执行与失败信息|source/grasping/executor.py|_run_pose_segment;execute_grasp|docs/progress/2026-10-14_dp_p8_report.md|位置插值/SLERP/smooth选项、阶段错误|L2指定函数|
|E07|普通回放与FINAL Strict不同|source/verification/strict_replay.py|strict_replay_manifest|outputs/five_task_p32_b_2026-11-03/lift_sugar_strict.json|FINAL布尔判据使用尾段穿透|L2函数+L2历史结果|
|E08|普通Replay尾部任务成功比例|source/grasp_pipeline/replay.py|replay_grasp_trajectory|—|不等同Strict稳定性门槛|L2函数|
|E09|多任务前缀与专属后缀|source/grasping/task_transfer.py|execute_pick_place_transfer|docs/progress/2026-11-03_p32_b_online_recorder.md|前缀复用/重新执行两路径不能混同|L2指定片段+L1报告|
|E10|正式导出接口先Strict筛选|apps/collect_generated_lerobot.py|_export_one|thesis/ROBOTICS_EVIDENCE.md R-06|allow_unverified诊断开关；行数审计留第四章|L2指定函数|
|E11|记录合同动作语义不同|source/data/production_recorder.py|validate_online_arrays|docs/progress/2026-11-03_p32_b_online_recorder.md|position_action与ik_action均13维但不同语义|L2指定函数+L1历史报告|

## 代码位置与Hash

- `source/grasping/contracts.py`；SHA256 `4b3397dfa21eebfd9a8d0ceb5bfbd21b9261452f1b33a51b96fc8bbd909b6d3c`；定义索引：_array:18–22, GraspCandidate:26–106, DemonstrationEpisode:110–190, to_dict:75–89, from_dict:92–106, validate:134–142, save:144–167, load:170–190。
- `source/grasping/catalog.py`；SHA256 `1f8aca83620988d6e9eba94b8120581d75ef418b1079e56bb19dd649b88cb23d`；定义索引：ObjectGeometry:21–37, resolve_object_mesh:40–56, resolve_object_collision_mesh:59–76, load_object_geometry:79–176, table_z:36–37。
- `source/grasping/hand_surrogate.py`；SHA256 `b207c53b595338f62403192f6a6e1b1a174f280521c3ff09d1d68bc6615e1bed`；定义索引：_farthest_indices:36–45, _mesh_points:48–55, _transform_geom_points:58–64, _thumb_powers:67–80, DexHandSurrogate:84–214, calibrate_dex_hand_surrogate:217–348, load_or_calibrate_surrogate:351–374, points_per_digit:96–97, surface_point_count:100–101, contact_indices:104–111, evaluate_numpy:113–134, evaluate_torch:136–169, save:171–192, load:195–214, sample:258–276。
- `source/grasping/graspqp_adapter.py`；SHA256 `c1c8a9c57d4779cbb732f9f4ed67ae65c900e8b806397d7abeb6574a2865c1f1`；定义索引：GraspQPConfig:21–53, graspqp_available:56–61, refine_candidates_with_graspqp:64–217, validate:35–53。
- `source/grasping/dexevolve.py`；SHA256 `f75c5e27a1cede8aebe14af645ce3ceeb60c5599f0d7e7d19cad520dfd8b3af0`；定义索引：DexEvolveConfig:25–73, EvolvedIndividual:77–85, candidate_embedding:88–95, candidate_distance:98–100, mutate_candidate:103–122, crossover_candidates:125–135, _materialize:138–177, episode_fitness:180–204, disturbance_lifetime:207–248, _evaluate:251–276, _evaluate_batch:279–354, _density:357–364, _insert:367–378, evolve_candidates:381–459, evolve_candidate:462–463, validate:59–73, success:84–85。
- `source/grasping/dexevolve_mjwarp.py`；SHA256 `0912f7dde1f75cda8bb176bd7537f8fcf71ce6d6f3532b3bb5d4b19d967646fe`；定义索引：MjWarpLifetimeConfig:19–27, MjWarpLifetimeEvaluator:30–189, _opposed_contact:88–110, _sync_to_warp:112–114, _sync_to_torch:116–117, _closed_frame:120–125, evaluate:127–176, metrics:178–186, close:188–189。
- `source/grasping/budget.py`；SHA256 `b1fee79bac43d7ea20814b578ac0629258e6fe09588bc4c46316c6989ecc20b5`；定义索引：GenerationBudget:9–20, dexevolve_evaluations:19–20。
- `source/grasping/executor.py`；SHA256 `2a0520f7010d7cb2d22fd469bab588c4ae2da8b505ba3bac22f8978644132baa`；定义索引：ExecutionConfig:36–88, ReachabilityResult:92–96, grasp_hand_targets:99–128, quaternion_wxyz_to_matrix:131–133, hand_attach_rotation:136–142, candidate_world_pose:145–158, rank_candidates_for_scene:161–244, actuator_targets_from_fractions:247–254, _slerp_wxyz:257–270, _orientation_error:273–275, _Recorder:278–355, _palm_object_contact:358–372, _contact_digit:375–399, _robot_object_contact_summary:402–430, _step:433–437, _run_pose_segment:440–473, _target_error:476–484, execute_grasp:487–741, maximum_steps:58–68, validate:70–88, append:300–337, arrays:339–355。
- `source/rl/grasp_edit/templates.py`；SHA256 `9e913a60b42040a9f4e3e461f084cedb7afcf99a6f22536ab9295640ae4e7309`；定义索引：GraspEditTemplate:37–52, _slug:55–56, _full_episode:59–69, discover_grasp_attempts:72–121, _axis_rotation:124–133, _rotation_delta:136–143, edit_candidate_pose:146–180, local_wrist_lattice:183–238, _code:241–244, _label:247–266, _write_index:269–294, build_grasp_edit_templates:297–583, wrist_edit:51–52, key:100–118, compile_key:395–427。
- `source/control/arm.py`；SHA256 `f511f6f3ed5ec4bed81e791c3dbf1bd7059256a8a12fff127cc5187cc00793b3`；定义索引：ArmPositionIkController:23–546, action_size:100–101, position_action_size:104–105, action_space:108–109, actuator_ids:112–115, qpos_addrs:118–121, ctrl_low:124–127, ctrl_high:130–133, site_id:136–139, arm_dof_addrs:142–145, ik_data:148–151, set_timestep:153–156, set_control_mode:158–164, bind:166–214, reset:216–252, apply_action:254–262, apply_position_action:264–277, apply_ik_action:279–309, current_action:311–315, current_ik_action:317–321, ik_action_layout:323–324, _solve_ik:326–422, _adaptive_damping:424–430, _smooth_joint_target:432–454, _slerp:457–471, _coerce_position_action:473–486, _bound_action_space:488–515, _unbound_action_space:517–546。
- `source/verification/strict_replay.py`；SHA256 `25edfdfbd2c10a5fae26b722fb06c6f105e16569deae12fe7352fa7e43fcb830`；定义索引：StrictReplayResult:49–70, _manifest_path:73–77, _payload:80–81, _signature:84–93, _contact_sets:96–128, _contact_snapshot:131–206, load_replay_controls:209–229, _resolved_settings:232–283, strict_replay_manifest:286–545。
- `source/verification/profiles.py`；SHA256 `e50fe451e25a4bb84bfc6acb2d9f93de694282776cfeae644fe1ed7f9bcfb4ab`；定义索引：verification_status:11–16。
- `source/grasp_pipeline/replay.py`；SHA256 `ad79c14f924ed5196aac1e79f3329a3c9ad191c6a8328c9ab9370d31f2c716cf`；定义索引：ReplayResult:16–20, replay_grasp_trajectory:23–72。
- `source/grasp_pipeline/reference.py`；SHA256 `52ffb63b7eab0d37e72faba43e41a2238b8e7e7b8cc5f4944600d79c60e41a86`；定义索引：EpisodeRecord:16–52, ReferenceTrajectory:56–196, resolve_reference_manifest:199–222, load_reference:225–238, load:26–52, horizon:109–110, action_dim:113–114, hand_action_size:117–118, hand_slice:121–122, from_episode:125–196。
- `source/grasp_pipeline/trajectory.py`；SHA256 `c82d554d55405b9e8a8cb3acaae34aa39656ba4a130963abc466de90ffb4b7b0`；定义索引：GraspTrajectory:16–97, validate:29–43, save:45–73, load:76–97。
- `tools/grasping/batch_grasp_edit.py`；SHA256 `f874d1a63a93e5377b197c62e8756cfbe76a2c6ad24a7b32d65fe3cb75da3d18`；定义索引：_is_noncacheable_interruption:59–74, ChildResult:128–131, LatticePreflight:135–144, ObjectWorkItem:148–156, _utc_now:164–165, _slug:168–169, _format_duration:172–180, _gpu_ids:183–206, _gpu_resource_rows:209–239, _estimated_worker_memory_mb:242–245, _resource_plan:248–350, _runtime_estimate:353–368, _wall_clock_estimate:371–383, _init_object_worker:386–399, _worker_gpu_identity:402–405, _gpu_phase:409–423, _atomic_json:426–430, _atomic_csv:433–441, _repo_root:444–453, _prepare_shared_surrogate:456–466, _validate_runtime_dependencies:469–482, _sha256:485–486, _source_hashes:489–510, _dataset_ids:513–522, _selection_ids:525–537, _episode_lifts:540–551, _discover_grasp:554–557, _grasp_summary:560–585, _run_child:588–620, _ensure_grasp:623–676, _parse_lattice:679–684, _parse_lattice_precheck:687–699, _preflight_lattice:702–768, _recovery_args:771–778, _apply_preflight:781–791, _parse_update_history:794–828, _adaptive_decision:831–875, _adaptive_train:878–1002, _read_json:1005–1011, _checkpoint_update:1014–1024, _classify_training:1027–1128, _empty_row:1131–1163, _save_object_result:1166–1172, _load_object_result:1175–1180, _write_summary:1183–1265, _format_progress:1268–1300, _run_object_item:1303–1427, build_parser:1430–1576, main:1579–1909, positive_or_auto:259–269。
- `tools/grasp_generation/graspqp_evolve.py`；SHA256 `70b9332a729d72ad2527e1b8b8fa4a6e496d14abc65ae923a679be7ca1085042`；定义索引：build_parser:35–55, _write:58–62, evolution_execution_config:65–67, main:70–351, finish_phase:75–80。
- `source/grasping/task_transfer.py`；SHA256 `ae176b54fac085c5fcab8a9614527141fe3b8da9f4a27363349b5f671b9c95af`；定义索引：PickPlaceTransferConfig:32–99, _select_transport_hand:102–189, _combine_arrays:192–210, _hand_collision_min_z:213–231, _safe_descent_distance:234–251, _object_touches_table:254–266, _run_adaptive_release:269–298, execute_pick_place_transfer:301–801, validate:59–86, appended_steps:89–99。
- `source/grasping/stack_transfer.py`；SHA256 `ea508777ae8936b4f0fcd65c3ae91cfb2bdf2b9ec5d57a6432c980776cf08e75`；定义索引：_StackTransferPlacement:28–47, StackTransferConfig:51–137, _servo_grasped_object:140–199, _best_lift_transfer_frame:202–229, _object_downward_extent:232–270, _restore_lift_terminal_into_stack:273–296, execute_stack_transfer:299–604, sample:31–47, validate:81–123, maximum_steps:126–137。
- `source/grasping/push_transfer.py`；SHA256 `16707ff4d2e688b6e46dd109547cf8ba830b67548e2962e49675fd3791139f22`；定义索引：PushTransferConfig:24–99, _palm_geometry_in_ee_frame:102–134, _palm_facing_rotation:137–159, _four_finger_geometry_in_ee_frame:162–212, _four_finger_tactile_indices:215–226, _four_finger_object_contact:229–241, _push_advance:244–251, _hand_table_contact_frames:254–278, execute_push_transfer:281–615, validate:54–99。
- `source/grasping/nut_transfer.py`；SHA256 `dad11466d10fd8e49a6c91dc6655748a2211d76eff6349735344866626aaa95f`；定义索引：NutTransferConfig:36–93, _servo_bolt_orientation:96–148, _servo_bolt_yaw:151–195, execute_nut_transfer:198–500, validate:60–93。
- `apps/collect_generated_lerobot.py`；SHA256 `4cc5607d4e40b1c3a921ddf7765e2ce7bcc48cd840b720f5e356c77cbf75339e`；定义索引：build_parser:19–53, _manifest:56–60, discover_trajectory_manifests:63–73, recorded_pre_action_indices:76–103, _export_one:106–237, run:240–255, main:258–259。
- `apps/run_p32_online_contract.py`；SHA256 `c8765d1a937a4c094c2f0b7adcc9962efde469e27e52064efe400f946811aaec`；定义索引：run:42–237, main:240–256, factory:57–81, create:58–80, config:114–120, planned_pick:126–139, suppress_probe:92–98。
- `source/data/production_recorder.py`；SHA256 `9a1e4f53e0f377f81ed91f9ce75b93760ad563d84a52c57ab57d2865294f6e0e`；定义索引：validate_stage_encoding:21–43, attach_stage_encoding:46–53, validate_online_arrays:56–96, OnlineEnvironment:99–223, recorder_type:226–250, elapsed_steps:139–140, elapsed_steps:143–144, _before:146–184, close:186–190, _after:192–199, step:201–206, step_physics:208–214, reset:216–217, prepare_recorded_control:219–223, OnlineRecorder:227–248, read_once:127–131, append:232–243, arrays:245–248。

## 实验登记与比较条件

|实验|时间|对象/组与初态|指标|代码/配置|测试独立性|原始结果|等级|
|---|---|---|---|---|---|---|---|
|历史Top100分级流水线|时间UNKNOWN；summary有源Hash|100个object_id；Root UNKNOWN|98 lattice、2 RL状态|18项代码Hash，12同/5异/1缺|无已认证共同独立测试集|outputs/dex_hand_top100_v2/summary.json|L2条目重计；成功物理含义仍按历史定义|
|P1扰动生成|2026-10-11目录记录|90尝试；按by_object区分，Root认证UNKNOWN|46接受、18Strict不稳定、26Lift失败|精确运行代码版本需当时档案|不是新盲测；与后续Sugar10重复|outputs/dp_p1_2026-10-11/generation_formal_summary.json|L1汇总|
|P8 Source A复用P1|2026-10-14审计，源为P1|10 seed/10不同NPZ；同物体模板小范围扰动|10历史FINAL_VERIFIED，hash登记一致|单份manifest/Strict配置，不等于当前全部源代码|历史训练/测试角色，不是本章新独立集|outputs/dp_p8_2026-10-14/dataset_audit/source_a.json|L2直接核对10源文件和Strict|
|PickPlace Top100|时间UNKNOWN|100对象记录，全部cached|100 SUCCESS状态|pipeline_version需随summary原件，不以当前代码回填|无可比新执行测试集|outputs/pick_place_top100_final/summary.json|L2条目重计|
|Stack task-valid v24|时间UNKNOWN|48筛选对象；排除52|43成功、5失败|v24声明，不认证当前模型全同|条件筛选域，非任意物体全集|outputs/stack_task_valid_v24/summary.json|L2条目重计|
|Push Top100|时间UNKNOWN|100对象，98 cached|100 SUCCESS状态|当前push代码与历史批次须分别说明|无新共同盲测|outputs/push_top100_final/summary.json|L2条目重计|
|Nut variants|时间UNKNOWN|100 variant；独立资产/Root UNKNOWN|100 SUCCESS声明，后续复现存在失败|旧几何与P32当前几何冲突|不是100独立物体测试|outputs/nut_assembly_top100_final/summary.json|L2条目重计；不认证物理泛化|
|P32五任务开发|2026-11-03|五个指定Episode；独立Root UNKNOWN|源结果及阶段/触觉已保存|online-v2等历史合同，详见第二章|开发案例；非新盲测|outputs/five_task_p32_b_2026-11-03/|L2指定NPZ和JSON，非L3|

所有历史JSON的Hash与实际读取路径见source_index.json；10条Sugar原件逐项Hash/Strict状态见historical_results_check.json。运行时间只按原summary读取，缓存行的0秒不作算法耗时。日期UNKNOWN不能以文件mtime替代实验时间。未开展L3。

## 历史原件SHA256

- `outputs/dex_hand_top100_v2/summary.json`：`e9a23b52f43d434d9634b25ae92ebc38fc96e2e81b84db5a130c0574bb9c77c3`。
- `outputs/pick_place_top100_final/summary.json`：`a70295cbaee6cc3a132d08d1fabf0c1080beb5c3914b660bf72f6e19a11a64d5`。
- `outputs/stack_task_valid_v24/summary.json`：`18ec12b71cb095e7dc40984364060ffb5e5d290a6fa6b67bceec964727f8bc40`。
- `outputs/push_top100_final/summary.json`：`8e5e829600d4c86e011ee7927990ed070072aea27da85e0095a8e5626c71b088`。
- `outputs/nut_assembly_top100_final/summary.json`：`42ca95049066483a19a58d77d116311478f082aa02f35c03bbdfc854ad5c4342`。
- `outputs/dp_p8_2026-10-14/dataset_audit/source_a.json`：`fba08216504499600b697507536b039332c0f9bc9eea2c1c92ecd1c2ed3a1c63`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p00/candidate_00/manifest.json`：`cd4e9ae42eac4c2fb0e1f279a096ce1d88c68aed027630198bf923944a2f1b85`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p00/candidate_00/strict_replay_final.json`：`65ab7496265b0c8871d4774072f80c0ad0f38e16a40bfab59ed7f3ad6302e8aa`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p01/candidate_00/manifest.json`：`5250edcc2fb07432906a1141647baf05a70acca926896e075776f93aeded63b8`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p01/candidate_00/strict_replay_final.json`：`1eecc6da894db80f9e1037e15b2b224ba07914f47687faab5f2415e2077e236e`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p02/candidate_00/manifest.json`：`0f7176d7762c9509ccd5e0d6a95a84b7296d6d35e2483d925894328877a8c200`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p02/candidate_00/strict_replay_final.json`：`228bba9ed99a8c5c003f183311fe6a6cecba149d18bb1a9ca07bcd5b535f60a8`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p03/candidate_00/manifest.json`：`87464967f2870acfcae48e0463ae9c8251002b74c71231b8adec844239d65c5d`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p03/candidate_00/strict_replay_final.json`：`375ec4f58aeffd2338f8417eb19fe5f8bec16a100fb9a844c9e1384160a6e354`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p04/candidate_00/manifest.json`：`757187045e18b02499bf3ae002895d69104e2eab80c0ccb05a703cb9e5d6a6cf`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p04/candidate_00/strict_replay_final.json`：`408eabb23aa03b50b875754dcb50d3d1b34d5660c71e930c3e92c518169a1893`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p05/candidate_00/manifest.json`：`97671f761063d1f6e9f6c721dcbc7b5f6a8aace67387455fa5702fffb8949f8f`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p05/candidate_00/strict_replay_final.json`：`68178c57f99c9409a1077b9fae8dea0d305c178a25b4bac5977fc14d59520761`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p06/candidate_00/manifest.json`：`5b368f421ac21ebd85bbdc9de6f060fc6ec5792dc073fb5247343ace94475a76`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p06/candidate_00/strict_replay_final.json`：`aac87c2c85674eb7dc0e230d2057b2d665ea48645b6d936ea966692de65f1d5f`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p07/candidate_00/manifest.json`：`48fff00742ed32cef0c0c5a363ab73e04072743cd3b72fd1114cd9ffaccc3fbd`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p07/candidate_00/strict_replay_final.json`：`a41abdcb5826169f6e7782a328b3a9748d224b418d61307e07b1245fa0b2b8de`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p08/candidate_00/manifest.json`：`03c03df3e8eb3fc608822327aa1a94d0d134c41b68bb9017dffce309f5b9d662`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p08/candidate_00/strict_replay_final.json`：`8e36c519912b4826d6c5679d300a361b894cb9415f0b5ebd6aff7f1b4cf4f8f6`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p09/candidate_00/manifest.json`：`55cc94a05691be53a2394a1883f2b78aeafc3c610133fd8bd00623c568cda404`。
- `outputs/dp_p1_2026-10-11/verified_formal_recovery_3x10/ycb_004_sugar_box/p09/candidate_00/strict_replay_final.json`：`fb8fd3c9e1d11ab9401d4d2e108e643a9591eec67a82f614a9933bda97fe9ecf`。
- `outputs/dp_p1_2026-10-11/generation_formal_summary.json`：`6ba85fb341b30cfc690a616c24189b878f0e6a3e50ca51772d61b089e7364540`。
- `outputs/five_task_p32_b_2026-11-03/lift_sugar_strict.json`：`00aaa91645953ed2c861bf8aa4afdc91089dfaa095fa085e3892f9b50aa08c8e`。
- `outputs/five_task_p32_b_2026-11-03/nut_original_handle_00/result.json`：`06dec461888b47ac7f0486ed882e17827b8192aba04e6bda37fbe2e28f924c74`。
- `outputs/five_task_p32_b_2026-11-03/nut_retain_lower_rgb_00/result.json`：`a1586fb8ef03158f4de581b50530a9c37d010266afd7f9e66b05019d27498657`。
