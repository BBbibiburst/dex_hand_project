# 第四章论点—证据矩阵

所有代码和数据路径相对项目根目录。L2限定到下列对象，不传播为全库验收；无L3。

|节|论点|实现路径|函数或字段|报告/原件|等级|不能外推|
|---|---|---|---|---|---|---|
|4.1|39D可观测执行器状态与13D低层动作|source/imitation/observation.py|real_observable_state;validate_position_action_contract|三组LeRobot Parquet与五v4源|L2指定字段/函数|不等于39关节/全qpos/实机认证|
|4.1|训练特征命名与写入|source/data/lerobot_schema.py|STATE_KEY;ACTION_KEY;TACTILE_KEY|meta/info.json和Parquet schema|L2|格式版本不自动证明语义|
|4.2|执行前采集/执行后关联|source/data/production_recorder.py|OnlineEnvironment._before/_after;recorder_type|11条指定NPZ|L2代码与数组|缺失字段记UNKNOWN，不用现合同回填旧记录|
|4.2|触觉缓存不重复推进滤波|source/data/production_recorder.py|OnlineEnvironment.__init__::read_once|P32修复报告|L2代码+L1报告|不能证明所有历史版本均使用缓存|
|4.2|RGB降频与控制索引|source/data/production_capture.py|capture_contract;recorded_rgb_contract|五v4包及video验证记录|L2代码/L1历史视频验收|未本轮解码全部视频|
|4.3|旧post-action到pre-action映射|apps/collect_generated_lerobot.py|recorded_pre_action_indices;_export_one|P2修复导出及源ctrl/tactile|L2|仅重算指定字段，不认证所有图像与state|
|4.3|P31移位不等于动作语义转换|source/data/five_task_dataset.py|write_causal_source;validate_source_episode|版本化metadata|L2代码|没有把原IK action自动变Joint13|
|4.3|原始源到规范modalities|source/data/production_artifacts.py|FIELDS;canonical_arrays|五v4 modalities与source数组|L2逐元素比较|NPZ v4包不自动是LeRobot|
|4.4|数据合同与stage语义|source/data/production_recorder.py|validate_online_arrays;validate_stage_encoding|11条NPZ；旧Stack缺pre_*|L2|字段/阶段检查是否开启需版本证据|
|4.4|队列状态、lease及去重|source/data/production_queue.py|register;transaction;recover_expired|代码定义|L2指定机制|未打开或认证服务器数据库|
|4.4|来源、语义内容及文件Hash|source/data/production_artifacts.py|content_hash;verify_episode;package_episode|五包episode.json与全部登记文件|L2文件Hash/代码|未本轮重算解码RGB参与的语义Hash|
|4.4|视频异常传播与PTS检查|source/data/production_video.py|OnlineVideo;verify_video|既有verification视频报告|L2代码/L1历史执行|像素内容与每帧时间未在本轮重验|
|4.5|旧LeRobot版本/数量/标签|apps/collect_generated_lerobot.py|版本区别详见代码与报告|3目录完整数值Parquet、tasks.parquet|L2统计/L1历史错误归因|不同版本不能直接合并为独立训练样本|
|4.5|完整状态/float64控制修复对照|source/data/production_recorder.py|pre_integration_state;applied_ctrl64|P38 comparison.json、380帧源NPZ|L2数组/L1回放记录|未分离双改动，非本轮L3|

## 源码位置与Hash

- `source/imitation/observation.py`；SHA256 `997c93adaee402faf71c8a64a07cc64ab9989c8cf1b50d6cba4f036c74268194`。函数/类位置：actuator_ids:20–28, real_observable_state:31–41, state_layout:44–50, action_layout:53–58, validate_position_action_contract:61–70。
- `source/data/lerobot_schema.py`；SHA256 `82192b828542a9b4980d7efdb53e5827026af6d38d54a83a6dbcf9ef99affbb9`。函数/类位置：。
- `source/data/lerobot_recorder.py`；SHA256 `cf3d184d7d57f202ed962399e16e6913a9c65002e287b82d853b0b558ed8ac0c`。函数/类位置：LeRobotEpisodeRecorder:22–194, _create_dataset:197–199, _archive_incomplete_dataset:202–211, _validate_dataset_compatibility:214–232, __init__:23–140, add_frame:142–177, save_episode:179–182, clear_episode:184–187, finalize:189–194。
- `apps/collect_generated_lerobot.py`；SHA256 `4cc5607d4e40b1c3a921ddf7765e2ce7bcc48cd840b720f5e356c77cbf75339e`。函数/类位置：build_parser:19–53, _manifest:56–60, discover_trajectory_manifests:63–73, recorded_pre_action_indices:76–103, _export_one:106–237, run:240–255, main:258–259。
- `source/data/production_recorder.py`；SHA256 `9a1e4f53e0f377f81ed91f9ce75b93760ad563d84a52c57ab57d2865294f6e0e`。函数/类位置：validate_stage_encoding:21–43, attach_stage_encoding:46–53, validate_online_arrays:56–96, OnlineEnvironment:99–223, recorder_type:226–250, __init__:102–133, __getattr__:135–136, elapsed_steps:139–140, elapsed_steps:143–144, _before:146–184, close:186–190, _after:192–199, step:201–206, step_physics:208–214, reset:216–217, prepare_recorded_control:219–223, OnlineRecorder:227–248, read_once:127–131, __init__:228–230, append:232–243, arrays:245–248。
- `source/data/production_queue.py`；SHA256 `3aa9b5fbc4f0631d0891d6309d0d3b3ccb2c1345d2c67d2b26abd81b30b37ebe`。函数/类位置：process_identity:17–26, ProductionQueue:29–474, __init__:30–122, _efficiency:125–130, _scheduled:133–140, _terminal:143–153, connect:156–164, transaction:167–175, event:178–180, enqueue:182–198, recover:200–227, claim:229–247, attach_worker:249–254, owned:257–263, heartbeat:265–291, validating:293–297, fail:299–313, register:315–347, record_attempt_metrics:349–375, scheduling_efficiency:377–381, scheduling_stats:383–388, observe_progress:390–407, status:409–430, quota_status:432–474。
- `source/data/production_video.py`；SHA256 `981b6239620531db6db5bbc58fec28616a5ea1ae5c5a6eedd3e93f1504db371f`。函数/类位置：verify_video:19–41, OnlineVideo:44–154, __init__:45–67, _consume:69–78, _check:80–89, _put:91–99, submit:101–107, should_capture:110–111, record_control:113–121, abort:123–129, finish:131–154。
- `source/data/production_capture.py`；SHA256 `a4e2eccec7c0e6657e629255d221912833d19ab3bbd1e81b9a292d5fbebced80`。函数/类位置：capture_contract:8–22, capture_from_config:25–33, recorded_rgb_contract:36–47。
- `source/data/production_artifacts.py`；SHA256 `a41c464af480a52ad0f5fffb96c3572807f230a20f59d31e9a1f2c4d1daa2fda`。函数/类位置：model_object_identities:28–38, copy_source_contents:41–59, sync_directory:62–67, write_json:70–74, canonical_arrays:77–82, content_hash:85–99, _DecodedVideoProof:102–119, verify_episode:122–238, package_episode:241–321, reconcile_episode:324–340, register_artifact:343–374, isolate_duplicate:377–386, __init__:105–112, checked_result:114–119。
- `source/data/production_inventory.py`；SHA256 `69be00386375f2bfb6df13be7a84a260f69e0e04d960648a012c704dd8dcc8c9`。函数/类位置：verify_inventory_row:11–27, audit_inventory:30–58。
- `source/data/production_duration.py`；SHA256 `b23af322eb0dab70380ee39775d747eb731baa33e57ca5ed45cb73d86d7ced5d`。函数/类位置：candidate_duration:10–34。
- `source/data/five_task_dataset.py`；SHA256 `30ba6c1b1acf7cf1dcf7b0da1bd00773c6fa6e2b0001110f8e07325507b9e6e8`。函数/类位置：classify_failure:25–34, sha256_file:37–42, canonical_hash:45–47, tactile_status:50–59, validate_source_episode:62–96, write_causal_source:99–136, validate_exported_arrays:139–158, EpisodeIndex:161–287, __init__:162–184, close:186–187, find_lineage:189–194, find_content:196–201, insert:203–222, inventory:224–273, write_inventory:275–287。
- `source/grasping/executor.py`；SHA256 `2a0520f7010d7cb2d22fd469bab588c4ae2da8b505ba3bac22f8978644132baa`。函数/类位置：ExecutionConfig:36–88, ReachabilityResult:92–96, grasp_hand_targets:99–128, quaternion_wxyz_to_matrix:131–133, hand_attach_rotation:136–142, candidate_world_pose:145–158, rank_candidates_for_scene:161–244, actuator_targets_from_fractions:247–254, _slerp_wxyz:257–270, _orientation_error:273–275, _Recorder:278–355, _palm_object_contact:358–372, _contact_digit:375–399, _robot_object_contact_summary:402–430, _step:433–437, _run_pose_segment:440–473, _target_error:476–484, execute_grasp:487–741, maximum_steps:58–68, validate:70–88, __init__:279–298, append:300–337, arrays:339–355。
- `apps/export_five_task_dataset.py`；SHA256 `e62e37fe20446b1aea8e97395e917381c6c85f549946f444c3b3de0268abfa3f`。函数/类位置：_git_head:30–31, _directory_bytes:34–35, _fsync_tree:38–47, _asset_record:50–62, build_parser:65–75, export:78–273, main:276–278。

## 直接数据与报告来源

- 三组LeRobot：outputs/dp_2026-10-11/lerobot_5、outputs/dp_p1_2026-10-11/lerobot_formal30、outputs/dp_p2_2026-10-11/lerobot_single_sugar_fixed。全部数值Parquet、info与tasks表均读取；dataset_checks.json保留逐Episode结果。
- 数值映射来源：outputs/dp_p2_2026-10-11/diagnostics/action_chain/action_chain_report.json给出源manifest；本轮重新对照原NPZ，而不只引用其中RMSE=0声明。
- 旧在线记录：outputs/five_task_p32_b_2026-11-03下五个指定rgb案例source_episode/episode.npz。不是全P32目录扫描，不能泛化缺陷比例。
- 新v4五包：outputs/five_task_p44_integration_2026-11-05/dataset/episodes下五个目录，全部对应episode.json及登记文件Hash已核验。
- 精度对照：outputs/five_task_p38_exact_replay_2026-11-03/comparison.json与所指380帧源NPZ。回放成绩是已有记录，不是重新执行。
- 反因果历史说明：docs/progress/2026-10-24_imitation_learning_roadmap_audit.md，P2合同错误及后续修复边界。
- 在线修复与版本：docs/progress/2026-11-03_p32_b_online_recorder.md、2026-11-03_p38_exact_replay_contract.md。
- 重复索引与降频边界：docs/fast_collection.md，仅作历史使用说明，不据其重复索引“1000小时”确认新增仿真时长。

实际读取文件的完整SHA256见integrity.json::audited_input_sha256。metadata字段名、dtype和shape、每条转移错误数量以及Hash匹配结果见dataset_checks.json。未读取服务器DB，也未将summary中的EXPORTED视作新生产验收。

## Real来源的额外登记

outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/data_contract.csv的Train/Validation两行已读，属于L1既有合同。引用Real/Natural角色及时间合同边界，不将其当成重新统计原始Real文件或来源同步认证。详情见dataset_checks.json::frozen_tactile_contract_metadata，Hash收录integrity.json。未打开Test/Holdout数组。
