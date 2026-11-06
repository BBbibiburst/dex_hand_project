# 五任务Episode格式

当前在线合同为 `five-task-online-v4`，正式文件合同为 `five-task-episode-v4`。旧v2/v3可读取但不得补造缺失字段冒充v4。

每条登记数据位于 `episodes/<job_id>/`：

- `episode.json`：任务、资产、来源、字段语义、逐文件SHA-256、lineage/content hash和准入等级。
- `modalities.npz`：规范学习字段及回放辅助字段。
- `agentview.mp4`：原生1920×1080、20fps，逐帧完整解码验证PTS与数量。
- `source_episode/`：本次仿真原始记录、manifest及有界无损压缩编译模型。
- `independent_replay.json`：同一源控制序列的独立物理回放证据。
- `strict_final.json`：Lift所需原FINAL Strict证据。
- `job_payload.json`：正式数据的不可变任务定义。

## 规范数组

| 字段 | 类型/形状 | 语义 |
|---|---|---|
| state | float32 `[T,39]` | 执行动作前的受控执行器观测 |
| action | float32 `[T,13]` | 本控制步实际施加的位置命令，arm7后hand6 |
| tactile | float32 `[T,712]` | 本次执行的processed传感器读数，非负且finite |
| object_state | float32 `[T,N,7]` | 世界坐标位置xyz和WXYZ四元数 |
| timestamp | float64 `[T]` | 执行动作前的仿真秒 |
| next_timestamp | float64 `[T]` | 当前动作执行完成的仿真秒 |
| stage | int16 `[T]` | 任务记录器阶段编号；结合源任务阶段定义读取 |
| success | bool `[T]` | 控制步后的任务判据结果，最终结果另在metadata中保存 |
| frame_index | int64 `[T]` | 从0开始、连续递增的RGB/模态对应帧 |
| integration_state | float64 `[T,D]` | `mjSTATE_INTEGRATION`执行前完整状态，D由编译模型决定 |
| control_exact | float64 `[T,model.nu]` | 完整MuJoCo ctrl，保留实际执行精度 |

39D state不是简单截取qpos：前13维为受控执行器的 `actuator_length`，中13维为 `actuator_velocity`，后13维为原位置命令；顺序由controller arm后hand的执行器ID决定。单位由模型传动/执行器定义，不应把所有分量一律解释成关节角或米。

原始源文件中的 `action`仍是旧执行器使用的笛卡尔IK指令；学习读取规范 `modalities.npz::action`，不要混淆。规范动作等于源中的 `position_action`，源 `sensor_tactile`对应规范 `tactile`。

## 时序与对象

物理1000Hz、控制20Hz，每步推进50个物理步。时间严格单调，控制间隔0.05秒；最后一步也计入有效时长：`next_timestamp[-1] - timestamp[0]`。观测与RGB在控制施加前采集；状态末13维应与前一步实际命令一致。不能把post-action状态与同一动作当成pre-action监督对。

Lift/Push记录被操作物体；PickPlace记录物体和目标site；Stack记录两个动态物体；NutAssembly记录Bolt与固定Nut。N可以不同。`object_names`及编译模型provenance给出顺序、body/site身份和动态/静态角色。目标site不能计为额外独立操作资产。

712D按原传感器patch顺序保存，patch offset/shape以源manifest中sensor metadata为准。无接触零值合法，缺传感器不得填零。成功交互必须有实测非零响应。processed读数不是未经标定即可声称的SI力值。

## 验收与计时

lineage绑定实际源记录及资产；content绑定规范观测、相对时间和完整解码RGB。回放辅助的高精度状态不制造新的内容独立性。所有文件hash、源数组一致性、视频、对象、回放和配额由程序重新计算，不以保存的“通过”JSON替代实际检查。

`development_verified_not_formal_production`只表示开发验证；正式验收要求 `production_verified`及全部目标满足。profile属于非语义诊断数据，不改变上述数组合同，跨层和异步耗时不能直接相加。
# 阶段编号解释

新采集记录在源manifest的 `metadata.online_contract.stage_encoding` 声明 `executor-stage-codes-v1`，完整表保存在 `metadata.stage_codes`：0 settle、1 transit、2 pregrasp、3 approach、4 close、5 hold、6 lift、7 verify、8 transport、9 descend、10 release、11 retreat、12 task_verify、13 clear。实际使用阶段不必连续，也不要求14项全部出现在同一任务。

新版本检查stage为一维整数数组、长度与timestamp一致、所有编号均有定义且语义符合版本。历史前缀的Lift阶段表可能只含0–7，因此新记录必须用当前执行器完整表替代。旧记录未声明版本时独立报告为未检查，不能推定阶段表完整；不得修改旧文件补标签。
