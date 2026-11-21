# 第三章材料盘点与写作就绪性审计

日期：2026-11-20；阶段2，仅读取已有实现和产物，没有正文写作、代码修改、训练、物理重放或服务器任务。

## 决策摘要

**建议立即进入第三章正文写作，采用限定证据范围的初稿。** 预计现有材料可支撑约8000–8600 TeXcount的章节（篇幅规划判断，不是已写字数或自动统计结果）。3.2、3.4、3.5最充分；3.3完整碰撞/全路点预检查证据较弱，3.6缺少公平算法对照及指定14/127批次原件。完成一版合格初稿不需要新增实验；不把算法优越性或全物体Strict通过作为本章论点。

本轮阶段3尚未授权，当前交付不是正式正文。第二章实际已冻结为初稿，见thesis/releases/ch02_2026-11-20；任务书中“最后小修”属于状态描述差异，不影响本轮只读工作。

## 证据等级与主线

L0为仅定位；L1为报告/配置/汇总中的声明；L2为指定函数、原始文件或结果记录的直接核对；L3为重新运行验证。本轮未进行L3。对summary逐行重计可标L2文件统计，但其中的物理成功仍是历史声明，不升级为本轮重放。

研究主线是：对象几何和六驱动候选 → 几何优化与物理反馈细化 → 机械臂可达排序和局部腕姿态扩展 → 分段连续执行 → 独立Strict质量检查 → 任务专属示教后缀。各环节解决不同约束，不是同一指标逐级重复，也不是所有五任务共用一个完全相同的规划器。

## 3.1 任务定义与总体流程

GraspCandidate在source/grasping/contracts.py记录object_id、物体相对手部平移/旋转、六维actuator_fractions、接触点/法向/距离及指标。构造检查旋转正交、右手系、驱动fraction属于[0,1]；它是候选合同，不是执行成功证书。

executor.py::candidate_world_pose将物体相对候选转换为世界目标，并补偿手部安装旋转。execute_grasp构建Lift执行链，记录DemonstrationEpisode。Lift支撑搬移/堆叠的抓持前缀；Push主要生成接触推动而不要求先抓稳抬升；Nut采用抓持Bolt及固定Nut的任务专属逻辑，不应写成五任务均是同一Lift轨迹的直接复制。

可写候选、机器人目标、低层控制和任务结果四层定义。机械臂机构/触觉原理引用第二章，不重写；序列如何产生属于第三章，序列如何关联、去重、转换存储属于第四章。

## 3.2 多来源候选生成

|来源|实际职责与输入输出|项目地位|证据边界|
|---|---|---|---|
|GraspQP指标适配|物体点/法向、六驱动surrogate及种子；优化手平移、6D旋转参数、fraction logits，输出候选|进入graspqp_evolve入口，batch缺少源尝试时调用|复用官方GraspSpanMetricFactory；不称本人原创GraspQP或完整第三方系统复现|
|DexEvolve-style|候选群体，变异/交叉、接触修正、执行与扰动反馈，输出档案及优选Episode|进入相同入口，生成档案供后续腕姿态扩展|本项目适配实现；第三方子模块只读参考，不等同其Isaac版全部实验|
|Wrist Lattice|对源候选进行局部平移/旋转，实际RM75B IK排序，再有限执行生成模板|batch_grasp_edit的主要阶段|默认格点39，默认1cm/15度；这是搜索设置，不是独立Root数或成功数|
|MJWarp PPO|模板选择及六驱动编辑，作为条件性后续阶段|batch对未由lattice解决者调用；Top100登记2条RL_SUCCESS|可简述兜底，不扩成强化学习性能论文，不称2例证明优于格点|
|UltraDexGrasp|在已查source/tools/apps/configs和PIPELINE中未找到调用证据|UNKNOWN，不纳入本章主链|不能把UniDexGrasp、DexGraspNet或其他名称代替该模块|

GraspQP的目标除官方指标还含接触距离、穿透、桌面距离、平移漂移和驱动耦合项；优化变量与目标可据实际公式描述。hand_surrogate.py是六驱动到皮肤点轨迹的可微代理，schema和模型Hash需随版本说明。graspqp_adapter的旧closed-chain措辞不替代当前固定腱实现。

DexEvolve实际使用平移/旋转/驱动变异、位姿与手部组合交叉、密度或新颖性档案管理，并调用execute_grasp及可选MJWarp lifetime评价；不能把同一后端的lifetime、抓持成功与FINAL Strict混成同一个得分。

当前统一预算budget.py为64种子、150步GraspQP、执行12候选；进化种群24、每代12子代、16代，名义216次候选评价，档案Top6。它是当前默认预算，不代表所有历史样本实际运行这些预算。adapter构造默认steps=100与入口采用统一预算150并存，写作按调用层级区分。

没有已认证同对象/初态/预算/后端的多方法公平排行榜。Top100的分级路由统计不能替代GraspQP、DexEvolve和Lattice的独立消融。

## 3.3 机器人约束筛选

rank_candidates_for_scene对pregrasp与grasp两个目标连续求IK，恢复环境状态及控制器历史；排序score=20*最大位置残差+0.6*最大旋转残差+0.08*末端行程。预抓取沿候选手局部方向退让，不能忽略手安装变换。该静态排序临时放宽目标速度限制，与真正执行时的速度平滑不是同一配置。

templates.py按ExecutionConfig位置容差0.025m、方向容差0.22rad保留候选，再按编辑幅度和源执行跟踪误差排序并受模板/执行预算限制。默认local_wrist_lattice围绕源位姿作局部改变，不是将腕部绕物体整体旋转；可用示意图说明局部编辑与世界目标的关系。

**硬边界**：本轮已核验路径只证明两个目标的预检，不支持“Lift之前对整条轨迹所有路点完成IK和连续碰撞认证”。代码中的table几何惩罚、保守下降高度、接触监控和Strict桌面拒绝分属不同阶段，不构成通用连续碰撞规划器。全路点预检查完整产物及历史14物体失败类别清单仍UNKNOWN。

失败原因可从execute_grasp的pregrasp/approach残差、提前终止、hold无拇指对向接触、lift/verify失败追溯；batch还区分无抓取源、无可达模板、训练无进展等。没有候选级统一分母时不合并计算“各失败占比”。

## 3.4 连续轨迹与Strict

_run_pose_segment按当前末端状态到目标进行位置线性插值、四元数SLERP、手命令线性插值；smooth=True时采用3u²−2u³。控制器沿用前一目标配置和速度滤波，不能保证不存在分支跳变或宣称全局最优连续IK。第二章已给出DLS，本节仅说明初值、分段连接、限位及执行误差，不重新推导通用IK。

普通replay_grasp_trajectory加载低层controls，在Lift环境回放，按尾部最后成功且success_fraction≥0.8判断。Strict另检查末段最小/最终抬升、速度、角速度、皮肤接触/对向抓持比例、桌面接触和穿透。FINAL默认：抬升≥0.055m、速度≤0.10m/s、角速度≤0.10rad/s、尾段接触/抓持≥0.80、桌面比例≤0、尾段穿透≤0.003m；verify_tail=20、extra_hold=12。最大全程穿透被记录，但实际success判断使用tail_max_penetration，不能写成全程最大值硬门槛。

profiles.py仅将final profile布尔结果映射FINAL_VERIFIED/FINAL_REJECTED。生成脚本中的strict_execution变量名不是调用此最终验收器的同义词；必须读取result.profile和verification_status。

动作语义同样需区分：旧生成action含末端IK输入，Strict重放使用ctrl中的arm7+hand6位置命令。部分历史代码通过状态重置/终态恢复衔接任务，是否物理连续不能凭数组拼接推断。专家策略学习及闭环控制性能不是本章验收重点。

## 3.5 多任务专家示教

- Lift：settle/transit/pregrasp/approach/close/hold/lift/verify分段执行及误差检查。
- PickPlace：抓持前缀后追加clear/transport/descend/release/retreat/task_verify。task_transfer.py存在保存原Lift前缀和恢复终态的路径，亦有从reset回放开关；必须区分。P32记录修复了旧数组复用路径，不能回填全部旧样本。
- Stack：stack_transfer.py在源抓持状态上对目标和支撑物执行运动、下降/释放及稳定性评价；task-valid物体域与任意物体全集不同。
- Push：push_transfer.py有手掌/四指几何和物体接触检测，采用任务反馈推动；不以Lift通过为成功充分条件。
- NutAssembly：nut_transfer.py执行Bolt抓持、姿态伺服、平移和下降，当前声明版本与旧variant成功物理版本有差异。P32保留失败及retain控制对照，只支持局部条件解释。

各任务返回或保存DemonstrationEpisode及阶段字段，是共享数据合同而不是共享全部控制逻辑。LeRobot入口的Strict筛选和动作字段只作第三章输出接口介绍，导出行数、时间对齐/lineage/格式修复集中第四章。第三章可解释旧post-action Recorder与P32 pre-action记录存在差异，但不在这里完成全量同步审计。

## 3.6 已有结果与可写强度

|实验/原件|本轮实际检查|可写结论|限制|
|---|---|---|---|
|outputs/dex_hand_top100_v2/summary.json|100行；{'LATTICE_SUCCESS': 98, 'RL_SUCCESS': 2}|限定历史路由/任务记录统计|不等于新执行或统一Strict|
|outputs/pick_place_top100_final/summary.json|100行；{'SUCCESS': 100}|限定历史路由/任务记录统计|不等于新执行或统一Strict|
|outputs/stack_task_valid_v24/summary.json|48行；{'SUCCESS': 43, 'FAILED': 5}|限定历史路由/任务记录统计|不等于新执行或统一Strict|
|outputs/push_top100_final/summary.json|100行；{'SUCCESS': 100}|限定历史路由/任务记录统计|不等于新执行或统一Strict|
|outputs/nut_assembly_top100_final/summary.json|100行；{'SUCCESS': 100}|限定历史路由/任务记录统计|不等于新执行或统一Strict|

Top100：100个不同object_id，98 LATTICE_SUCCESS和2 RL_SUCCESS；可作分级路由案例分布，不称100/100 FINAL。PickPlace100条均cached；Push100条中98cached；Nut100条是variant而非100个独立object_id，cached字段缺失不能当成全新执行证明。Stack限定task-valid域48对象，43成功、5失败，并非Top100全覆盖成功。

已保存的source_hashes共18项：12项与当前一致（包括hand XML、adapter、surrogate、templates等），5项不同（含executor、catalog、task objects及batch），1项缺失（collision_decomposition.py）。详细清单见historical_results_check.json。不能把当前全部代码流程未经说明地回填成批量历史执行版本。summary中的runtime含缓存/复用，不能作为算法生成耗时的公平比较。

**主要Strict证据**：outputs/dp_p8_2026-10-14/dataset_audit/source_a.json登记的10条Sugar Box扰动示教，本轮直接检查10份manifest、episode.npz Hash及strict_replay_final.json：10个seed、10个不同NPZ Hash，均与登记Hash一致，10份历史Strict结果均FINAL_VERIFIED。源文件真实存在，数组qpos[477,26]、ctrl[477,13]。独立性限于已有记录中同物体/模板附近的扰动；初态位置两两距离0.707–2.828mm来自历史审计声明，不等于覆盖宽工作空间。历史root0–7用于训练、8–9用于测试的设置属于当时学习实验，不是本章新盲测或未暴露测试组。

**失败分类材料**：outputs/dp_p1_2026-10-11/generation_formal_summary.json登记90次扰动尝试、46接受，分类46成功/18 Strict不稳定/26 Lift失败；角速度和低抬升诊断可重叠。此处为L1历史汇总，本轮未全量重放或逐一认证全部90份原件，作为背景分解而非新统计验收。与10条Sugar样本有重叠，不相加为100条独立新示教。

**五任务及修复对照**：五条P32在线开发Episode的字段检查已在第二章完成，本章主要引用生成来源、阶段和任务控制区别，避免重复触觉合同表。Lift单例Strict final_lift=0.0608926m、tail_contact/grasp=1/1，原始strict JSON已核对。Nut原始抓持失效、降低抓点仍失败、降低抓点并retain抓持命令后单例成功有对应结果和报告；不能称跨seed稳定改进。P32任务结果均是历史执行状态，不是本轮L3。

14物体初期批次、127物体批次的确切原始清单及统一失败分类，本轮未定位，记UNKNOWN；不得用100物体、100个Nut variant或127D触觉实验替代。UltraDexGrasp同样不纳入主结果。

## 3.7 小结可写边界

总结候选、可达、实际任务和Strict质量四层关口及其分级流程；强调可执行示教不是几何分数的直接结果。承接第四章的数据记录质量，而不重复数据集规模、恢复数据库或触觉跨域实验。

## 能主张的贡献与不能主张的贡献

可主张：六驱动候选表达及官方指标适配；结合物体相对目标、腕部局部扩展及机器人执行约束的工程方法链；区分生成成功与严格示教合格的验证流程；任务专属后缀及失败保留的示教实现。属于系统方法与适配特色，未获得充分原创性/优越性认证。

不能主张：原创GraspQP/DexEvolve、完整复现所有第三方系统、统一全路点无碰撞保证、全部五任务统一Strict100%、127物体独立测试通过、实机操作成功或学习策略泛化已经解决。

## 最终就绪判断

不需要为了论文重做优化器或补一轮大规模仿真。采用实现链+历史批量路由+小范围Strict完整证据+任务失败对照，可以写出一版可信初稿；缺失14/127及公平对照应保留为材料边界。建议立即由ChatGPT提供阶段3整章任务书，优先明确标题与历史版本口径，不自行开始写作。
