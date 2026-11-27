# 全篇数据来源与流转总账

日期：2026-11-27。仓库根目录 `/home/zmy/dex_hand_project`。本轮为Stage A，只读静态审计及既有标量/登记复核，不产生L3，不修改论文或实验，不编译、不训练、不推理、不运行仿真，不commit/push。路径均相对仓库根目录；`文件:行`为可定位源码/CSV/JSON行，CSV行号含表头。PDF页码一律为文件物理页号，不能与印刷页码混用（本版本正文区相差8页）。

证据标记：VERIFIED=本轮直接核对所列代码/合同/既有结果；UNKNOWN=材料不足；UNVERIFIED=本轮未作所需深度认证；CONFLICT=来源之间实际冲突。直接读取历史结果文件仅证明其记录内容，不升级为本轮重新执行。L2只限所列范围。

## 一、两套数据体系必须分开

A体系回答“机器人示教如何生成、记录、转换并具备特定训练资格”；B体系回答“完整触觉分布如何以有限统计监督重建”。A可提供自然任务执行及规范检查基础，但LeRobot Parquet不是C3训练器直接读取的输入。

### A：仿真示教/数据工程

```
物体/候选/任务条件 → 专家执行 → source_episode NPZ + manifest
→ 在线pre-action/旧post-action版本识别
→ 语义和时间索引修复 → modalities/episode元数据
→ 按目标合同导出LeRobot / 另存视频和完整回放辅助状态
→ 数值、时序、Hash、来源和用途分层检查
```

|范围|直接来源与口径|已核对的结果|独立性及限制|
|---|---|---|---|
|旧LeRobot5|`outputs/dp_2026-10-11/lerobot_5/meta/info.json` total_episodes/frames|5 Episode/1550 Frame，train0:5|历史语义合同，不能直接作为新合规数据|
|旧formal30|`outputs/dp_p1_2026-10-11/lerobot_formal30/meta/info.json`|30/14310，train0:30|与P1源相关，不等于30独立Root|
|修复Sugar|`outputs/dp_p2_2026-10-11/lerobot_single_sugar_fixed/meta/info.json`|1/340，源477行，ctrl[137:477]；状态命令/触觉用[136:476]|单来源导出修复，不能三组相加称36独立Root|
|旧五任务在线|`outputs/five_task_p32_b_2026-11-03/`指定5例|447+638+376+605+599=2665行|含历史PickPlace373转移超容差、Stack缺pre_*；未全部修复|
|v4开发包|`outputs/five_task_p44_integration_2026-11-05/`对应导出，准确包路径见下述dataset_checks|447+648+638+613+381=2727行；5lineage/4asset_hash/5content|开发准入非生产验收；独立Root UNKNOWN，视频存在≠全解码同步认证|
|Sink资产Stack案例|`outputs/five_task_p38_exact_replay_2026-11-03/comparison.json:18–54`|380帧；旧误差1.6288973093，完整状态+float64为0|Sink是gso:3D_Dollhouse_Sink物体，任务是Stack，非独立Sink任务；同一来源不计新增小时|

原始字段语义、全部指定包路径和既有逐字段检查见`docs/writing_audit/ch04_material_audit_v1_2026-11-20/dataset_checks.json`。本轮回查info、comparison及已有检查原件，不重解码视频、不物理重放，不将L1历史回放或旧L2数组检查升级为本轮全库认证。

动作时序：pre o_i与当前u_i配对；历史post o_(i-1)只有无隐藏推进/恢复/缺口时可作o_i前态。P31移位与动作ctrl语义修复是不同过程（`ch04:58–89,129–140`）。prepare_recorded_control保存覆盖ctrl前状态，next_timestamp为控制结束时刻（`production_recorder.py:146–223`）。源Hash一致、转换误差0、任务成功、Strict通过与训练资格各自回答不同问题。

### B：触觉跨域

```
Real既有processed数组 → Train/Validation Session索引
Natural任务source NPZ → 按Root/阶段取四帧 → 数值唯一主窗口清单
Real Train → 冻结GMM索引偏移先验
字典探针物理响应 + 指定Real目标 → 最多3候选 → Paired Train/Validation
Real → Real VAE；Natural+Paired Sim → Sim VAE
Paired Sim后验均值 → Translator ← 配对Real后验均值及条件NLL
Natural Validation → 自由输出 → 冻结C0/分布与结构评价
```

## 二、当前划分的直接登记复核

|数据|窗口与组织|原始登记、实际调用|本轮判定|
|---|---|---|---|
|Real Train|5153，32 Session，4×712|P43 `real_train/index.json` indices/window_ids长度5153，sessions32；C3 `code/common.py:24–25`按source及indices读取|VERIFIED登记和索引长度；不是本轮重新认证原始驱动标定|
|Real Validation|780，4 Session|P43 `real_validation/index.json` indices780，sessions4；P44 `data_contract.csv:3`一致|VERIFIED，开发Validation非新盲测|
|Natural Train|4049，50 Root|P43 `natural_sim_train/windows_manifest.csv:2–4050`；C3 common.py:26–27读取windows.npz values/groups|本轮直接按清单计数和Root分组；不借全dataset_manifest所有行替代|
|Natural Validation|905，10 Root|P43 `natural_sim_validation/windows_manifest.csv:2–906`；P44 data_contract.csv:3一致|VERIFIED；被多轮开发评价，不能称未使用盲测|
|Paired Train|1200，每窗Sim/Real4×712|C3 `expansion_complete.json:2`；`code/pipeline_cpu.py:16–18`形状断言；`pair_expansion_manifest.csv` accepted/role|登记及代码VERIFIED；Root独立性未认证|
|Paired Validation|240，同形状|C3 expansion_complete.json:3；pipeline_cpu.py:17；实际映射验证行|登记及代码VERIFIED；不以240窗作为240独立初态|

P43根：`outputs/tactile_p43_latent_translator_integrated_2026-11-16/`；C3根：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`；P44根：`outputs/tactile_p44_integrated_iterative_adversarial_2026-11-18/`。

重要口径：P43大`dataset_manifest.csv`内Natural Train有6631条登记、Validation有1400条登记；实际模型主窗口为4049/905，对应各role的windows_manifest（primary_unique_numeric_window全True）。登记全集、候选窗口和模型使用窗口不可混用。此差异不证明论文计数错误，而说明需绑定实际读取清单。

## 三、数据角色矩阵

|阶段|Real Train|Real Validation|Natural Train|Natural Validation|Paired Train|Paired Validation|
|---|---|---|---|---|---|---|
|GMM参数拟合|是|不拟合，只可提供指定验证目标|否|否|由其后生成|由其后生成|
|Real VAE|训练|模型选择，真实Mask条件|否|否|Real目标已出自真实划分，不另算新增数据|同理|
|Sim VAE|否|否|训练|验证目标之一|Sim侧训练，与Natural每批32/32|Sim侧验证，与Natural目标等权|
|原C3 Translator|用Paired登记对应的Real目标|用Paired登记对应的Real目标验证|不直接配对训练|不直接配对选模|1200监督索引|240监督选模|
|原C0训练|是|不梯度训练|是|不梯度训练|否|否|
|原C0迁移评价|无新训练|真实参照780|无新训练|转换输入905|不替代Natural评价|不替代Natural评价|
|P44/P45/P49/P50优化|对抗参照/尺度/约束|开发评价|源输入/保持约束|开发评价|只经原C3初始化继承，不新增配对训练|不是本章后续主迁移评价集|
|P51|历史D fit24 Session|4Session外层接受评价|历史D fit及inner Root|10Root外层905窗|不使用为主评价|不使用为主评价|

P49/P51 D-inner来自Real Train中8 Session/936窗及Natural Train中10 Root/800窗；D-fit为24/4217与40/3249。D-inner不参与对应D参数更新，但生成器可访问全部Natural Train，故不能称G独立测试集。P51包络校准使用8个Real inner Session，LOSO遍历这8组，外层4组另行评价。

## 四、重复访问、复用与独立性

- Episode是具体执行及来源；Frame是保存控制行；Window是四帧切片；Root是按具体初态/来源规则登记的仿真组；Session是真实采集会话。一个Session可含多个Trial和窗口，一个Root可有多个派生记录。两个不同文件Hash不等于两个独立物理初态。
- Sim VAE等量混合导致1200 Paired组重复访问；45轮采样不是新增独立采集。Translator1200×45=54000次访问，完整训练855步与选中30轮570步是不同量（C3 translator/complete.json:2–9）。
- 字典16×5×3×2=480登记条件、160种放置后首帧指纹；指纹使用qpos与探针位置量化，非全部积分状态。不能480或160直接称认证独立Root。
- C4 `paired_cross_split.json:3–54`：Paired Train/Validation的reported_root共享19值，涉及Validation97窗；Real来源/Trial/Session及窗口数值Hash无相同值不等于物理全独立。reset_state复用与放置后指纹是两种记录，不得混为同一“初态完全相同”。
- 19/97的指纹源：C3 `inverse.py:84`及扩展同类记录，qpos与mocap_position首保存帧round(9)后拼接散列，不含速度等完整条件。支持登记层跨划分复用风险；不能推出97窗物理接触完全相同，也不能因Real Session隔离就忽略该风险。
- A体系的v4五包、旧五例、Sugar修复、Sink精度对照均有来源关联，不能累加为新增独立采集或新服务器时长；B体系也不能用A的Episode库存替代4×712窗口分母。

## 五、P37与P43

P37旧差异分析使用Real5153/1408、Sim1824/406；当前P43使用5153/780和4049/905。Real Train部分来源相同不等于整个实验合同相同。第5章5.1的Mask、幅值和帧变化统计来自P39对旧P37的诊断，5.2以后的方法数据是P43。旧Probe分解、当前原冻结C0、C23/C24自重建与P49/P50逐轮D是四种评价角色，不能因都计算BA/AUC就互相替换。

## 六、衔接结论与UNKNOWN

论文第4章:279–292和第5章5.2:7已基本避免“LeRobot即跨域直接训练集”的错误。本轮也未发现C3代码读取上述三个LeRobot目录。仍建议增加明确数据角色桥接说明（A06），不补造跨库一一来源关系。

UNKNOWN保留：全服务器独立Root/全库泄漏、真实传感器物理单位和通道配准、全部视频解码同步、历史模型完整一致性、各Seed与模板派生的独立物理泛化。若不扩大当前结论，这些可作为已披露局限；无需为审计或初稿而启动新实验。

## 七、Paired数量的分阶段核对

直接清点C3 `pair_pilot_manifest.csv`：60个尝试目标，accepted为Train32/Validation10；`pair_expansion_manifest.csv`：1807个扩展尝试目标，accepted为Train1168/Validation230。合并后`paired_manifest.csv:2–1441`按role登记1200/240，与`expansion_complete.json:2–3`一致。不能仅统计扩展清单就误判最终1200/240不成立，也不以所有尝试数作为训练分母。
