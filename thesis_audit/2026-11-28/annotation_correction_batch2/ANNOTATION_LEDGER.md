# 原始64条批注｜第二批累计台账

继承原始R2批注ID与原文；上批台账原样保留。此表仅统计原批注，不代表外部基线、消融或长期计划全部完成。原source_line_hint来自旧导航，正文行号已随编辑变化，请以本批SOURCE_EVIDENCE.md及changes.diff定位。

## 01｜原PDF第1页｜已处理（待人工验收）

> 改成2026年12月

`thesis/front/cover.tex`：中文封面日期统一为2026年12月；CSV同步。

## 02｜原PDF第2页｜待处理

> 感觉整个论文字体不太对，我听说要安装windows专用的什么字体

`thesis/front/cover.tex`：字体保持当前模板Fandol；须先核对学院正式字体要求及字体授权，再决定替换。

## 03｜原PDF第2页｜已处理（待人工验收）

> 同上日期去掉14日

`thesis/front/cover.tex`：内封去除14日；CSV同步。

## 04｜原PDF第2页｜已处理（待人工验收）

> 改为计算机技术

`thesis/front/cover.tex`：中文专业改为计算机技术；CSV同步。

## 05｜原PDF第3页｜已处理（待人工验收）

> 计算机技术译为Computer Technology

`thesis/front/cover.tex`：英文专业改为Computer Technology；CSV同步。

## 06｜原PDF第3页｜已处理（待人工验收）

> 日期问题

`thesis/front/cover.tex`：英文日期改为December 2026。

## 07｜原PDF第4页｜部分处理

> 这段不太好，没有说明仿真数据相较于实机数据的意义，即廉价，大规模之类的，如果不这么说机器人示教为什么不用实机示教，参考例文：随着深度学习模型在边缘端部署需求的增长，GPU 与 CPU 大小核组成的 异构多核架构为模型推理任务提供了重要的算力基础。然而，此类平台在多模 型并发推理场景下面临两方面核心挑战：一是模型计算图规模庞大，如何挖掘 图中的并行结构并将其合理映射到异构设备上以最小化推理延迟；二是多模型 请求动态到达且携带端到端截止时间约束，如何在有限的异构资源上进行实时 调度以最大化截止时间达标率。本文围绕上述问题，从计算图划分与多模型在 线调度两个层面展开研究。

`thesis/front/abstract.tex`：摘要已按实机采集代价→仿真用途→研究问题重写；新增实验对照尚未开展。

## 08｜原PDF第4页｜部分处理

> 没有标注机械臂品牌RealMAN（睿尔曼智能）

`thesis/front/abstract.tex`：摘要加入RealMAN（睿尔曼智能）；官方品牌来源见本轮报告。

## 09｜原PDF第4页｜部分处理

> 这段写的不行，没有体现出我的创新，建议写改进优化什么的：我想要的感觉：针对模型部署过程中硬件资源利用不充分的问题，提出了一种面向异构平 台的计算图划分策略。通过串行链折叠与层次化划分将算子级计算图压缩为可 并行执行的分层结构，结合两阶段块分配算法将各基本块调度至合适的目标设 备，使异构硬件负载趋于均衡，相较于仅使用单一类型设备执行取得了更优的 推理性能。同时建立算子性能建模方法，为分配决策提供数据支撑。实验结果 表明，在六个视觉推理模型上相比 dlnne 推理时延降低了 5%至 12%。

`thesis/front/abstract.tex`：摘要区分指标适配与参考设计的搜索，加入既有10条局部扰动Strict记录；未虚构原创性或方法优势。

## 10｜原PDF第4页｜部分处理

> 这段和上一段衔接的不好，有点莫名其妙

`thesis/front/abstract.tex`：摘要按平台/示教→数据质量→跨域方法与评价衔接，尚待人工确认。

## 11｜原PDF第4页｜部分处理

> 这段全是具体技术细节，写在摘要感觉不太好

`thesis/front/abstract.tex`：压缩数据处理细项，保留指定340行字段一致性结果。

## 12｜原PDF第4页｜证据限制/不按字面执行

> 论文里不要提缺少物理配对，默认一样，后面有涉及的也一并更改

`thesis/front/abstract.tex`：按字面改为默认物理一样将改变事实，未执行。摘要改以统计先验构造伪配对正面叙述；同步物理真值与统计配对仍须区分。

## 13｜原PDF第4页｜部分处理

> 这个只有触觉迁移的结果，少了其他两个创新点的数据表现，同时说的触觉迁移好像没有什么效果，应该强化优势表述只在最后提一句不足和展望

`thesis/front/abstract.tex`：加入抓取与指定数据修复结果；迁移未通过事实保留于结果结尾，不能改写为成功。

## 14｜原PDF第5页｜已处理（待人工验收）

> 根据中文版的修改同步修改

`thesis/front/abstract.tex`：英文摘要根据本轮中文摘要同步重写。

## 15｜原PDF第9页｜已处理（待人工验收）

> 这段写的不像人话，看不懂

`thesis/body/ch01_introduction.tex`：绪论起句改为多指接触与稳定持物的具体问题。

## 16｜原PDF第9页｜部分处理

> 这段也不是人话

`thesis/body/ch01_introduction.tex`：摘要改以采集操作、复位和维护解释实机代价，其他背景段落仍需逐段复核。

## 17｜原PDF第9页｜已处理（待人工验收）

> 应该展开讲仿真的优势

`thesis/body/ch01_introduction.tex`：补充初态保存/恢复、完整状态读取、碰撞筛查与失败复现的作用。

## 18｜原PDF第10页｜已处理（待人工验收）

> 带上牌子

`thesis/body/ch01_introduction.tex`：绪论首次说明RealMAN（睿尔曼智能）RM75B。

## 19｜原PDF第10页｜已处理（待人工验收）

> 看不懂

`thesis/body/ch01_introduction.tex`：改写绪论记录关联段，说明动作前观测、动作后反馈、索引与采样频率的用途；与跨域对应分开。

## 20｜原PDF第10页｜已处理（待人工验收）

> 这句删去或者改一改

`thesis/body/ch01_introduction.tex`：改为解释TACTO和Taxim的读出机制，不重复否定其作为本文物理依据。

## 21｜原PDF第10页｜待处理

> 提一下机械臂和手与实验室的实机是对齐的

`thesis/body/ch01_introduction.tex`：绪论已说明硬件选择对应实验室；逐模型几何/动力学一致性仍待证据，不能扩成全参数对齐。

## 22｜原PDF第12页｜部分处理

> 这部分应该提前到1.2.1，逻辑是平台-》抓取-》收集数据，并且应该介绍一下目前数据收集框架的进展，如robosuite，maniskill

`thesis/body/ch01_introduction.tex`：将已有平台相关工作移到1.2.1，再接抓取和数据；SAPIEN/robosuite/ManiSkill原文调研未完成。

## 23｜原PDF第12页｜证据限制/不按字面执行

> 加上sapien，介绍mujoco的独特优势-易于模拟欠驱动，而其他模拟器不行，还有解释mujoco只是物理引擎，需要在上面搭建框架

`thesis/body/ch01_introduction.tex`：原批注要求SAPIEN/欠驱动优势解释，不能无对照断言其他引擎不能模拟欠驱动；待原文与功能核验。

## 24｜原PDF第13页｜待处理

> 加入仿真平台用灵巧手人工示教不现实，我试过但由于灵巧手不同于夹爪，有强遮挡，同时由于仿真的示教局限，很难仿真示教，最好再找篇引用证明这一点

`thesis/body/ch01_introduction.tex`：应补作者实际遥操作困难及经核验相关文献；本轮未添加未经查证文献。

## 25｜原PDF第14页｜待处理

> 应该是即研究了，又研究了，现在这个不行

`thesis/body/ch01_introduction.tex`：Narang方法与本文工作的对照需按监督来源/读出差异重新组织。

## 26｜原PDF第14页｜已处理（待人工验收）

> 应该先论述为什么会缺少配对，即实机和仿真很难精准复现同一接触过程

`thesis/body/ch01_introduction.tex`：解释逐样本物理对应需要接触位置、姿态、加载和采样时刻可核对，引出真实统计监督。

## 27｜原PDF第16页｜待处理

> 这个表前后都是空白，这样不好，尝试前后扩写

`thesis/body/ch01_introduction.tex`：表1-1留白需在完整排版审核中定位；本轮未靠补字填空。

## 28｜原PDF第18页｜待处理

> 换个正式说法

`thesis/body/ch02_simulation_platform.tex`：该句需在上下文中换为执行结果与记录质量分别判定。

## 29｜原PDF第19页｜待处理

> 这个图太简单，画一个复杂图

`thesis/body/ch02_simulation_platform.tex`：需以真实模块输入/输出、控制周期和模型关联重画架构，未将换样式视为完成。

## 30｜原PDF第20页｜已处理（待人工验收）

> 没有解释什么是MJCF

`thesis/body/ch02_simulation_platform.tex`：首次解释MJCF及模型文件组织内容。

## 31｜原PDF第21页｜待处理

> 图太简单

`thesis/body/ch02_simulation_platform.tex`：需实际关节轴与固定腱驱动标注，不能画替代机构示意。

## 32｜原PDF第22页｜已处理（待人工验收）

> 这句像ai写的，明明没有人问你

`thesis/body/ch02_simulation_platform.tex`：删除闭链/Flex/硅胶的连续否定，改为说明刚体与固定腱模型。

## 33｜原PDF第22页｜部分处理

> 写的有问题

`thesis/body/ch02_simulation_platform.tex`：删除堆叠的否定句，正面说明资产与模型表示；实物几何/物理误差测量未补。

## 34｜原PDF第22页｜已处理（待人工验收）

> 删去

`thesis/body/ch02_simulation_platform.tex`：删除段落中重复的非任务姿态声明，保留图题对静态模型的说明。

## 35｜原PDF第22页｜已处理（待人工验收）

> 写的太有ai味，同时没有解释各参数是什么，“不表示”，”不能“这些人不会这么写

`thesis/body/ch02_simulation_platform.tex`：按XML与官方文档说明condim4、滑动/扭转/滚动维数及solref正值格式，去除成串否定。

## 36｜原PDF第23页｜已处理（待人工验收）

> 这部分没有突出这个是我自己写的控制器，也没有解释清楚控制器的总体设计

`thesis/body/ch02_simulation_platform.tex`：明确本文组合控制器的目标求解、双模式接口、手部腱位置控制与MuJoCo执行分工。

## 37｜原PDF第23页｜已处理（待人工验收）

> 图太简陋

`thesis/body/ch02_simulation_platform.tex`：替换控制图：关节/末端分支、SVD与姿态项、临时状态、平滑、固定腱和多速率时序。

## 38｜原PDF第23页｜已处理（待人工验收）

> 人不会这么说

`thesis/body/ch02_simulation_platform.tex`：直接解释固定腱位置由关联关节加权组合确定。

## 39｜原PDF第24页｜已处理（待人工验收）

> 人不会这么说

`thesis/body/ch02_simulation_platform.tex`：保留默认参数可覆盖的必要说明，删去逐历史Episode否定。

## 40｜原PDF第24页｜已处理（待人工验收）

> 人不会这么说

`thesis/body/ch02_simulation_platform.tex`：删去传感器带宽旁论，说明积分与采样间隔。

## 41｜原PDF第24页｜已处理（待人工验收）

> 删去

`thesis/body/ch02_simulation_platform.tex`：删去电极等价性的重复旁论，说明安装与通道顺序。

## 42｜原PDF第24页｜部分处理

> 目的是为了和实机相吻合

`thesis/body/ch02_simulation_platform.tex`：说明阵列按指节/手掌感知区组织；尚不能认证与实物电极逐点配准。

## 43｜原PDF第24页｜已处理（待人工验收）

> 人不会这么说

`thesis/body/ch02_simulation_platform.tex`：改为说明各区域分别从STL计算采样点。

## 44｜原PDF第24页｜待处理

> 在策略前先说明为什么要分好几种策略，同时后面的三个策略写的太简略，我期望是要写的很详细，再加上几个各策略的算法图和效果图

`thesis/body/ch02_simulation_platform.tex`：需对三类实际采样算法按曲率适用条件逐步解释；已有代表形状图不等于该条全部完成。

## 45｜原PDF第29页｜已处理（待人工验收）

> 这段全是不怎么样不怎么样，不像人，touch_grid之前就没提过突然冒出来

`thesis/body/ch02_simulation_platform.tex`：上批去除touch_grid突现，本批以局部响应分配说明邻域扩散用途，原生/processed语义集中定义。

## 46｜原PDF第29页｜已处理（待人工验收）

> 图太简单

`thesis/body/ch02_simulation_platform.tex`：替换处理图：真实公式的幅值曲线、IIR状态缓存、内部8邻域分配系数及实际开关。曲线为算子示意非新实验。

## 47｜原PDF第30页｜已处理（待人工验收）

> 应该给几个示例热力图

`thesis/body/ch02_simulation_platform.tex`：补入五条历史仿真记录的完整序列热图，来源及Hash见figure_sources.json；不是实机测量图。

## 48｜原PDF第30页｜已处理（待人工验收）

> 这段通体都是不怎么样不怎么样，完全不像人，同时内容也感觉没什么意义

`thesis/body/ch02_simulation_platform.tex`：重组2.6为数组/时间/热图/场景观察，删除通用验证方框，限制集中一段。保留UNKNOWN版本问题与原数值。

## 49｜原PDF第32页｜部分处理

> 这段前还应该加上操作环境的建模和对五个任务的定义和展示

`thesis/body/ch02_simulation_platform.tex`：补五任务操作对象说明及历史视频起终帧；更详细的各任务目标/成功判据图仍待补。

## 50｜原PDF第34页｜待处理

> 图太简单

`thesis/body/ch03_grasp_demonstrations.tex`：图3-1需真正贯穿同一物体候选/可达性/轨迹与任务。

## 51｜原PDF第35页｜已处理（待人工验收）

> 本论文

`thesis/body/ch03_grasp_demonstrations.tex`：当前第一、三章的本项目改为本文。

## 52｜原PDF第35页｜已处理（待人工验收）

> 还是防御性问题，用了太多不怎么样了

`thesis/body/ch03_grasp_demonstrations.tex`：直接解释官方指标适配六驱动代理的作用，接续物理反馈；不再重复否定原创性。

## 53｜原PDF第36页｜待处理

> 图太简陋

`thesis/body/ch03_grasp_demonstrations.tex`：图3-2需候选形状与物理反馈的真实例子。

## 54｜原PDF第36页｜已处理（待人工验收）

> 删去

`thesis/body/ch03_grasp_demonstrations.tex`：删除未完整复现Isaac的独立重复声明，保留DexEvolve设计参考与适配定位。

## 55｜原PDF第36页｜部分处理

> 防御性问题

`thesis/body/ch03_grasp_demonstrations.tex`：改为说明档案保留不同构型、后续执行验证；尚无公平多样性基准。

## 56｜原PDF第36页｜已处理（待人工验收）

> 防御性问题

`thesis/body/ch03_grasp_demonstrations.tex`：用源候选的局部编辑和被评价候选预算解释39格点，不再连续否定独立性。

## 57｜原PDF第38页｜待处理

> 图太简陋，没有实际演示图

`thesis/body/ch03_grasp_demonstrations.tex`：缺指定案例腕部扰动与可达性原始可视化。

## 58｜原PDF第40页｜部分处理

> 图太简陋

`thesis/body/ch03_grasp_demonstrations.tex`：新增447帧Sugar Box历史关节目标、物体高度与接触数，阶段直接取manifest；未补末端跟踪误差/连续IK残差图。

## 59｜原PDF第42页｜待处理

> 图太简陋

`thesis/body/ch03_grasp_demonstrations.tex`：第二章已加五任务场景；第三章专家阶段图还需关联真实轨迹分段。

## 60｜原PDF第42页｜已处理（待人工验收）

> 不，而是这种太ai

`thesis/body/ch03_grasp_demonstrations.tex`：Push改为桌面接触、物体目标偏差反馈和实际位移评价的正面定义。

## 61｜原PDF第44页｜待处理

> 防御性问题

`thesis/body/ch03_grasp_demonstrations.tex`：失败案例图题与正文需集中说明记录范围，未将已有图当作闭合。

## 62｜原PDF第45页｜待处理

> 防御性问题

`thesis/body/ch03_grasp_demonstrations.tex`：需将n14/n127批次条件移入统一比较说明，再分析能比较的结果。

## 63｜原PDF第45页｜已处理（待人工验收）

> 防御性问题

`thesis/body/ch03_grasp_demonstrations.tex`：第三章仅说明轨迹附带任务与验证信息，数据转换/来源问题交第四章，删重复训练资格警告。

## 64｜原PDF第47页｜待处理

> 图太简陋

`thesis/body/ch04_multimodal_data.tex`：数据载体图需真实文件/索引/父来源实例，不能再画通用方框。
