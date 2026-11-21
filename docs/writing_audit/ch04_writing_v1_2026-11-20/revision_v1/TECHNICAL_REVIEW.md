# 第四章R1定点技术复核

状态：临时编译完成，等待PDF审核；不冻结。

## A 时间字段与式4-2

直接核对 `source/data/production_recorder.py`：`_before`（146–187行）在物理推进前从`env.data.time`读取timestamp；`_after`（192–200行）在物理推进后读取next_timestamp。`step`、`step_physics`（202–214行）分别包围物理执行；`OnlineRecorder.append`（232–244行）仅提交任务记录器接受的pending行，未接受的执行不自动成为正式数据行。

因此 t_i 是第i条被接受记录的执行前采样时刻，t_i^+ 是这一步结束时刻，t_{i+1} 是下一条被接受记录的执行前时刻。在连续执行、每步均被接受且没有额外物理推进/重置时，后两者衔接。跳步、未接受记录、异常中断或额外推进不能假设衔接。`validate_online_arrays`第73–76行分别检查相邻timestamp间隔和单行next_timestamp−timestamp；这是两个独立的时间合同检查，并非直接对每一对next_timestamp与后继timestamp作单独断言。require_causal路径另检查命令与qpos/qvel连续性（77–83行）。本轮未重新读取或认证历史全部时间数组。

保留原式4-2，只补充三个符号的定义与连续性前提；不将固定周期等同于缺口可忽略。

## B 队列状态与图4-4

`source/data/production_queue.py`不是Python Enum定义，而是在SQL及逻辑分支中使用真实状态字符串：PENDING、RUNNING、VALIDATING、EXPORTED、DUPLICATE、QUARANTINED。

register第315–348行：要求有效所有权与VALIDATING，外部verification.passed为True；来源或content重复进入DUPLICATE，无重复则插入episodes并进入EXPORTED。fail及租约恢复（203–224、300–312行）根据重试条件回到PENDING或进入QUARANTINED。EXPORTED确实存在，且不等同于success=True；合法失败是否计账仍取决于用途与冻结规则。

因此保留EXPORTED名称，仅将图和文字改为“登记：EXPORTED/DUPLICATE；异常隔离：QUARANTINED”，避免误读为成功登记有三个并列出口。长状态使用mbox保持完整。

## C 外部引用

现有reference.bib有已核验的 `mandlekar2022demonstrations`（CoRL/PMLR164），可支持开头“完成示教不自动等于适合训练”的一般背景，本轮仅复用该条目增加一处引用。它不作为LeRobot字段或本项目统计的依据。

逐项查阅现有30条BibTeX条目，没有LeRobot项目自身条目。该工具的字段、版本和转换合同继续以内部审计与实际产物为依据，不以其他数据集论文替代官方软件来源。建议后续获授权时为实际使用版本补充官方软件引用，记录版本/commit与URL；本轮不检索新题录，不新增或修改reference.bib，不猜作者和DOI。

## 修改边界

仅第四章正文的时间定义、队列出口说明、图内对应标注及一处已有文献引用改变。原六节结构、全部历史数值、质量边界和公式主体均保留。代码、原始数据、第一至三章、主文件、BibTeX、模板及配置Hash复核未变，详见static_checks.json。旧V1审计与阅读稿保留为历史版本，以本目录修订源及PDF为R1依据。
