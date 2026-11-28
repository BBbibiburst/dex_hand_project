from pathlib import Path
import json
R=Path('/home/zmy/dex_hand_project');O=R/'thesis_audit/2026-11-28/writing_reconstruction_r1';e=json.loads((O/'edits.json').read_text())
def rep(n,a,b,why):
 p=R/'thesis/body'/n;s=p.read_text();assert s.count(a)==1,(n,a[:45],s.count(a));p.write_text(s.replace(a,b,1));e.append({'source':str(p.relative_to(R)),'before':a,'after':b,'purpose':why})
rep('ch01_introduction.tex','全文按七章组织，由实验条件逐步过渡到数据构建、触觉重建及评价。第一章说明研究背景和相关技术，归纳研究问题及工作范围。第二章建立机械臂、欠驱动手和曲面触觉观测的仿真基础，交代控制与信号来源。第三章在此基础上研究抓取候选的适配、可达性筛选和轨迹验证，说明多任务专家示教如何形成。第四章从数据使用角度检查示教记录的组织、关联、转换和质量，使后续研究所使用的数据及其限制具有可追溯依据。第三、四章分别回答示教如何产生及如何成为可核验数据，避免将任务成功与数据质量混为一谈。','全文按七章组织。第一章说明研究背景、相关工作及研究问题；第二章建立机器人控制与触觉观测的仿真基础；第三章研究抓取候选的适配、可达性筛选和专家轨迹验证；第四章进一步检查记录的组织、时序、转换和质量。第三、四章由操作执行转向样本使用，分别回答示教如何产生，以及记录如何满足学习所需的数据要求。','组织结构不重复前节技术清单')
rep('ch01_introduction.tex','第五章围绕缺少同步物理配对的条件，介绍真实统计先验指导的探针伪配对、独立双域表示和潜变量映射，并分析初始重建的效果与局限。第六章承接初始迁移结果，研究共享参数向量场和迭代对抗优化，结合多视角判别、分布质量、输入保持及跨会话分析检验优化结果，同时讨论评价器的适用范围。第七章归纳各研究方向获得的结论，区分已实现功能、受限实验结果与尚未解决的问题，并提出与现有证据缺口相对应的后续方向。','第五章研究非配对条件下的统计监督、双域表示和潜变量映射；第六章进一步检验共享向量场与对抗优化，并从生成质量、输入保持和评价器泛化分析其效果；第七章归纳研究结论及未来工作。','保留章节递进，避免重复方法名堆砌')
rep('ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex','仅4个评价Session使不确定性估计受限，组内大量窗口不能补成大量独立Session；该区间也不认证未知采集条件下的覆盖。','评价仅含4个Session，不确定性估计仍受会话覆盖范围限制；组内窗口主要增加同一条件下的观测。','集中组数限制，避免三句同义否定')
rep('ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex','矩阵是窗口计数，不是组宏比例。高FR只说明生成分数越过阈值，与真实样本接受能力不足可以同时出现；它既不证明域不可区分，也不证明生成分布正确。AUC则描述连续分数的排序能力，并非分布距离或物理质量。','矩阵按窗口计数，FR与Real宏接受率则另按来源分组。生成样本越过阈值的同时，真实样本仍可能被拒绝；AUC对连续分数排序的检查，为这一阈值现象提供了补充描述。','具体解释阈值与排序结果，不堆叠指标禁区')
rep('ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex','这是经验分数包络，不是经物理标定的触觉置信区间；Joint包络也不同于三支各自包络的交集，不能替代原0.5分类阈值。','该包络描述inner样本的经验分数范围，与原0.5分类阈值承担不同作用；Joint包络由最小logit直接构造，区别于三支包络的交集。','明确包络构造对象与评价角色')
rep('ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex','说明平均覆盖不能充分代表跨Session稳定性，不能将某一组的较好校准直接推广到另一组。该诊断仍限于8个历史inner Session，未登记新的稳定性硬门槛，不能推出跨Session理论不可达，也不能单凭会话标签归因传感器、任务或物体因素。','这些差异表明，某一Session上的较好覆盖在转移到另一Session后仍可能失效。诊断范围为8个历史inner Session，未登记新的稳定性硬门槛；具体差异来源还需结合传感器、任务和物体条件进一步区分。','由跨会话结果推导实际意义，保留未归因状态')
(O/'edits.json').write_text(json.dumps(e,ensure_ascii=False,indent=2));print(len(e))
