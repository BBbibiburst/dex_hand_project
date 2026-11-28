from pathlib import Path
import json
r=Path(__file__).resolve().parents[3];a=Path(__file__).parent;ed=[]
def rep(path,old,new,ids):
 p=r/path;s=p.read_text();assert s.count(old)==1,(path,old[:70],s.count(old));p.write_text(s.replace(old,new));ed.append({'file':path,'ids':ids,'before':old,'after':new})
p='thesis/body/ch02_simulation_platform.tex'
for old,new,ids in [('figures/revision_20261127/ch02_architecture.pdf','figures/annotation_R2/platform_actual_architecture.pdf',[29]),('figures/revision_20261127/ch02_drive.pdf','figures/annotation_R2/joint_tendon_geometry.pdf',[31])]:rep(p,old,new,ids)
# captions are identified by label context rather than stale numeric figure numbers.
s=(r/p).read_text();anchor='\\label{fig:ch02_drive}';i=s.index(anchor);ca=s.rfind('\\caption{',0,i);end=s.index('\n',ca);old=s[ca:end];rep(p,old,r'\caption{当前MuJoCo关节几何与XML固定腱系数、执行器对应关系}',[31])
rep(p,'机器人组合模型由RM75B机械臂、途见科技自研灵巧手和安装底座构成。','机器人组合模型选用与实验室设备型号对应的RM75B机械臂及途见科技自研欠驱动灵巧手，并通过底座和末端安装变换完成组合。XML规定关节、驱动和刚体连接，URDF提供机构描述，STL保存连杆与皮肤形状；构建器据此统一模型接口。本文在该资产配置上研究控制与感知布局，几何尺寸误差、动力学参数和触觉响应的实机标定作为后续验证内容。',[21,33])
rep(p,'平台采用由皮肤STL网格派生采样点的方式，将阵列布局与所属刚体的几何关系连接起来。','平台按实验室灵巧手的指节、指尖和手掌感知区域组织阵列，由相应皮肤STL派生采样点，再将局部位置与朝向安装到所属指节刚体。该建模过程使仿真感知区域随机构运动，并保留区域与通道的对应。',[42])
# Insert actual intermediate geometry, preserving existing representative-shape figure.
s=(r/p).read_text();idx=s.index('网格采样点还需安装到正确的刚体上。')
new=r'''三类策略的区别来自表面几何：指节适合沿主要轴向组织截面，指尖需要处理封闭弯曲外壳与极点，手掌则通过投影三角形保留不规则网格。图\ref{fig:ch02_sampling_steps}给出实际采样函数的中间输出。指节和指尖先从STL识别密集外壳样本，再按各自Bezier配置拟合并取参数单元中心；手掌在PCA参数域设置格点，通过三角形定位与重心插值恢复三维位置。图中从原始几何到密集样本、再到最终阵列的变化，展示了规则索引如何依附不同曲面，而非将三种形状套用同一平面网格。

\begin{figure}[p]
 \centering
 \includegraphics[width=0.98\textwidth,height=0.80\textheight,keepaspectratio]{figures/annotation_R2/sampling_processes.pdf}
 \caption{三类曲面采样的实际中间输出：STL与局部轴、外壳或参数域、最终采样中心}
 \label{fig:ch02_sampling_steps}
\end{figure}

'''
s=s[:idx]+new+s[idx:];(r/p).write_text(s);ed.append({'file':p,'ids':[44],'before':'','after':new})
# Detailed task interface description with historical case identity kept separate.
s=(r/p).read_text();idx=s.index('五类任务分别考察抓持后的垂直抬升')
new=r'''五类任务使用桌面、机器人及任务对象组成场景，初始条件由物体位姿、手部配置和目标区域确定。Lift以桌面物体为对象，依次接近、闭合、保持和抬升，任务接口按物体高度判断抬升结果；严格示教另检查尾段稳定性。PickPlace增加目标区域，抓持后运输、下降、释放并检查物体进入目标且手部离开。Stack在场景中增加支撑物，搬移目标物体到支撑面，释放后检查物体间接触与持续稳定。Push在桌面支撑下建立手物接触，反馈推动至目标区域，并检查位置、竖直范围与平面速度。NutAssembly场景包含待操作件与固定装配结构，抓持后调整姿态、运输和下降，按具体几何版本检查对准、插入及稳定状态。表中历史案例对应Sugar Box抬升、Tomato Soup Can搬移与推动、Stacking Ring堆叠及保留抓持命令的装配记录；各历史结果按自身manifest和版本解释，当前接口用于说明任务定义。

'''
s=s[:idx]+new+s[idx:];(r/p).write_text(s);ed.append({'file':p,'ids':[49],'before':'','after':new})
p='thesis/body/ch03_grasp_demonstrations.tex'
for old,new,ids in [('figures/revision_20261127/ch03_overall_pipeline.pdf','figures/annotation_R2/grasp_evidence_pipeline.pdf',[50]),('figures/revision_20261127/ch03_candidate_sources.pdf','figures/annotation_R2/candidate_geometry_feedback.pdf',[53]),('figures/revision_20261127/ch03_reachability.pdf','figures/annotation_R2/wrist_recorded_targets.pdf',[57]),('figures/revision_20261127/ch03_strict_replay.pdf','figures/annotation_R2/recorded_tracking_strict.pdf',[58]),('figures/revision_20261127/ch03_task_transfer.pdf','figures/annotation_R2/five_task_phase_records.pdf',[59])]:rep(p,old,new,ids)
# Caption labels are stable.
labels={'fig:ch03_overall_pipeline':('Sugar Box同物体案例的候选几何、执行反馈与示教构建证据',[50]),'fig:ch03_candidate_sources':('同次运行的代表候选、驱动比例与进化档案反馈',[53]),'fig:ch03_reachability':('腕姿模板的保存目标、预检残差与历史执行结果',[57]),'fig:ch03_strict_replay':('保存模型与关节状态的末端跟踪重建及同源历史Strict结果',[58]),'fig:ch03_task_transfer':('五类历史专家示教的实际阶段画面（按保存阶段中点选帧）',[59])}
for label,(caption,ids) in labels.items():
 s=(r/p).read_text();i=s.index('\\label{'+label+'}');ca=s.rfind('\\caption{',0,i);end=s.index('\n',ca);rep(p,s[ca:end],'\\caption{'+caption+'}',ids)
rep(p,'图\\ref{fig:ch03_overall_pipeline}中，机器人约束筛选将对象相对目标转换为世界坐标下的机械臂末端目标，轨迹执行再连接接近、闭合、保持和后续操作。','图\\ref{fig:ch03_overall_pipeline}以Sugar Box贯穿候选几何与执行反馈：物体表面和保存的接触候选为几何评价提供输入，进化档案补充物理反馈，腕姿扩展与IK筛选再连接机械臂目标，连续命令进入实际执行与Strict回放。各画面保留原始记录身份，用于说明同一物体上的各环节，而非将不同候选拼成同一条优化轨迹。机器人约束筛选将对象相对目标转换为世界坐标下的机械臂末端目标，轨迹执行再连接接近、闭合、保持和后续操作。',[50])
rep(p,'图\\ref{fig:ch03_candidate_sources}中的MJWarp PPO仅作为条件性补充，在格点阶段未解决且仍有可用模板时进行模板选择与手部驱动编辑。','图\\ref{fig:ch03_candidate_sources}展示同次运行保存的几何候选、进化代表候选及档案最优适应度；接触点与法向直接来自候选文件，六维驱动比例反映不同抓持配置。MJWarp PPO仅作为条件性补充，在格点阶段未解决且仍有可用模板时进行模板选择与手部驱动编辑，本图Sugar Box代表记录未将该分支作为已执行结果。',[53])
# Clarify recorded precheck maxima, no false unreachable examples.
s=(r/p).read_text();idx=s.index('\\section{连续逆运动学与轨迹验证}')
new=r'''图\ref{fig:ch03_reachability}将腕姿编辑与已保存的预检结果对应。同次日志登记39个编辑中18个通过预抓取及抓取两目标检查，模板索引保存其中12个执行或复用模板。图中的残差是两目标误差的最大值；橙色模板虽然预检残差较小，实际执行仍失败。这说明腕姿可达性只解决目标求解条件，后续接触保持仍需通过物理执行检查。被预检排除候选的逐姿态状态未保存在该索引中，因此图中以原日志给出筛选规模，不另画未经记录的不可达姿态。

'''
s=s[:idx]+new+s[idx:];(r/p).write_text(s);ed.append({'file':p,'ids':[57],'before':'','after':new})
s=(r/p).read_text();idx=s.index('已有普通Replay接口主要检查')
new=r'''图\ref{fig:ch03_strict_replay}进一步利用另一条开发记录保存的模型快照与447帧关节状态，作静态正运动学重建。位置差为步末末端位置与同一行保存IK目标之差的范数，朝向差取四元数对应的最短旋转角；它们包含目标滤波、伺服和动力学跟踪影响，区别于IK求解器内部残差。关键阶段画面也仅恢复保存姿态。图中同源历史Strict文件登记322帧回放为FINAL\_VERIFIED，源序列与回放帧数分别标注，未在本次绘图中重新执行验证。

'''
s=s[:idx]+new+s[idx:];(r/p).write_text(s);ed.append({'file':p,'ids':[58],'before':'','after':new})
# Correct an obsolete reference describing a branch graph, now replaced by actual phases.
rep(p,'图\\ref{fig:ch03_task_transfer}将该过程单独表示，以对应桌面支撑条件下的持续接触控制。','图\\ref{fig:ch03_task_transfer}给出预接近、接触接近、推动及任务验证的历史画面，展示桌面支撑条件下的持续接触控制。',[59,60])
# Source-accurate archive mechanism, no new diversity performance claim.
rep(p,'档案在搜索中保留不同构型；其执行表现由后续机器人约束检查与任务验证评价。','档案按平移、旋转和驱动比例构成的嵌入计算候选距离；新候选若与最近档案项的距离小于新颖性阈值，仅在适应度更高时替换该项，否则作为新构型加入。父代选择再以局部密度修正适应度，减少拥挤邻域反复占用搜索预算；最终按适应度整理候选，供后续腕姿与机器人约束筛选。',[55])
p='thesis/body/ch04_multimodal_data.tex';rep(p,'figures/revision_20261127/ch04_carriers.pdf','figures/annotation_R2/data_lineage_instances.pdf',[64]);rep(p,'数据载体及其转换关系示意','v4抓取放置与Sugar Box修复导出的实际来源、字段及行索引关系',[64]);rep(p,'这些载体并非互相独立的数据集，每次转换都需要保留父来源。','图中分别使用v4抓取放置包和Sugar Box修复导出两个实例：前者保留source\\_episode、规范modalities及登记文件Hash，后者将源477行中的动作与观测按偏移映射为340帧LeRobot序列。两者属于不同执行来源，转换路径各自保留父来源与版本，来源哈希用于追踪而非推断独立Root。',[64])
(a/'edits.json').write_text(json.dumps(ed,ensure_ascii=False,indent=2)+'\n');print('integrated',len(ed),'edits')
