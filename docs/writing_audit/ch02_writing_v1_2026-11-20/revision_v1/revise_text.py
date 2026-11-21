from pathlib import Path
p=Path('thesis/body/ch02_simulation_platform.tex');s=p.read_text()
a=s.index('  \\fbox{\\parbox[c][2.4cm]');b=s.index('  \\label{fig:ch02_robot_model}',a)
s=s[:a]+r'''  \includegraphics[width=0.96\textwidth]{figures/ch02/revision_v1/assembled_robot.pdf}
  \caption{由当前XML与STL离线组合的机械臂和手部模型（关节零位，右侧为同一姿态的手部放大图）}
'''+s[b:]
s=s.replace('图\\ref{fig:ch02_robot_model}因此保留为待补的准确几何图位，驱动关系则按图\\ref{fig:ch02_drive}所示的实际拓扑表达。','图\\ref{fig:ch02_robot_model}按当前安装变换展示关节零位的静态几何，不表示任务执行姿态或动态验证结果；驱动关系按图\\ref{fig:ch02_drive}表达。')
s=s.replace('其中，$\\sigma_i$为奇异值，$\\lambda$为阻尼系数，$k_n$为姿态调节增益，$\\boldsymbol q_0$为参考姿态。','其中，$\\sigma_i$为奇异值，$\\lambda$为阻尼系数，$k_n$为姿态调节增益。默认开启零空间项且$k_n=0.1$；未显式指定时，$\\boldsymbol q_0$取各关节执行器控制上下限的中点，当前机械臂对应七维零向量。')
s=s.replace('求解后还需限制单次关节增量、关节范围及目标变化速度，防止迭代结果直接产生过大的控制跳变。','另一个参数posture\\_weight控制在误差与Jacobian中追加加权姿态残差，其默认值0仅关闭该附加残差，不关闭上述零空间项。式\\ref{eq:ch02_ik}对应默认设置下SVD成功的路径；SVD失败时退回阻尼最小二乘求解，该分支不叠加零空间项。迭代达到位姿容差后可提前退出，所得目标还需进行关节限幅及速度平滑。指定profile采用position模式，不调用该IK求解路径。')
s=s.replace('零空间调节增益 & 0.1 & 默认开启，可由调用方配置\\\\','零空间调节增益 & 0.1 & 默认开启，可由调用方配置\\\\\n    附加姿态残差权重 & 0 & 不影响零空间项的默认开关\\\\')
s=s.replace('    v_{t,j}&=\\alpha b_{t,j}+(1-\\alpha)v_{t-1,j}.','    v_{t,j}&=\\alpha b_{t,j}+(1-\\alpha)v_{t-1,j},\\\\\n    \\boldsymbol y_t&=\\operatorname{float32}\\!\\left(\\frac{\\operatorname{clip}(\\mathcal C_c(\\boldsymbol v_t),0,s)}{s}\\right).')
s=s.replace('其中，$t$为连续调用的观测索引。','其中，$t$为连续调用的观测索引，$\\mathcal C_c$为下述Patch内串扰算子，$\\boldsymbol y_t$为processed向量。末行适用于当前关闭高斯滤波和噪声、开启归一化的profile：先完成串扰及范围限制，再除以饱和尺度$s$，最后转换为float32。')
s=s.replace('保留8703','保留8703')
s=s.replace('非零比例 & 历史成功状态','非零比例（\\%） & 历史成功状态')
for a,b in [('0.084049720','8.405'),('0.152964408','15.296'),('0.182632829','18.263'),('0.004954035','0.495'),('0.127175911','12.718')]:s=s.replace(a,b)
s=s.replace('其中Push约为0.005，Stack约为0.183','其中Push为0.495\\%，Stack为18.263\\%')
# Annotate units without changing any XML numbers.
s=s.replace('表\\ref{tab:ch02_actuators}列出的腱执行器力与机械臂关节力矩对应不同传动坐标，不能直接比较数值大小。','表\\ref{tab:ch02_actuators}采用模型的SI单位约定：机械臂$k_p$单位为N$\\cdot$m/rad，腱执行器$k_p$单位为N/m；腱力与关节力矩不能直接比较。这些是仿真参数，不是实物标定值。')
p.write_text(s)
