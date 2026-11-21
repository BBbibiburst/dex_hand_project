# 定点技术核查与修订

## A IK的两个姿态机制

重新核对arm.py::__init__/bind/_solve_ik和composite.py::create_robot_controller及指定profile。use_nullspace默认True，nullspace_gain=0.1；未提供nullspace_posture时，bind使用0.5*(ctrl_low+ctrl_high)，当前七关节范围对称，故q0为七维零向量。posture_weight默认0关闭的是额外的加权姿态残差行，不是零空间项；大于0时添加diag(sqrt(weight*joint_weights))及对应姿态误差。

SVD使用full_matrices=False；伪逆为V diag(s/(s²+lambda²)) Uᵀ，投影I-Jinv*J。自适应阻尼实际参与SVD路径。SVD失败时使用Jᵀ(JJᵀ+damping²I)⁻¹e，此分支不增加零空间项。默认posture_weight=0且达到位姿容差时提前退出，因此不能说每步必定继续优化姿态。位置profile不调用IK；其未传入覆盖上述默认值的选项。正文已补齐这些区别，不误写默认姿态项失效。

## B 处理链逐行核验

signal_processing.py::process实际顺序：float64副本 → max(raw-deadzone,0) → clip[0,saturation] → saturation*(value/saturation)^exponent → IIR → Patch内crosstalk → 可选Gaussian → patch结果clip → 可选noise并clip → normalize时除saturation → float32。

IIR保存的是串扰前_filtered状态；后处理值不反馈进IIR。reset(size)建立零向量，reset(None)置None，两种首次调用路径不同。当前profile的crosstalk=.12、Gaussian=0、noise=0、normalize=True，正文公式末行明确写为float32(clip(C_c(v),0,s)/s)，没有把除s放到IIR或串扰之前。通用可选链在文字中保留，公式末行明确限定当前profile。

## C XML和参数单位

逐项读取当前arm/hand actuator节点和tendon/fixed，实际字段保存在actuator_xml_check.json；Builder仅覆盖solver/timestep/iterations/cone，没有更改执行器ctrlrange/kp/forcerange或腱系数。表中数值与XML相符，无需更换数值。

采用模型SI约定：arm控制角rad、输出Nm、kp为Nm/rad；固定腱模型控制长度m、输出N、kp为N/m，腱系数具有模型长度/角度换算意义。不是已测实物力/刚度；引擎按dampratio生成的实际阻尼数值未经本轮编译模型读取，仍UNKNOWN，不填造数值。表头缩短为“输出限幅”以避免不自然换行。

## D 历史统计

只改变显示：8.405%、15.296%、18.263%、0.495%、12.718%。对应原比例和NPZ SHA仍在上一阶段model_and_episode_audit.json中；本轮不重算或生成实验数据，也不改原统计档案。段落中的Push/Stack描述同步改为百分比。

## E 组合几何

新增plot_assembled_geometry.py。纯解析XML并对STL应用刚体变换，复现Builder将arm根变换设为arm_mount、将hand根平移置零以及在right_hand后追加XYZ(-90,-90,0)度旋转的过程。所有转动关节取零，保留body/geom原四元数，执行mesh scale；重复引用同mesh同变换的可视几何去重。不调用MuJoCo编译、forward或step。未画底座，只画机械臂与手部，图题明确静态关节零位。37个mesh实例及输入Hash见geometry_provenance.json。

原图位替换为实际离线组合图及同姿态手部放大，不将该几何图视为动态或碰撞正确性实验。其他段落保持原结构，只作上述技术说明和显示修正。
