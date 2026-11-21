# 文献与官方技术资料核验

日期2026-11-20。本轮不修改reference.bib。共9项：5篇已核验论文、3项MuJoCo官方文档主题及1项作者教学资料。论文复用1.2核验档案，不声称本轮全文重读；详细作者/题录/原PDF/阅读页码同步保存在reference_records.json。官方在线资料本轮核对相关段落，stable版本不能替代历史运行版本证明。

## S06 MuJoCo: A physics engine for model-based control

作者：Emanuel Todorov; Tom Erez; Yuval Tassa。2012，IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS):5026–5033。DOI：10.1109/iros.2012.6386109。

原文：https://roboti.us/lab/papers/TodorovIROS12.pdf；本地：`thesis/docs/writing_audit/ch01_1_2_sources/S06.pdf`。阅读范围：引擎摘要、求解流程、模型元素和计时实验开头，PDF页[1, 3, 7]（沿用已审核阅读记录，不标为本轮全篇阅读）。

用途：2.1/2.3物理引擎背景；实际版本参数仍以项目代码为准。

## S08 TACTO: A Fast, Flexible, and Open-Source Simulator for High-Resolution Vision-Based Tactile Sensors

作者：Shaoxiong Wang; Mike Lambeta; Po-Wei Chou; Roberto Calandra。2022，IEEE Robotics and Automation Letters, 7(2): 3930–3937。DOI：10.1109/LRA.2022.3146945。

原文：https://arxiv.org/abs/2012.08456；本地：`thesis/docs/writing_audit/ch01_1_1_sources/L09_tacto.pdf`。阅读范围：复用1.1方法与位姿估计实验阅读，本轮复查流程/计时/背景校正，PDF页[3, 4, 7]（沿用已审核阅读记录，不标为本轮全篇阅读）。

用途：2.4/2.5区别视觉触觉渲染和原生标量touch，不将其成像模型写成本项目实现。

## S09 Taxim: An Example-Based Simulation Model for GelSight Tactile Sensors

作者：Zilin Si; Wenzhen Yuan。2022，IEEE Robotics and Automation Letters, 7(2): 2361–2368。DOI：10.1109/LRA.2022.3142412。

原文：https://arxiv.org/abs/2109.04027；本地：`thesis/docs/writing_audit/ch01_1_1_sources/L10_taxim.pdf`。阅读范围：复用方法阅读；本轮核对形变标定、受控压入、图像实验及CPU限制，PDF页[2, 3, 5, 6]（沿用已审核阅读记录，不标为本轮全篇阅读）。

用途：2.4/2.5传感器响应/形变/读出的区别，不能声称已复现GelSight模型。

## S10 Sim-to-Real for High-Resolution Optical Tactile Sensing: From Images to Three-Dimensional Contact Force Distributions

作者：Carmelo Sferrazza; Raffaello D'Andrea。2022，Soft Robotics 9(5):926–937。DOI：10.1089/soro.2020.0213。

原文：https://journals.sagepub.com/doi/10.1089/soro.2020.0213；本地：`thesis/docs/writing_audit/ch01_1_2_sources/S10.pdf`。阅读范围：预印本v3摘要、FEM方法、相机映射、测试指标与可视化选段，PDF页[1, 3, 4, 7, 9, 11]（沿用已审核阅读记录，不标为本轮全篇阅读）。

用途：2.5物理单位与标定边界，仅简短背景，不把其FEM/3D接触力输出作为本项目能力。

## S25 Sim-to-Real for Robotic Tactile Sensing via Physics-Based Simulation and Learned Latent Projections

作者：Yashraj Narang; Balakumar Sundaralingam; Miles Macklin; Arsalan Mousavian; Dieter Fox。2021，IEEE International Conference on Robotics and Automation (ICRA):6444–6451。DOI：10.1109/icra48506.2021.9561969。

原文：https://research.nvidia.com/publication/2021-06_sim-real-robotic-tactile-sensing-physics-based-simulation-and-learned-latent；本地：`thesis/docs/writing_audit/ch01_1_2_sources/S25.pdf`。阅读范围：摘要、两VAE结构、监督投影、FEM及信号/接触重建实验，PDF页[1, 3, 4, 5, 6]（沿用已审核阅读记录，不标为本轮全篇阅读）。

用途：2.5传感器模型差异边界，可选引用；潜空间迁移详论已放1.2/第五章，不在本章重复。

## D01 MuJoCo Contributors：XML Reference，sensor/touch

官方链接：https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-touch 。在线持续更新资料，无固定期刊/DOI，访问2026-11-20。本轮阅读touch段：空间区域、body关联、软接触重投影、法向力累加及非负标量。对应2.5；配合本项目P40封存engine_sensor.c，不凭文档推断服务器实际版本。

## D02 MuJoCo Contributors：Computation，Soft contact model

https://mujoco.readthedocs.io/en/stable/computation/index.html 。本轮阅读contact模型概述及算法目的；对应2.3，引擎柔顺接触不是Flex软体。无固定出版年份/DOI，不编造。

## D03 MuJoCo Contributors：XML Reference，position actuator / fixed tendon / mesh

https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position 。本轮核对position的dampratio以及mesh相关属性；结合XML说明执行器层与几何语义。与D01同一文档的不同主题，正式BibTeX可合并为一个官方文档条目，不应重复充数。fixed tendon实际系数由本项目XML证明，未把第三方资料当模型来源。

## D04 Kevin M. Lynch、Frank C. Park：Modern Robotics教学资源，6.2 Numerical Inverse Kinematics

Cambridge University Press配套作者资料；教材2017，网页年份不擅填。官方链接：https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/ 。本轮阅读网页Transcript中Newton–Raphson、Jacobian伪逆、冗余和迭代初值的说明。仅支持2.3背景；实际SVD阻尼/零空间配置以arm.py为准，不将网页当本项目DLS实现证明。未阅读全文教材，不编造教材DOI。

## 题录和引用处理

S06/S08/S09/S10/S25沿用1.2记录，正式写作先复用现有bib中的对应标题条目；本轮不生成新的键或改元数据。官方资料仅记录URL和访问日期，是否合并条目由后续任务书授权。无需为凑6–12项引入未读的欠驱动机构论文。

RM75B和Dex Hand模型的厂商原始CAD出处、参数认证、手部结构论文与硬件版本仍UNKNOWN；此缺口由项目资产描述明确披露，不能用相似产品网页补证。近似拟合曲面和site几何是项目代码证据，不自动具有独立论文级新颖性。
