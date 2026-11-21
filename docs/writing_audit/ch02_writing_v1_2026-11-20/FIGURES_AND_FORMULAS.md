# 图表与公式来源

图按TeX实际顺序编号：9个figure，7个table，5个equation环境。正式章号和页面未编译。

|图|内容|来源及状态|
|---|---|---|
|1|五层架构|builder/factory/rl_env/production_recorder实际接口；TeX框图|
|2|组合模型|明确待补图位；XML/STL已定位，未借用异版本截图|
|3|驱动拓扑|当前arm/hand XML；TeX关系图|
|4|多速率控制|arm.py、rl_env.py、指定profile；TeX流程图|
|5|五指与手掌索引|fitting/layout.py；索引组织图，不冒充整手几何图|
|6|指尖STL与7×8采样|当前skin_0_2_p.STL及fingertip_ellipsoid_grid_points；离线PDF/PNG|
|7|局部采样坐标|相同采样点及生产_taxel_frame；外向修正前的三轴|
|8|处理链|signal_processing.py及profile；高斯/噪声注明关闭|
|9|验证层次|五条已有NPZ及材料核验JSON；不表示重新执行任务|

几何图可用`PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /home/zmy/miniconda3/envs/mujoco/bin/python docs/writing_audit/ch02_writing_v1_2026-11-20/plot_geometry.py`复现。脚本仅读取STL并拟合/绘图，不创建或推进MuJoCo物理模型。坐标为mesh局部坐标，不能当作完整手部世界姿态或Real电极位置。

7表依次为模块、执行器、控制参数、布局参数、16Patch索引、信号阶段、五Episode统计；来源为材料核验目录model_and_episode_audit.json/source_inventory.json与对应原始文件。图表中的技术参数不是新增实验测量。

|公式标签|内容|代码依据|
|---|---|---|
|eq:ch02_tendon|固定腱关节线性组合|当前dex_hand.xml tendon/fixed|
|eq:ch02_ik|加权Jacobian的SVD阻尼伪逆及姿态项|source/control/arm.py::_solve_ik；自适应/限幅在正文解释|
|eq:ch02_bezier|张量积Bezier及加权ridge求解|fitting/bezier.py::fit_bezier_surface/evaluate_bezier_surface|
|eq:ch02_site_size|邻距和overlap决定box半宽|dex_hand.py::_neighbor_spacing/SimpleBoxTactileSensor._add_taxel|
|eq:ch02_processing|死区、饱和、幂响应及IIR|signal_processing.py::process；reset边界在正文说明|

未添加Hertz/FEM材料模型、显式推杆闭链或未实现控制律。第二章仅解释当前模拟读出，不声称实物传感器标定。
