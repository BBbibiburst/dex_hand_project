# 图表计划

图表是阶段3候选，不是已生成的正式论文图。禁止AI合成截图或压力真值。

|编号|目的|来源|可生成性/限制|
|---|---|---|---|
|图2-1|模块架构与控制/数据流|builder、factory、rl_env、record代码|可据已核验接口画框图；不是实际截图|
|图2-2|组合模型与安装|arm/hand XML、STL、body变换|现有截图须先查来源；本轮不启动渲染仿真|
|图2-3|关节—腱—执行器关系|model_and_episode_audit.json|可离线画7关节+12手关节/6腱，不能画虚构推杆|
|图2-4|控制周期与物理子步|controller、env、profile|可离线画20Hz/1kHz限定配置流程|
|图2-5|曲面阵列总体布局|16个STL、geom变换、fitting调用|可离线计算；本轮未生成，不用旧异版本图冒充|
|图2-6|局部Taxel与box感知区|_taxel_frame/_add_taxel、STL|可离线绘制，标局部坐标与感知体积|
|图2-7|外壳拟合/Bezier/mesh-uv|segment/fingertip/projection|可离线可复现图，不声称物理面积积分|
|图2-8|raw→processed|signal_processing.py、profile|可画流程，Gaussian/noise标该profile关闭|
|图2-9|功能记录核验流程|五条NPZ/result/manifest|可画读取—shape—数值—时间检查；不是本轮执行任务流程|

## 表格

- 表2-1 模块及功能：builder/control/sensors/tasks/recorder；不展开第四章存储设计。
- 表2-2 执行器：从JSON完整列7个arm、6个hand传动坐标/range/kp/force；物理关节另列，区分rad与腱长度。
- 表2-3 控制频率及参数：代码默认、profile、历史记录三列，未知不填默认。
- 表2-4 16 Patch的rows/cols/kind/start/stop：使用实际JSON，接口名与实际拟合策略并列。
- 表2-5 raw/processed/image层级：维度、dtype、变换、单位与限制；可视化色标不是采集幅值缩放。
- 表2-6 五条指定sensor_tactile：T、finite、非负、非零率、20Hz、历史success及证据等级。
- 表2-7 功能核验边界：历史记录存在/本轮数组检查/当前物理动态UNKNOWN/硬件标定UNKNOWN。

不必为满足编号重复制作热图。若展示触觉，使用各Patch原4×8、7×8、7×16方阵统一色标，不把712向量拉成长条。多时刻来自同一源Episode，标timestamp与字段名，不混用tactile和sensor_tactile。几何图标明仿真内部坐标，未认证Real对应关系。
