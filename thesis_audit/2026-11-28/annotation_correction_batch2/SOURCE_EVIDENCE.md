# 实现与图示证据对应

以下行号由本轮文件直接定位；代码仅静态阅读，没有导入控制/仿真执行入口。证据为L2静态实现及既有历史数组，不产生新实验L3。

|论点/图示|来源|行号|

|---|---|---:|

|组合控制接口|`source/control/composite.py`|16|

|关节与末端模式|`source/control/arm.py`|264|

|临时IK状态与SVD|`source/control/arm.py`|326|

|速度滤波与限幅|`source/control/arm.py`|432|

|手部传动坐标|`source/control/end_effectors.py`|174|

|控制更新与物理推进|`source/envs/rl_env.py`|207|

|皮肤接触参数|`assets/grippers/dex_hand/dex_hand.xml`|20|

|传感处理顺序|`source/sensors/tactile/signal_processing.py`|162|

|源单元邻域分配|`source/sensors/tactile/signal_processing.py`|51|

|指定机器人profile|`configs/robot_profiles/rm75b_dex_hand.json`|6|

|历史轨迹字段|`outputs/five_task_p32_b_2026-11-03/lift_sugar_rgb_00/source_episode/manifest.json`|239|


## 参数解释的官方依据

接触维数与摩擦配置：[MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-condim)。condim4包含法向、两个滑动与一个绕法向扭转方向；滚动方向用于condim6。

正值solref：[MuJoCo solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)。本XML(0.01,1)按时间常数与阻尼比解释。正文复用已有mujoco2026xml/mujoco2026computation键，未新增或修改文献。


## 轨迹的统计身份

单条P32历史Sugar Box Lift记录，447帧；命令、物体位置、接触数与阶段均来自原NPZ。manifest中的metadata.stage_codes及metadata.online_contract.position_action_layout用于字段解释。无模型加载、无新任务执行、无末端测量精度或新成功率推断。
