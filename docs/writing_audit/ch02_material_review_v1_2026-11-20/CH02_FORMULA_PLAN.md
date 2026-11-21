# 公式计划与实现约束

使用现有术语：raw为模拟法向力累加，processed为后处理标量；空间坐标单位m，关节rad；不能将下列量都称压力。这里只列可写公式，不写论文正文。

## F1 坐标与运动学（2.2/2.3）

\(p_b=R_{bg}p_g+t_{bg}\)，局部点、旋转3×3、平移3；依据dex_hand.py::_transform_points和builder安装变换。世界位姿由body链复合，可写\(T_{we}(q)=\prod T_i(q_i)\)为MJCF链的记号，不能虚构DH参数表。

\(\dot x=J(q)\dot q\)，arm q∈R⁷、末端速度六维；实际mj_jacSite获取平移和旋转Jacobian，位置m、旋转rad。不是本项目新算法。

## F2 实际IK（2.3）

\(J_w=[w_p J_p; w_r J_r]=U\Sigma V^T\)，\(e_w=[w_p e_p;w_r e_r]\)。代码默认权重1和0.35；旋转误差来自目标四元数乘当前共轭后转rotvec，不是Euler角相减。

\(J_\lambda^+=V\operatorname{diag}(\sigma_i/(\sigma_i^2+\lambda^2))U^T\)，\(\Delta q=J_\lambda^+e_w+(I-J_\lambda^+J_w)k_n(q_0-q)\)。第二项只在use_nullspace启用时使用；阻尼投影不是严格正交零空间，不能声称绝对不影响主任务。

更新\(q\leftarrow\operatorname{clip}(q+\operatorname{clip}(\Delta q,-s,s),q_{min},q_{max})\)，之后还有_smooth_joint_target速度约束与滤波；不得只写未受限伪逆当完整控制器。SVD失败分支退回\(J^T(JJ^T+\lambda^2I)^{-1}e\)。posture_weight>0时Jacobian/误差另增正则姿态行，默认0。

依据source/control/arm.py::_solve_ik/_adaptive_damping/_smooth_joint_target。自适应阻尼阈值和滤波应由代码表表达，无需再推导标准理论。

## F3 固定腱传动与伺服（2.2/2.3）

\(l_t=\sum_j c_{tj}q_j\)，当前MJCF tendon/fixed；六行系数见JSON。c采用模型长度/角度约定，不是测得的物理推杆尺寸。控制输入为腱位置目标而非手指所有关节独立目标。

可按MuJoCo position actuator文档解释\(f_a=k_p(u-l)-k_v\dot l\)，其中kv由dampratio等引擎规则给定，最终有force限幅。此为执行器层说明，不是自编通用PD控制器；未经编译不填实际kv数值，不把腱力与关节力矩混为同一单位。

## F4 Bezier拟合及参数采样（2.4）

\(S(u,v)=\sum_{i=0}^{d_u}\sum_{j=0}^{d_v}B_i^{d_u}(u)B_j^{d_v}(v)C_{ij}\)。控制点三维m、Bernstein参数无量纲。fit_bezier_surface构造设计矩阵A并解
\((A^TWA+\rho I)C=A^TWP\)，\(\rho=\eta\max(\mathrm{tr}(A^TWA)/n_c,10^{-12})\)。不是固定未缩放ridge；指尖有端点权重10，segment无该额外加权。

segment参数中心\(u_r=(r+1/2)/R,v_c=(c+1/2)/C\)；指尖u先在[0.06,1]分格再取中心。掌心用投影三角形重心权重\(p=\sum_i\beta_i v_i\)，\(\sum_i\beta_i=1\)，需交代多层投影及回退策略，不声称等真实面积。

## F5 朝向与感知区（2.4）

由列差分\(a\)归一化得\(e_x\)，行差分b作\(e_y=\mathrm{normalize}(b-(b^Te_x)e_x)\)，\(n=e_x\times e_y\)，\(R=[e_x,e_y,n]\)，再按实现外向启发式翻转。边界/退化情况按函数保护，不称精确网格法线。

\(h_x=\max(h_{min},\frac12\gamma d_c)\)，\(h_y=\max(h_{min},\frac12\gamma d_r)\)，d取对应邻距最小值，hz来自depth配置。box截面4hxhy只是感知区几何量，不是实物Taxel电极面积。

## F6 raw与processed（2.5）

\(r_j=\sum_{k\in\mathcal C_j}f^n_k\)。C由原生touch的body、site区域及引擎重投影规则决定；重叠C可能重复，不可把sum_j r_j当总力。原生无面积除法、无三轴切向分量。

\(a=\operatorname{clip}(r-d,0,s)\)，\(b=s(a/s)^p\)，\(v_t=\alpha b_t+(1-\alpha)v_{t-1}\)。初态按reset(size)路径，不能默认每段独立都从首帧初始化。

Patch串扰\(w_i=(1-c)v_i+c\sum_{j:i\in N(j)}v_j/|N(j)|\)，N为可用8邻域，注意按源节点邻居数归一化。可选高斯\(g_k\propto\exp[-k^2/(2\sigma^2)]\)，radius=max(1,ceil(3sigma))、一维核归一化、两轴顺序卷积、edge填充。指定profile sigma=0未启用。

有噪声时添加高斯噪声并clip，normalize时除s，最终float32。不能把Gaussian后处理写成原生接触力渲染，也不把normalized输出写N/Pa。

## 不应写入

未实现的Flex/FEM硅胶本构、Hertz接触压力推导、touch_grid插件核、真实Taxel面积积分、剪切三分量、虚构DH/推杆闭链方程、未标定Real/Sim通道等价、跨域网络损失。它们均非本章当前实现证据。
