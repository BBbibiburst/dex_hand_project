# 混合统计量、s/d/e与候选选择精确定义

唯一实际入口：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/expand.py`第14–16、77行→`matching.py::inherited_metrics`第3–9行。pilot inverse.py同样调用。以下描述冻结实现，不修改指标。

## 1. 记号和数值下限

r_{tpj}、v_{tpj}分别表示Real与候选Sim在时刻t=0,1,2,3、Patch p、Taxel j的非负processed响应；q_{pj}为该Patch固定归一化索引位置。为避免与GMM责任度重名，本报告把Real响应记作r，不把它当后验概率。令ε=10⁻¹²。

$$
 R_{tp}=\sum_jr_{tpj},\quad S_{tp}=\sum_jv_{tpj},\qquad
 p^*=\arg\max_p\sum_{t=0}^3R_{tp}.
$$

主Patch按Real四帧总量选定，不按Sim选，也不在不同帧分别切换；NumPy argmax遇并列取第一项。入口不额外将四帧先平均为一帧评分。

## 2. 冻结GMM混合统计量：权重为π，不是目标分量响应量m

matching.py第6行读取g['components'][0,p*]及同一ALL模型的weights/means/covariances。省略主Patch下标：

$$
 \bar{\boldsymbol\mu}=\sum_k\pi_k\boldsymbol\mu_k,
 \qquad
 \boldsymbol C_G=\sum_k\pi_k\left[
 \boldsymbol\Sigma_k+(\boldsymbol\mu_k-\bar{\boldsymbol\mu})
 (\boldsymbol\mu_k-\bar{\boldsymbol\mu})^{\mathsf T}\right].
$$

这是含“分量内+分量间”项的完整混合协方差，不是仅Σπ_kΣ_k；π来自冻结拟合模型，不是expand.py按当前Real窗口算出的m_pk，也不是m_pk归一化后的权重。协方差来自训练偏移坐标，平移不改变理论协方差，但不因此认证其等于目标Real的实际响应协方差。

## 3. s：四帧Patch比例余弦

matching.py第4–5行：

$$
 P^r_{tp}=\frac{R_{tp}}{\max(\sum_{p'}R_{tp'},\varepsilon)},
 \quad P^s_{tp}=\frac{S_{tp}}{\max(\sum_{p'}S_{tp'},\varepsilon)},
$$
$$
 s=\frac14\sum_{t=0}^{3}
 \frac{\langle\boldsymbol P^r_t,\boldsymbol P^s_t\rangle}
 {\max(\|\boldsymbol P^r_t\|_2\|\boldsymbol P^s_t\|_2,\varepsilon)}.
$$

输入每帧16维Patch总量向量，先除以各域该帧全手总量，再算余弦，最后四帧等权平均。任一比例向量为全零时该帧余弦为0；包括双方全零在内，不忽略、不把空帧视为相似度1。通过条件s≥0.8。极小总量受ε下限影响，不无条件声称所有比例之和精确为1。

## 4. d：双方活跃帧的质心—锚点距离

在p*上，matching.py第7–8行：

$$
 \boldsymbol c^r_t=\frac{\sum_jr_{tp^*j}\boldsymbol q_j}{\max(R_{tp^*},\varepsilon)},
 \quad
 \boldsymbol c^s_t=\frac{\sum_jv_{tp^*j}\boldsymbol q_j}{\max(S_{tp^*},\varepsilon)},
$$
$$
 \boldsymbol a_t=
 \begin{cases}\boldsymbol c^r_t+\bar{\boldsymbol\mu},&R_{tp^*}>0,\\
 \boldsymbol0,&R_{tp^*}\le0,
 \end{cases}
 \qquad I=\{t:R_{tp^*}>0\ \text{且}\ S_{tp^*}>0\}.
$$

若I非空，d=|I|⁻¹Σ_{t∈I}‖c^s_t−a_t‖₂。只对双方活跃帧平均，不对4帧统一除4，也不除以Real质心范数或Patch直径。I为空时返回center=None，登记失败；结构代价中使用替代值d_sort=1。正文的d在排序公式里包含这一替代约定，不能把1当作实际测得的质心误差。有效d通过条件为d≤0.2。

## 5. e：绝对协方差差值，无相对误差分母

matching.py第8行先在Sim主Patch上按响应量归一化：

$$
 w^s_{tj}=\frac{v_{tp^*j}}{\max(S_{tp^*},\varepsilon)},\qquad
 \boldsymbol C^s_t=\sum_jw^s_{tj}
 (\boldsymbol q_j-\boldsymbol c^s_t)(\boldsymbol q_j-\boldsymbol c^s_t)^{\mathsf T},
$$
$$
 \boxed{e=\frac14\sum_{t=0}^3\|\boldsymbol C^s_t-\boldsymbol C_G\|_F.}
$$

实际代码：

```python
scov = np.einsum('nt,nti,ntj->nij',
    sim[:,slices[p]]/np.maximum(smass[:,p,None],1e-12), dd, dd)
error = float(np.linalg.norm(scov-targetcov,axis=(1,2)).mean())
```

**不存在除以‖C_G‖_F、trace(C_G)、Real协方差或其他相对误差尺度的操作。** “相对GMM协方差”只能指比较对象，容易误读为relative error，修订稿改为“与该混合协方差之差”。响应总量分母用于构造空间加权矩，并不是e的相对误差分母。

Sim该帧全零时权重全零、C^s_t=0，但该帧仍贡献‖C_G‖_F，并未按d的双方活跃集合过滤。若0<S<ε，权重总和小于1，代码仍按上述下限执行，不改成理论上的严格概率协方差。通过条件e≤0.1。e在索引坐标平方尺度上，是绝对矩差，不赋予Pa或真实面积含义。

## 6. gate、提前停止、argmin

`passed = sim.any() and center is not None and center<=.2 and cos>=.8 and error<=.1`。重复检测不在这个gate内。

expand.py第64–82行：

1. trial0：使用字典位置、深度插值及固定Real四帧加载profile。
2. trial1：重新从初始specs复制配置，依据上一个候选输出计算深度反馈。
3. trial2：仍从初始specs复制，依据上一个候选再次计算深度反馈，再将row/col分别沿质心差符号最多改一格并限位。可同时改两轴形成对角邻居；不是对trial1深度进行无记录累乘。

任一候选score.passed为True立即break；尚未执行的trial不进入搜索集合。已有同条件Episode可以直接读历史输出，不为重复构造而新运行。

对实际已经取得输出的集合A_exec，定义d_sort=d（有效时）或1（无双方活跃帧时）：

$$a^*=\arg\min_{a\in A_{\rm exec}}(1-s_a+d_{{\rm sort},a}+e_a).$$

选优在**全部已执行候选**中，而非只在已过门槛集合中；Python min遇代价相同保留先出现者。然后以所选score.passed且生成窗不重复作为最终接受条件。它只保证已执行集合内结构代价最小，不保证潜在全部位置/加载条件或最多3个设想候选的全局最优。

两种边界由代码可知但本轮没有量化发生率：统计已通过但重复的候选也会触发提前停止；先前未通过某项门槛的候选可能有更低加和代价，最终被选出而拒收，即使执行集合中还有通过者。不能把源码中的这种可能性写成已确认的主要失败原因。

本报告只描述现有指标，没有补充归一化分母、改变阈值、重算评分或重新选择候选。
