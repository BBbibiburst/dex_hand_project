# GMM训练与逆构造坐标合同专项复核

结论：**代码确有坐标原点语义不一致，不是正文漏写了一步已有的中心化。** 保留责任度公式中的固定网格q，并在正文明确这一限制；不虚构q−目标质心或μ+目标质心的变换。原责任度公式忠实于代码，不应为追求理论形式而改写成未实现版本。

路径简称：G=`outputs/tactile_gmm_dual_vae_2026-11-16/code/`；C=`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/code/`。

## 1. 训练坐标及质心

G/gmm.py第5–6行：列和行分别用linspace(-.5,.5,cols/rows)，meshgrid后按行展平为二维q=(列轴,行轴)。这是固定索引归一化，不是毫米/世界坐标，也没有按触觉幅值标准化坐标轴。

G/fit_gmm.py第23–25行：只从P37/train.npz取Real、Session及Stage，用该Patch训练正值P5作筛选阈值。第13–15行给出实际中心化：

```python
a = frames[i].astype('float64')
a = np.where(a > threshold, a, 0)
center = (a[:,None] * xy).sum(0) / a.sum()
chosen = rng.choice(len(a), size=16, replace=True, p=a/a.sum())
points.extend(xy[chosen] - center)
```

因此，先令a_tj=x_tj·1[x_tj>τ_p]，c_t=Σa_tj q_j/Σa_tj，再采样u_tj=q_j−c_t。质心用阈值处理后的同一帧响应，而非整个窗口均值，也不是无权Taxel几何均值。原网格q落在[-.5,.5]²，不代表偏移u也被限制在该区间；代码没有把偏移重新缩放或裁剪回网格。混合参数μ、Σ描述这些偏移样本。

第11、18–20行限制每Session取帧、轮转抽样，改变的是样本构成，不负责恢复绝对位置。

## 2. 逆构造没有目标中心化

C/expand.py第6–8行重新创建相同数值范围与行列顺序的固定网格；第42、44–51行：

```python
w = x[:,sl].mean(0)
delta = xy - means[kk]
responsibility = np.exp(lp - logsumexp(lp, axis=1)[:,None])
ww = w * responsibility[:,kk]
center = (ww[:,None] * xy).sum(0) / ww.sum()
```

pilot的C/inverse.py采用相同路径。责任度求值用q，而非q−c_target；没有将μ平移为μ+c_target。两处变量虽然都是二维数值，训练μ的语义是“相对响应质心的偏移”，求值q却以Patch索引中心为原点，不能因维度和网格范围相同就认为坐标语义已一致。

输出中心c_pk=Σw_j r_jk q_j/Σw_j r_jk直接由绝对索引q重加权计算。**它不是对μ做加回质心的坐标转换**，也不能被解释成由相对分量精确恢复的真实接触中心。除非特定数据/参数具有相应特殊结构，该实现没有一般的平移一致性保证。本轮没有运行平移测试或评估其对配对质量的贡献，影响大小为UNKNOWN。

## 3. 字典检索和物理位置

C/expand.py第10–13行moments用同一个固定网格计算字典响应质心；第55–60行以其到c_pk的欧氏距离检索。因此**目标加权中心和字典响应中心在相同的数值索引坐标中比较**，但这不会修复前一步GMM训练/责任度的原点不一致。

检索输出是字典条目的Patch、row、col等条件。C/inverse_physics.py第47–48行通过site_name查仿真site，使用d.site_xpos、d.site_xmat的实际世界位置与朝向；第59–62行据此构造球心目标。不存在把归一化q直接当米坐标使用的步骤。这里可追溯的是仿真内部映射，不能认证Real硬件Taxel的物理对应。

## 4. 不能混用的两处“加回中心”

1. G/gmm.py第16–17、29行的GMMRenderer使用kernel(model,shape,anchor)，其中均值变为μ+anchor。这是早期直接统计渲染分支，不是C3物理探针责任度路径，不能拿来补写C3中缺失的变换。
2. C/matching.py第6–7行在评价时构造“Real逐帧质心+冻结混合均值”锚点。这是候选d的参考，不参与expand.py第46行的责任度计算，同样不能当作其已经中心化的证据。

## 5. 正文修订及限定

- 不改实际责任度公式q→q−c，不新增代码不存在的反变换。
- 明确“训练中心化、逆构造未中心化；分量中心为原网格重加权结果”。
- 将原稿“通过统计启发式连接”的宽泛描述收紧为明确的原点语义不一致和平移一致性不获保证。
- 不据此否定所有GMM或宣布整个方法无效。它是当前实现局限，尚无本轮实验隔离其贡献。

仅只读代码、登记文件和既有Train/Validation候选元数据；没有拟合、运行模型、MuJoCo或访问Test/Holdout样本。
