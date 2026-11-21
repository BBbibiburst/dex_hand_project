# 可成立的数学定义与监督关系（材料卡，非正文）

## 1. 对象和域

定义观测空间 \(\mathcal X=\mathbb R_{\ge0}^{4\times712}\)。训练样本另需满足finite条件。\(X_{tj}\)为保留帧t和固定通道j的processed数值响应；16个Patch为索引组织，不是跨设备物理配准证明。当前数据合同支持经验样本集

\[\mathcal D_R^{tr/va}=\{X_{R,n}\},\qquad \mathcal D_S^{tr/va}=\{X_{S,n}\}.\]

其中S明确表示Natural Sim，不把物理探针Paired Sim当作自然部署分布。符号tr/va分别列集合而非合并划分；数量见角色矩阵。

统计伪配对单列为
\[\widetilde{\mathcal P}^{tr/va}=\{(X_{S,p,n},X_{R,p,n})\}.\]

下标p表示由既定生成/准入程序关联的窗口，不表示真实配对概率。该数据只给出代理监督，不是自然Sim和Real的真实联合分布。无需为本节引入未执行OT、随机条件传输或“已学习完整多模态条件分布”的假设。

## 2. 目标与实际映射

\[F:\mathcal X\to\mathcal X,\qquad \widehat X_R=F(X_S).\]

目标为转换后的Natural响应具有Real域统计特征，并尽可能保留输入响应与短时序关系。C3现有确定性实现可抽象为
\[F=D_R^{\rm free}\circ T_\theta\circ\mu_S.\]

此式只是接口组合：Sim后验均值、Translator、Real Decoder自由解码。不是新增目标函数，不要求5.2展开网络。Real Encoder用于监督端，不是部署必需入口；Sim Decoder不参与最后Real读出。代码inference.py:15–19及models.py:18–33支持。

输出仍为四帧完整712通道，Decoder按t=0,1,2,3递推并堆叠，不重排。**保留帧序不等于内容结构自动保持**。窗口内时间间隔与跨域频率不能默认相同。

Real幅值头的混合均值乘回Real positive_scale，恢复训练目标processed数值尺度；Encoder的log1p/尺度变换不是最终输出单位。可称“Real域风格的触觉重建/转换”，不可称“恢复该Sim样本对应的实测真实触觉”。尺度恢复不代表分布已匹配，更不代表压力/力学标定。

## 3. 三个学习层次

1. **域内自重建**：Real VAE重建Real；Sim VAE重建Natural和Paired Sim各自输入，参数独立。
2. **代理监督映射**：Paired Sim后验均值经T预测Paired Real后验均值。相同4×64形状不赋予共同语义；独立坐标可由监督函数连接，但可靠性受配对歧义、覆盖和验证角色限制。
3. **完整触觉读出**：预测Real latent进入固定Real Decoder；自由运行用预测Mask及恢复尺度的Gamma混合均值。

训练Huber度量Real潜坐标中预测与代理目标的误差。Decoder NLL含真实前帧Mask递推、真实当前帧Mask幅值条件和当前Mask头BCE；冻结参数不切断对Translator的输入梯度。Huber+0.1 NLL是训练/选权重目标，不是Natural输出正确性的度量，不是直接跨域分布对齐。

## 4. 成立和不成立的假设

|可成立|不能据此推出|
|---|---|
|相同数值shape和固定索引|同物理Taxel、同传感器面积、同单位|
|域内重建与监督映射实现存在|真实公共语义坐标已经建立|
|Paired按索引关联|同步接触真值或Natural有真实对应目标|
|输出非负finite、完整维度|无坍塌、输入接触保持、真实任务有效|
|有限模型下域混淆目标|总体分布相等、因果一致或完整物理迁移|

没有同步Natural–Real目标，因此不能定义可由现有数据直接求得的Natural对真实目标逐点MSE。域内自重建MSE、Paired代理误差、输入输出依赖和跨域统计距离分别报告，不混为同一个“真实重建误差”。
