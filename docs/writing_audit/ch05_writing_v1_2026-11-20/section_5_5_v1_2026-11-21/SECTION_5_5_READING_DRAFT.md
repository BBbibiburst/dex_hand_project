# 5.5 基于伪配对监督的跨域潜变量映射
双域VAE分别获得域内表示，但相同的潜变量维数并不构成语义对应。为利用统计伪配对建立联系，本文在两套VAE训练完成后固定其参数，以配对窗口的后验均值构造监督。记第$n$组Paired Sim与Real窗口为$(X_{s,n},X_{r,n})$，其均值表示分别为$Z_{s,n}=\mu_s(X_{s,n})$与$Z_{r,n}=\mu_r(X_{r,n})$，均具有$4\times64$形状。Translator学习
$$
 \widehat Z_{r,n}=T_\theta(Z_{s,n}),\qquad
 \widehat X_{r,n}=D_r(\widehat Z_{r,n}),
 $$
其中$\theta$为映射参数，$D_r$为Real Decoder。该监督关系由伪配对确定，不是已知物理接触变换的真值。每组输入、均值目标及Real触觉沿同一配对索引取出，不在训练过程中重新寻找近邻。回归比较发生在Translator输出与Real目标坐标之间，不将未映射的两域向量直接作物理距离比较。

Translator由两层单向GRU和线性投影组成。GRU每帧输入64维，隐状态为128维，输出经128至64维线性层形成预测Real潜变量，总参数量181824。四帧之间递推隐状态，每个窗口以零状态开始，不跨窗口延续；批量输入输出均为$[B,4,64]$。采用GRU旨在使映射利用当前及前序帧信息，尚无同等数据和预算的公平消融证明其优于MLP。此阶段使用固定后验均值，不重复VAE训练时的随机重参数化。

为兼顾潜变量回归与目标域读出，联合目标定义为
$$
 \begin{aligned}
 L_H&=\frac1{B\cdot4\cdot64}\sum_{n,t,d}
 \rho_1(\widehat z_{r,ntd}-z_{r,ntd}),\\
 L_T&=L_H+\frac{0.1}{B}\sum_n
 L_{D_r}(\widehat Z_{r,n},X_{r,n}),
 \end{aligned}
 $$
其中$B$为批量大小，$t$与$d$分别为帧和潜变量维度索引；Huber函数在$|e|<1$时为$\rho_1(e)=e^2/2$，否则为$|e|-1/2$。$L_{D_r}$沿用上一节的Mask BCE与正值Gamma混合NLL之和，保留其归一化和尺度Jacobian，不包含KL。它约束预测潜变量在Real Decoder中的条件重建，而非直接度量跨域分布对齐。

训练数据流如图（伪配对监督训练与自由推理的数据流）所示。两域Encoder仅提供均值缓存，Real Decoder参数也保持冻结，但计算NLL时不切断其对预测潜变量的梯度，因此损失仍可经Decoder回传至Translator，只有$\theta$参与更新。参数冻结意味着不累积其参数梯度，并不意味着该模块退出计算图；映射输出对条件似然的影响仍由固定读出函数传递。Decoder采用真实前帧及当帧Mask条件，即Teacher Forcing，不能将该似然当作自由运行质量。


**图：伪配对监督训练与自由推理的数据流**

- 训练输入：Paired Sim → 冻结Sim Encoder的后验均值 → Translator → 预测Real latent。
- 回归目标：配对Real → 冻结Real Encoder的后验均值 → Huber监督。
- 读出约束：预测Real latent与配对Real → 冻结Real Decoder的真实Mask条件NLL。Huber和NLL梯度回传，只有Translator更新。
- 部署：Natural Sim → Sim Encoder后验均值 → Translator → Real Decoder自由解码 → 完整4×712触觉。


训练和验证分别使用1200与240个配对窗口。随机种子为4303，采用AdamW，学习率$3\times10^{-4}$、权重衰减$10^{-4}$，批量64，梯度范数上限5。每轮打乱训练窗口并执行19步，45轮共855步；按第30轮起的验证联合目标选择权重，最终采用第30轮，对应570次更新。末批包含48窗，验证时按各批样本数加权汇总，与训练采用相同的Huber和NLL组合，不依据域分类器分数挑选检查点。总训练步数与所存检查点经历的更新次数应分别记录。Natural Sim不直接参与该映射的配对训练。部署时仅需Natural输入，经Sim均值编码、Translator和Real Decoder生成完整触觉；Mask由模型预测，幅值取Gamma混合均值，无需Real样本、GMM或探针。

这种映射的有效范围受监督来源限制。GMM责任度不是物理接触归属概率，Top-5分量筛选与单一候选选择压缩了多解信息，Translator未接收完整分量概率或候选分布。1200与240窗也不代表1440个独立Root。已有总体回归误差优于配对训练Real均值模板参照，该模板仅沿1200个配对训练目标的窗口维取均值，保留四帧位置，并广播至待评价窗口，不使用验证目标拟合。然而，部分Session例外；这仅支持有限配对上的映射学习。Natural没有同步对应的Real目标，低潜变量误差不能保证712维输出结构或域分布正确，下一节需以冻结评价器及结构指标共同检验。

