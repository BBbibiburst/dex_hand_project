# 实际损失函数、尺度与训练合同

本报告公式仅定义代码已实现的量。C3根：`outputs/tactile_p43_c3_dictionary_inverse_c0_2026-11-16/`，模型为code/models.py，训练为code/pipeline_cpu.py。数学归纳为L2代码核验，不是新增实验；数据字段见verified_facts.json及checkpoint_metadata.json。

## 1. 两个域各自的概率表示

对d∈{Real,Sim}、X∈R非负^(4×712)，令M_tj=1[X_tj>0]。Encoder输入为拼接[M,log(1+X)/s_enc,d]，两个域使用相同形式、各自参数与尺度，输出mu_d、ell_d∈R^(4×64)，ell限[-8,4]。

$$
 Z_d=\mu_d(X)+\exp\!\left(\frac12\ell_d(X)\right)\odot\epsilon,
 \qquad\epsilon\sim\mathcal N(0,I).
$$

这是给定窗口后的对角Gaussian后验重参数化（models.py第32–33行）。训练采样，确定性监控与读出用mu；没有以相同噪声或潜变量坐标绑定两域。

## 2. Mask时间条件与正值幅值

令h_-1=0，b_0=BOS，t>0时b_t=previous(M_ctx,t−1)，则

$$
 h_t=\operatorname{GRUCell}([z_t,b_t],h_{t-1}),\qquad
 P(M_t\mid z_{\le t},M_{<t})=
 \prod_{j=1}^{712}\operatorname{Bern}(M_{tj};\sigma(l_{tj}(h_t))).
$$

这表达时间自回归及给定隐状态的帧内因子化似然，不暗示逐Taxel空间自回归。C3训练nll时M_ctx是真实Mask，当帧Amplitude条件也是真实M_t；自由读出改为逐帧预测的hard Mask。

Amplitude头以[h_t,current(M_ctx,t)]生成每Taxel两组(eta_k,a_k,b_k)。omega=softmax(eta)，a,b=softplus(raw)+dtype eps，b是**rate**而非scale。以s=s_pos为Decoder正值单位尺度：

$$
 f_X(x\mid h_t,M_{{\rm ctx},t})=
 \frac1s\sum_{k=1}^2\omega_{tjk}
 \frac{b_{tjk}^{a_{tjk}}}{\Gamma(a_{tjk})}
 \left(\frac xs\right)^{a_{tjk}-1}
 \exp\!\left(-b_{tjk}\frac xs\right),\quad x>0.
$$

models.py第28行的实现是先用float64计算Gamma log density及logsumexp，再**减log(s)**，即正确的密度Jacobian。并非仅把目标除以s后忘记恢复概率密度单位。inactive位置先以1替代target再取log，最终由真实active Mask剔除，避免log0；这不是把真实零值改写为训练正值。

确定性自由输出（第22、26行）：

$$
 \widehat X_{tj}=\mathbf1[l_{tj}\ge0]\,
 s\sum_{k=1}^2\omega_{tjk}\frac{a_{tjk}}{b_{tjk}}.
$$

没有从Gamma或Mask分布随机抽样部署输出，也未设置输出幅值上限；混合均值与一次随机样本的语义不同。

## 3. 各损失的归约必须与代码一致

以下针对单窗n，T=4、J=712、D=64，N_+=Σ_tj M_tj：

$$
 L_{{\rm mask},n}=\frac1{TJ}\sum_{t,j}
 \operatorname{BCEWithLogits}(l_{tj},M_{tj}),
$$
$$
 L_{{\rm amp},n}=-\frac{\sum_{t,j}M_{tj}\log f_X(X_{tj})}{\max(N_+,1)},
$$
$$
 L_{{\rm KL},n}=\frac1{2TD}\sum_{t,d}
 \left[\mu_{td}^2+\exp(\ell_{td})-1-\ell_{td}\right],
$$
$$
 L=\frac1B\sum_{n=1}^B
 \left(L_{{\rm mask},n}+L_{{\rm amp},n}+0.001 L_{{\rm KL},n}\right).
$$

依据：models.py第28行BCE取mean((1,2))，正值NLL除每窗active数（最低1），第33行KL同样取mean((1,2))，pipeline_cpu.py第39行再对batch求平均。

**不能把KL写成未除4×64的标准逐维求和再直接沿用0.001**，否则有效权重相差256倍。这个损失也是按Mask全元素均值、正值数均值、latent元素均值组合的加权训练准则，不应无说明地称为严格未加权的窗口总ELBO。

NLL是连续密度负对数，密度可大于1，故NLL为负可以正常；不能将负NLL当作概率非法或训练失败。全零窗amp项为0，Mask项仍有效。正值按窗归一化，再均衡各窗，并非按整个batch全部正值统一平均。

## 4. 两域的尺度合同

pipeline_cpu.py第19–20、28行：

$$
 s_{{\rm enc},d}=\max(\operatorname{std}(\log(1+X_{{\rm fit},d})),10^{-3}),\qquad
 s_{{\rm pos},d}=\max(\operatorname{median}\{x\in X_{{\rm fit},d}:x>0\},10^{-7}).
$$

中位数仅对训练池中严格正值计算。若无正值，代码返回正值尺度1。std为所有训练元素的总体标准差；输入仅除std，不减均值、不按特征独立standardize。

|项目|Real VAE|Sim VAE|
|---|---:|---:|
|拟合池|5153 RealTrain窗|4049 NaturalTrain+1200 PairedTrain窗|
|encoder_scale（元数据）|0.14118273556232452|0.04088745638728142|
|positive_scale（元数据）|0.005002501420676708|1e−7|
|下限作用|正值中位数未触底|合并正值中位数8.784991401e−9触发1e−7下限|
|Validation参与拟合|否|否|

模型内部buffer为float32，元数据1e−7在buffer中为相应float32表示，精确值见checkpoint_metadata.json。两种尺度都不是C0的pooled log1p均值/逐特征标准差，不能混用；也不是Real、Sim共享的物理单位。

C4/amplitude_scale_audit.csv：NaturalTrain正值中位3.78344682e−11，PairedTrain为3.91502661e−4。正值数分别1,836,833和173,351。尺度拟合按汇合元素，而训练batch两类窗口1:1，权重口径不同。共享同一Sim尺度不意味着幅值已对齐；1e−7是Gamma目标单位尺度，不是Encoder直接放大器。下限大于未截底中位数，会减小相对该中位数的除法倍率，但动态范围仍大，不据此判定主因或擅改尺度。

## 5. C3实际训练与Checkpoint选择

|模块|Seed|窗口池|Batch/每Epoch Step|实际Epoch|Best Epoch|全部训练Step|所存Checkpoint Step|
|---|---:|---|---|---:|---:|---:|---:|
|Real VAE|4301|5153 Real|64/81|45|30|3645|2430|
|Sim VAE|4302|4049 Natural+1200 Paired|32+32/127|45|30|5715|3810|

两者参数量各1,094,232，AdamW lr0.001、wd0.0001，梯度范数上限5；实际CPU，2线程。Real训练全部参数；Sim另一次独立训练全部参数，不是Real权重继续训练。模型训练设备来源为model_device_registration.json和pipeline_cpu.py第11行，不混用C0设备切换记录。

每Epoch Real5153访问一次，45轮231885次；直接读取已有coverage计数，所有窗口各45次。Sim每Epoch两类各4064次、总8128次；45轮每类各182880次。Natural实际有15个补齐采样/轮，Paired重复循环以填满127个batch。修正后的coverage_exact.npz计数中Natural每窗45–49次，Paired每窗141–161次。均不是新增窗口或Root。

旧coverage计数因高级索引重复加法少计Natural2次、Paired21次，coverage_counter_audit.json保留旧值和更正。训练本身未改；本轮只读取已有更正文件，未重放采样或训练。

Validation角色：Real780窗/4Session；Natural905窗/10Root；Paired240窗，来自4个Real Session目标；Train的Real为32Session、755Trial，Natural50Root，Paired不能按1200视为独立Root（来源审计30个Real Session，机器人reset复用）。

Real选取Validation NLL+0.001KL最小；Sim选取NaturalVal与PairedVal这两项的各半均值。每Epoch的Train/Val日志均用posterior mean和Teacher Forcing重新计算，不是随机采样训练batch Loss的直接均值。

pipeline_cpu.py第45–50行额外要求epoch≥30才可存最优，patience15、最大100。两者最终最佳30且实际45；Real所有轮中更早的第4轮selection_loss更低，Sim为第26轮。不能把“训练至少30轮”当成必须剔除早期Checkpoint的验收要求；不能在本轮重选。早期权重未保存时，换它的自由重建/域评价为UNKNOWN。

Sim每Epochpaired mask/amp/KL分项未在history中完整保存，虽然selection计算使用了paired KL；不能从paired_NLL一列补造所有分项曲线。完整列名和所存第30/45轮记录见verified_facts.json。

## 6. C13/C24不是继续优化上述全VAE损失

C13仅对真实目标正值优化Gamma-mixture NLL，使用冻结mu、自由运行h及预测Mask缓存，不再更新Encoder、循环骨干、Mask或KL，也没有新增重参数化头；条件修复的训练对象是current和amplitude。

C24在同一正值NLL上增加L_temp=mean_(3×712)|diff log1p(Xhat)−diff log1p(X)|，仅继续更新这两组参数。其lambda=269.8845743604945由RealTrain初始0.1·mean|每窗NLL|/初始L_temp固定获得，不由C0或Validation调权重。**Checkpoint仍按Validation正值NLL选择，不按含时序项总损失选择**；固定18轮，最佳2轮，终点另存，不根据C0重选。

具体版本、参数和结果边界见CH05_54_VERSION_COMPARISON.md。C24增量是顺序保留帧差，不是按秒的物理时间导数。
