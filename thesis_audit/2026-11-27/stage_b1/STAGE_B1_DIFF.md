# Stage B1最终修改前后对照

所有原文来自本工作目录before备份，修改后文本直接读取当前活动源码。以下为最终版本，不包含中间措辞。所有条目仅作叙述消歧、范围限定或注释更新，无新增模型、实验或验收结论。

## 1. A04｜thesis/body/ch01_introduction.tex:98

修改原因：对应三条主线与四项工作，保持原有归类层级及贡献范围。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第98行起）：

```latex
本文围绕仿真操作、示教数据和触觉重建开展研究。主要工作包括平台与观测集成、适应具体机器人约束的示教流程，以及有限统计监督下的跨域方法探索。三者分别提供实验条件、可追溯数据和重建分析依据，其贡献需要结合实现与验证范围说明，平台集成、开源适配和迁移效果不作同一层次的创新判断。
```

修改后（第98行起）：

```latex
本文围绕仿真操作、示教数据和触觉重建开展研究。主要工作包括平台与观测集成、适应具体机器人约束的示教流程，以及有限统计监督下的跨域方法探索。三者分别提供实验条件、可追溯数据和重建分析依据，其贡献需要结合实现与验证范围说明，平台集成、开源适配和迁移效果不作同一层次的创新判断。上述三条主线对应四项工作：平台与触觉观测、抓取候选与专家示教、多模态示教数据与质量控制、有限监督触觉重建与评价。其中第二、三项同属专家示教与数据主线，第七章据此分项总结。
```

## 2. A10｜thesis/body/ch02_simulation_platform.tex:296

修改原因：统一仿真Patch内邻域扩散术语及实现参数名，不改变算子或数值。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第296行起）：

```latex
  $\rightarrow$ IIR滤波 $\rightarrow$ Patch内邻域串扰\\[5pt]
```

修改后（第296行起）：

```latex
  $\rightarrow$ IIR滤波 $\rightarrow$ Patch内邻域扩散\\[5pt]
```

## 3. A10｜thesis/body/ch02_simulation_platform.tex:315

修改原因：统一仿真Patch内邻域扩散术语及实现参数名，不改变算子或数值。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第315行起）：

```latex
其中，$t$为连续调用的观测索引，$\mathcal C_c$为下述Patch内串扰算子，$\boldsymbol y_t$为processed向量。末行适用于当前关闭高斯滤波和噪声、开启归一化的profile：先完成串扰及范围限制，再除以饱和尺度$s$，最后转换为float32。滤波初态由处理器reset路径决定，不能假定任何数据片段都以相同方式独立初始化。指定profile使用$d=0.002$、$s=1$、$p=0.7$、$\alpha=0.45$，并开启归一化。死区和饱和设置改变了弱响应及尾部范围，幂变换与时间滤波又改变幅值和相邻观测关系，因此处理后的值不再保留原始模拟力的直接单位含义。

邻域串扰在每个Patch恢复的二维阵列中进行。系数$c=0.12$表示每个源单元保留$1-c$的响应，其余部分按可用的8邻域数量平均分配。边界单元的邻居少于内部单元，程序按源单元归一化，不能写成所有接收单元统一除以8。串扰只作用于同一Patch，数组中相邻但属于不同Patch的两个索引不会据此交换响应。当前profile的gaussian\_sigma和noise\_std均为0，高斯滤波与随机噪声属于可选能力，并未在该配置中启用。
```

修改后（第315行起）：

```latex
其中，$t$为连续调用的观测索引，$\mathcal C_c$为下述Patch内邻域扩散算子，$\boldsymbol y_t$为processed向量。末行适用于当前关闭高斯滤波和噪声、开启归一化的profile：先完成邻域扩散及范围限制，再除以饱和尺度$s$，最后转换为float32。滤波初态由处理器reset路径决定，不能假定任何数据片段都以相同方式独立初始化。指定profile使用$d=0.002$、$s=1$、$p=0.7$、$\alpha=0.45$，并开启归一化。死区和饱和设置改变了弱响应及尾部范围，幂变换与时间滤波又改变幅值和相邻观测关系，因此处理后的值不再保留原始模拟力的直接单位含义。

邻域扩散在每个Patch恢复的二维阵列中进行。该仿真后处理操作的实现参数名为crosstalk，不表示已完成真实电子皮肤的物理串扰标定。系数$c=0.12$表示每个源单元保留$1-c$的响应，其余部分按可用的8邻域数量平均分配。边界单元的邻居少于内部单元，程序按源单元归一化，不能写成所有接收单元统一除以8。邻域扩散只作用于同一Patch，数组中相邻但属于不同Patch的两个索引不会据此交换响应。当前profile的gaussian\_sigma和noise\_std均为0，高斯滤波与随机噪声属于可选能力，并未在该配置中启用。
```

## 4. A10｜thesis/body/ch02_simulation_platform.tex:330

修改原因：统一仿真Patch内邻域扩散术语及实现参数名，不改变算子或数值。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第330行起）：

```latex
    信号处理 & Patch方阵及时间状态 & 死区、截断、幂响应、IIR及串扰\\
```

修改后（第330行起）：

```latex
    信号处理 & Patch方阵及时间状态 & 死区、截断、幂响应、IIR及邻域扩散\\
```

## 5. A10｜thesis/body/ch02_simulation_platform.tex:396

修改原因：统一仿真Patch内邻域扩散术语及实现参数名，不改变算子或数值。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第396行起）：

```latex
控制链路将关节目标或末端IK求解与底层位置伺服、物理积分分开，在指定配置下按20~Hz控制和1000~Hz积分执行。触觉阵列由皮肤STL派生，通过指节与指尖外壳拟合、手掌参数域投影及局部坐标构造形成16个Patch，再由712个原生touch感知区读取标量响应。死区、幂响应、时间滤波和邻域串扰进一步产生无量纲processed观测，其含义不同于经过标定的真实压力。
```

修改后（第396行起）：

```latex
控制链路将关节目标或末端IK求解与底层位置伺服、物理积分分开，在指定配置下按20~Hz控制和1000~Hz积分执行。触觉阵列由皮肤STL派生，通过指节与指尖外壳拟合、手掌参数域投影及局部坐标构造形成16个Patch，再由712个原生touch感知区读取标量响应。死区、幂响应、时间滤波和邻域扩散进一步产生无量纲processed观测，其含义不同于经过标定的真实压力。
```

## 6. A06｜thesis/body/ch04_multimodal_data.tex:292

修改原因：区分两套数据体系、数据规范支撑与训练文件直接读取。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第292行起）：

```latex
在上述数据条件下，后续研究需要进一步区分真实触觉与仿真触觉的分布及监督关系。下一章将基于已保存的真实统计信息和冻结触觉子集讨论跨域重建，不将规范化存储等同于两域已经具有同步物理对应。
```

修改后（第292行起）：

```latex
在上述数据条件下，后续研究需要进一步区分真实触觉与仿真触觉的分布及监督关系。本章的数据组织、来源追踪与质量控制为后续研究提供管理基础；下一章使用单独登记并冻结的Real、Natural Sim与Paired Sim触觉子集，本章讨论的LeRobot示教导出并非其初始跨域模型直接读取的训练输入。两套体系在仿真触觉观测和质量规范上相关，但没有已认证的一一数据对应，服务器示教库存也不能与触觉窗口数混为同一规模。
```

## 7. A08｜thesis/body/ch05_sections/section_5_5.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% Standalone section draft; not included in the official thesis main file.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
```

## 8. A08｜thesis/body/ch05_sections/section_5_4.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% Standalone section fragment. Not included in thesis.tex.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
```

## 9. A07｜thesis/body/ch05_sections/section_5_4.tex:5

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第5行起）：

```latex
统计伪配对为两域提供有限对应，但触觉窗口仍同时包含精确零值、正值幅度和跨帧变化。为在映射前提取各域表示，本文采用变分自编码器\cite{kingma2014vae}，分别建立Real VAE与Sim VAE，如图\ref{fig:ch05_dual_vae}所示。两者结构相同，Encoder与Decoder参数独立，通过各自输入的自重建学习表示；均输出$4\times64$潜变量，但维数一致并不赋予共同语义坐标。Real分支仅学习真实窗口，Sim分支同时接收自然任务仿真和伪配对仿真，重建目标仍为各自的Sim输入，对应Real窗口不作为Sim自重建标签。
```

修改后（第5行起）：

```latex
统计伪配对为两域提供有限对应，但触觉窗口仍同时包含精确零值、正值幅度和跨帧变化。为在映射前提取各域表示，本文采用变分自编码器\cite{kingma2014vae}，分别建立Real VAE与Sim VAE，如图\ref{fig:ch05_dual_vae}所示。该双域表示与下一节Translator组成的初始方案记为C3。两者结构相同，Encoder与Decoder参数独立，通过各自输入的自重建学习表示；均输出$4\times64$潜变量，但维数一致并不赋予共同语义坐标。Real分支仅学习真实窗口，Sim分支同时接收自然任务仿真和伪配对仿真，重建目标仍为各自的Sim输入，对应Real窗口不作为Sim自重建标签。
```

## 10. A07｜thesis/body/ch05_sections/section_5_4.tex:59

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第59行起）：

```latex
上述为C3基础模型。后续C13仅调整Real Decoder的当前帧Mask条件嵌入与幅值头，其余Encoder、Mask预测及循环骨干冻结；C23复用其自重建输出诊断，C24仅在相同可训练范围增加时序幅值约束，具体结果留待实验分析。各域重建目标只要求潜表示能够服务于本域读出，不约束同一维度表示相同接触特征。即使采用相同先验，也不能直接将两套坐标中的近邻解释为物理对应。独立自重建尚未建立两域语义对应，下一节以有限伪配对监督研究潜变量映射，而不预设同维表示已完成对齐。
```

修改后（第59行起）：

```latex
上述为C3基础模型。后续C13仅调整Real Decoder的当前帧Mask条件嵌入与幅值头，其余Encoder、Mask预测及循环骨干冻结；C23是复用C13输出的Real自重建读出诊断；C24是同一可训练范围内增加时序幅值约束的修复版本，具体结果留待实验分析。各域重建目标只要求潜表示能够服务于本域读出，不约束同一维度表示相同接触特征。即使采用相同先验，也不能直接将两套坐标中的近邻解释为物理对应。独立自重建尚未建立两域语义对应，下一节以有限伪配对监督研究潜变量映射，而不预设同维表示已完成对齐。
```

## 11. A08｜thesis/body/ch05_sections/section_5_6.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% Independent section draft; no official main-file integration.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
```

## 12. A07｜thesis/body/ch05_sections/section_5_6.tex:5

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第5行起）：

```latex
为检验统计伪配对与双域表示是否支持完整触觉转换，本节采用迁移前训练并冻结的C0评价器。其由原Real Train与Natural Sim Train训练，包括Logistic、kNN及MLP、GRU、Transformer各Seed 11、12、13，共11个实例。权重、Scaler、log1p处理、通道顺序与0.5阈值保持不变，标签方向为Real=1、Sim=0。跨域评价比较780个Real Validation窗口与同划分905个Natural Sim窗口的转换结果，分别来自4个Session和10个Root。这些是开发验证数据，不是新独立盲测；后训练的Probe不能替代该协议。
```

修改后（第5行起）：

```latex
为检验统计伪配对与双域表示是否支持完整触觉转换，本节沿用第\ref{sec:ch05_unpaired_problem}节定义的迁移前冻结域分类评价基线C0。其由原Real Train与Natural Sim Train训练，包括Logistic、kNN及MLP、GRU、Transformer各Seed 11、12、13，共11个实例。权重、Scaler、log1p处理、通道顺序与0.5阈值保持不变，标签方向为Real=1、Sim=0。跨域评价比较780个Real Validation窗口与同划分905个Natural Sim窗口的转换结果，分别来自4个Session和10个Root。这些是开发验证数据，不是新独立盲测；后训练的Probe不能替代该协议。
```

## 13. A07｜thesis/body/ch05_sections/section_5_6.tex:22

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第22行起）：

```latex
表\ref{tab:ch05_transfer_summary}汇总同角色跨域结果。C3采用原双域VAE与GRU Translator，其Validation结果取已有冻结链路复现，而非历史开发Test。虽然D全部达标，仅kNN同时满足AUC条件。表\ref{tab:ch05_validation_instances}中，C3的MLP分数仍保持正向排序差异，GRU和部分Transformer则出现反向排序；两种情况均不满足域不可辨识要求。例如MLP Seed11将905个迁移窗口全部判Real，Real中776/780判对，BA约0.497436；但AUC为0.591571。固定阈值下的高D因此可能来自类别预测偏向，不能证明分布已接近。
```

修改后（第22行起）：

```latex
表\ref{tab:ch05_transfer_summary}汇总C3、C13与C27的同角色跨域结果。C27指保留原C3双域VAE、改进Translator训练的方案，未与C13的Decoder修复合并。C3采用原双域VAE与GRU Translator，其Validation结果取已有冻结链路复现，而非历史开发Test。虽然D全部达标，仅kNN同时满足AUC条件。表\ref{tab:ch05_validation_instances}中，C3的MLP分数仍保持正向排序差异，GRU和部分Transformer则出现反向排序；两种情况均不满足域不可辨识要求。例如MLP Seed11将905个迁移窗口全部判Real，Real中776/780判对，BA约0.497436；但AUC为0.591571。固定阈值下的高D因此可能来自类别预测偏向，不能证明分布已接近。
```

## 14. A08｜thesis/body/ch05_sections/section_5_3.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% CH05-5.3 only: standalone section fragment, not included in thesis.tex.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
```

## 15. A08｜thesis/body/ch05_sections/section_5_1.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% Standalone section draft; not included in thesis.tex.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
```

## 16. A10｜thesis/body/ch05_sections/section_5_1.tex:5

修改原因：统一仿真Patch内邻域扩散术语及实现参数名，不改变算子或数值。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第5行起）：

```latex
真实与仿真触觉具有相同数组维度，并不意味着能够直接用于同一操作表示。本文采用的观测按16个Patch组织，共712个标量通道，仿真阵列覆盖五指及手掌。其传感器原生raw读数来自MuJoCo native touch，对感知区域内的法向接触力累加\cite{mujoco2026xml}；仿真processed观测则经过死区、限幅、非线性响应、时间滤波及邻域扩散处理。Real processed观测来自实机自身采集处理流程，因此两域processed均应视为处理后的数值响应，不能直接称为真实压力或共同力学单位。对这种观测进行迁移，需要检查保留了哪些接触信息，而不只是核对输出形状。
```

修改后（第5行起）：

```latex
真实与仿真触觉具有相同数组维度，并不意味着能够直接用于同一操作表示。本文采用的观测按16个Patch组织，共712个标量通道，仿真阵列覆盖五指及手掌。其传感器原生raw读数来自MuJoCo native touch，对感知区域内的法向接触力累加\cite{mujoco2026xml}；仿真processed观测则经过死区、限幅、非线性响应、时间滤波及Patch内邻域扩散处理（实现参数名为crosstalk）。Real processed观测来自实机自身采集处理流程，因此两域processed均应视为处理后的数值响应，不能直接称为真实压力或共同力学单位。对这种观测进行迁移，需要检查保留了哪些接触信息，而不只是核对输出形状。
```

## 17. A07｜thesis/body/ch05_sections/section_5_1.tex:13

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第13行起）：

```latex
表\ref{tab:ch05_domain_feature_difference}列出旧P37数据上既有转换输出经幅值校准后的表示分解诊断。Full与分位数先作log1p，各表示分别用Train拟合标准化，Probe由Logistic、kNN各1个及MLP、GRU、Transformer各Seed11、12、13组成，共11个；Full复用P38在相同输出、划分和预处理上的Probe拟合，不新增训练。此处为迁移后Probe诊断，非后文冻结C0验收。定义$D=1-2|\mathrm{BA}-0.5|$，只度量BA偏离0.5的程度；表中三个表示的最小D均低于0.90，表明至少存在Probe未满足固定阈值下的BA等价门槛；其排序判别能力仍需结合AUC判断。因此，不能仅凭这些结果认定支持或正值摘要已经实现域混淆。幅值调整保持Mask不变，不代表完整触觉已匹配。
```

修改后（第13行起）：

```latex
表\ref{tab:ch05_domain_feature_difference}列出旧P37数据上既有转换输出经幅值校准后的表示分解诊断。Full与分位数先作log1p，各表示分别用Train拟合标准化，Probe由Logistic、kNN各1个及MLP、GRU、Transformer各Seed11、12、13组成，共11个；Full复用P38在相同输出、划分和预处理上的Probe拟合，不新增训练。此处为迁移后Probe诊断，非后文迁移前冻结的域分类评价基线（C0）验收。定义$D=1-2|\mathrm{BA}-0.5|$，只度量BA偏离0.5的程度；表中三个表示的最小D均低于0.90，表明至少存在Probe未满足固定阈值下的BA等价门槛；其排序判别能力仍需结合AUC判断。因此，不能仅凭这些结果认定支持或正值摘要已经实现域混淆。幅值调整保持Mask不变，不代表完整触觉已匹配。
```

## 18. A08｜thesis/body/ch05_sections/section_5_2.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% Independent section fragment; not included in thesis.tex.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
```

## 19. A06｜thesis/body/ch05_sections/section_5_2.tex:7

修改原因：区分两套数据体系、数据规范支撑与训练文件直接读取。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第7行起）：

```latex
研究使用三类数据：Real提供真实域观测，用于域内表示学习，并提供统计先验及伪配对目标侧信息；Natural Sim来自自然任务仿真，与Paired Sim共同参与Sim域表示学习，也是最终转换对象；Paired Sim则由真实统计先验指导仿真物理探针生成，按登记关系与Real窗口构成统计伪配对，为潜变量映射提供有限监督。任意Real与Sim窗口并无已知共同接触状态，不能直接互作真实标签。现有1200个训练窗口与240个验证窗口也不代表1440个独立Root；登记关联不构成同步物理接触真值，来源复用与覆盖限制仍须保留。Natural参与Sim表示学习，不等于其具有对应Real目标，亦不等于直接参与原始配对映射训练。
```

修改后（第7行起）：

```latex
本章使用第\ref{sec:data_quality_summary}节所区分的单独登记并冻结的触觉子集，不直接读取LeRobot示教导出。其中三类数据分别承担不同角色：Real提供真实域观测，用于域内表示学习，并提供统计先验及伪配对目标侧信息；Natural Sim来自自然任务仿真，与Paired Sim共同参与Sim域表示学习，也是最终转换对象；Paired Sim则由真实统计先验指导仿真物理探针生成，按登记关系与Real窗口构成统计伪配对，为潜变量映射提供有限监督。任意Real与Sim窗口并无已知共同接触状态，不能直接互作真实标签。现有1200个训练窗口与240个验证窗口也不代表1440个独立Root；登记关联不构成同步物理接触真值，来源复用与覆盖限制仍须保留。Natural参与Sim表示学习，不等于其具有对应Real目标，亦不等于直接参与原始配对映射训练。
```

## 20. A08｜thesis/body/ch06_sections/section_6_1_shared_vector_field.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% Standalone draft, 2026-11-26; not included in the thesis main file.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch06_adversarial_evaluation.tex.
```

## 21. A07｜thesis/body/ch06_sections/section_6_1_shared_vector_field.tex:5

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第5行起）：

```latex
第五章利用统计伪配对监督Translator，在独立训练的双域VAE之间学习直接潜变量映射，再由Real Decoder生成目标域触觉。该方法通过GRU内部状态建模四帧序列，但未显式构造潜变量的积分更新路径；配对潜变量误差也不能直接说明Natural Sim输出的域分布是否接近Real。为在保留既有表示与读出模块的条件下研究可调整的迁移路径，本文将原映射扩展为源条件共享参数向量场，通过数值积分获得目标潜状态。这里的扩展是迁移计算形式的改变，并未重新建立公共物理坐标，也不预设两域潜变量具有物理等价性。
```

修改后（第5行起）：

```latex
第五章利用统计伪配对监督Translator，在独立训练的双域VAE之间学习直接潜变量映射，再由Real Decoder生成目标域触觉。该方法通过GRU内部状态建模四帧序列，但未显式构造潜变量的积分更新路径；配对潜变量误差也不能直接说明Natural Sim输出的域分布是否接近Real。为在保留既有表示与读出模块的条件下研究可调整的迁移路径，本文将原映射扩展为源条件共享参数向量场，通过数值积分获得目标潜状态。本节将该共享参数向量场实验记为P44。这里的扩展是迁移计算形式的改变，并未重新建立公共物理坐标，也不预设两域潜变量具有物理等价性。
```

## 22. A08｜thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% Standalone 6.2 draft, not included in the official thesis main file.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch06_adversarial_evaluation.tex.
```

## 23. A07｜thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex:21

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第21行起）：

```latex
为核查输入表示与梯度机制，设置表\ref{tab:ch06_discriminator_comparison}的三种配置。C组在判别输入端另加STE：
```

修改后（第21行起）：

```latex
为核查输入表示与梯度机制，在P44基础上开展编号为P45的结构对照，设置表\ref{tab:ch06_discriminator_comparison}的三种配置。C组在判别输入端另加STE：
```

## 24. A08｜thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex:1

修改原因：只修正当前活动片段的过时接入注释。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第1行起）：

```latex
% Standalone 6.3 draft; historical CSV analysis only, not included in thesis.tex.
```

修改后（第1行起）：

```latex
% Active section fragment; included through body/ch06_adversarial_evaluation.tex.
```

## 25. A07｜thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex:5

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第5行起）：

```latex
前述对照区分了判别输入与梯度路径，但尚未回答：判别器适应已有生成结果后，再更新生成场，能否持续改善迁移质量。为建立可比较的起点，先对冻结候选开展统一分布评价。以未转换Natural Sim为Original，将上一节A组训练终点按$K=1$读出的生成器记为$F_0$；该权重本身采用四种$K$调度训练，并非仅用单步训练或重新随机初始化。统一评价未新增生成器训练，其组等权SWD由Original的0.411343降至$F_0$的0.206609，相对下降49.77\%，定义$R_W=(\mathrm{SWD}_{\mathrm{Original}}-\mathrm{SWD}_{F_0})/\mathrm{SWD}_{\mathrm{Original}}$，本次为0.4977202817，95\%组Bootstrap区间约为$[0.408514,0.566743]$，如图\ref{fig:ch06_f0_swd_baseline}所示。这只支持指定距离合同下的改善，不能解释为迁移成功率。
```

修改后（第5行起）：

```latex
前述对照区分了判别输入与梯度路径，但尚未回答：判别器适应已有生成结果后，再更新生成场，能否持续改善迁移质量。本节将这一多轮实验记为P49，$F_i$与$D_i$分别表示生成器与判别器的阶段状态，后者不同于第五章由BA定义的域混淆指标$D$。为建立可比较的起点，先对冻结候选开展统一分布评价。以未转换Natural Sim为Original，将上一节A组训练终点按$K=1$读出的生成器记为$F_0$；该权重本身采用四种$K$调度训练，并非仅用单步训练或重新随机初始化。统一评价未新增生成器训练，其组等权SWD由Original的0.411343降至$F_0$的0.206609，相对下降49.77\%，定义$R_W=(\mathrm{SWD}_{\mathrm{Original}}-\mathrm{SWD}_{F_0})/\mathrm{SWD}_{\mathrm{Original}}$，本次为0.4977202817，95\%组Bootstrap区间约为$[0.408514,0.566743]$，如图\ref{fig:ch06_f0_swd_baseline}所示。这只支持指定距离合同下的改善，不能解释为迁移成功率。
```

## 26. A07｜thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex:4

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第4行起）：

```latex
多轮优化提高了当前判别器下的接受率，却未维持固定距离评价中的质量。为此，本节探索条件残差Refiner与Global、Spatial、Temporal多视角判别约束。实验从$F_0$重新开始，而非续接上一节$F_3$，采用Seed 5001、单轮训练；两套VAE冻结，共享向量场与Refiner可训练。$D_{50}$是实验标识，不表示训练50轮。
```

修改后（第4行起）：

```latex
多轮优化提高了当前判别器下的接受率，却未维持固定距离评价中的质量。为此，本节探索条件残差Refiner与Global、Spatial、Temporal多视角判别约束。本实验记为P50，从$F_0$重新开始，而非续接上一节$F_3$，采用Seed 5001、单轮训练；两套VAE冻结，共享向量场与Refiner可训练。$F_{50}$表示本实验生成器的终点结果，$D_{50}$表示多视角判别器；下标50对应实验编号，不表示训练50轮。
```

## 27. A07｜thesis/body/ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex:4

修改原因：简要解释实验/模型/评价角色，保持原方法、数据与指标。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第4行起）：

```latex
前文区分了边际分布、邻域覆盖、输入依赖和判别接受情况。进一步作出迁移质量判断，还需审查评价器能否稳定接受真实触觉。本节依据冻结$D_{50}$的历史只读审计，分析真实域接受能力、生成分数及跨Session校准差异，不继续优化生成器。本次P51只读审计未执行训练或参数优化，$D_{50}$权重保持冻结，未改变类别方向；本节仅复核已有记录，不重复历史推理。
```

修改后（第4行起）：

```latex
前文区分了边际分布、邻域覆盖、输入依赖和判别接受情况。进一步作出迁移质量判断，还需审查评价器能否稳定接受真实触觉。本节依据编号为P51的冻结$D_{50}$历史只读审计，分析真实域接受能力、生成分数及跨Session校准差异，不继续优化生成器。本次P51只读审计未执行训练或参数优化，$D_{50}$权重保持冻结，未改变类别方向；本节仅复核已有记录，不重复历史推理。
```

## 28. A04｜thesis/body/ch07_sections/section_7_1_research_work_summary.tex:4

修改原因：对应三条主线与四项工作，保持原有归类层级及贡献范围。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第4行起）：

```latex
针对灵巧手操作中机器人运动、接触感知与任务执行需要协同组织的问题，本文构建了RM75B机械臂与途见科技自研欠驱动灵巧手的多任务仿真平台。在MuJoCo环境中集成机器人模型、控制器、任务逻辑和触觉记录模块，结合曲面几何布局建立16个Patch、712维processed触觉观测体系，明确感知区域、原生读取与信号处理之间的关系。通过多任务接口及指定历史案例的观测检查，形成支撑后续抓取和触觉研究的仿真基础。这部分工作侧重系统集成与感知建模，其验证范围为所列模型实现和记录，不等同于实机触觉标定或真实机器人部署。
```

修改后（第4行起）：

```latex
以下按第\ref{sec:research_contents}节的四项具体工作总结，其中抓取示教与数据质量控制共同对应专家示教与数据主线。针对灵巧手操作中机器人运动、接触感知与任务执行需要协同组织的问题，本文构建了RM75B机械臂与途见科技自研欠驱动灵巧手的多任务仿真平台。在MuJoCo环境中集成机器人模型、控制器、任务逻辑和触觉记录模块，结合曲面几何布局建立16个Patch、712维processed触觉观测体系，明确感知区域、原生读取与信号处理之间的关系。通过多任务接口及指定历史案例的观测检查，形成支撑后续抓取和触觉研究的仿真基础。这部分工作侧重系统集成与感知建模，其验证范围为所列模型实现和记录，不等同于实机触觉标定或真实机器人部署。
```

## 29. A05｜thesis/body/ch07_sections/section_7_1_research_work_summary.tex:6

修改原因：依据第三章的实际分层检查收紧总结，避免全轨迹碰撞认证含义。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第6行起）：

```latex
针对几何抓取候选不能直接转化为机器人可执行操作的问题，本文适配GraspQP官方抓取评价指标，参考DexEvolve设计实现进化搜索，并结合腕部姿态扩展形成多来源候选。通过机械臂逆运动学、碰撞约束和连续轨迹求解，将所选候选连接为分阶段专家示教轨迹，再结合执行反馈、任务结果与Strict检查验证轨迹，区分操作完成与示教质量。局部扰动轨迹和失败分类为流程的适用范围提供了历史依据，并形成后续数据记录的轨迹来源。该工作体现为现有工具的适配整合与验证流程构建，不将第三方算法本身作为原创，也不把候选质量、Strict通过与实际任务成功合并为同一结论。
```

修改后（第6行起）：

```latex
针对几何抓取候选不能直接转化为机器人可执行操作的问题，本文适配GraspQP官方抓取评价指标，参考DexEvolve设计实现进化搜索，并结合腕部姿态扩展形成多来源候选。通过机械臂逆运动学和连续轨迹求解连接分阶段专家示教，并按第\ref{sec:grasp_reachability}节所述，结合桌面几何惩罚、局部接近约束、执行监测和尾段Strict检查验证轨迹与任务结果，区分操作完成与示教质量。局部扰动轨迹和失败分类为流程的适用范围提供了历史依据，并形成后续数据记录的轨迹来源。该工作体现为现有工具的适配整合与验证流程构建，不将第三方算法本身作为原创，也不把候选质量、Strict通过与实际任务成功合并为同一结论。
```

## 30. A06｜thesis/body/ch07_sections/section_7_1_research_work_summary.tex:10

修改原因：区分两套数据体系、数据规范支撑与训练文件直接读取。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第10行起）：

```latex
针对仿真与真实触觉存在分布差异且缺少认证同步物理配对的问题，本文利用真实统计先验与仿真探针构建伪配对，分别学习独立双域VAE表示，再由GRU Translator建立潜变量映射。在此基础上研究共享参数向量场、对抗优化及多视角判别与残差修正，并结合分布距离、邻域覆盖、输入依赖和判别器跨Session可靠性开展综合评价。这些工作以指定触觉子集为研究对象，完成了统计监督、跨域重建与质量诊断的方法探索链路，不意味着全部平台数据均参与训练。现有实验获得了限定指标下的局部改善，同时揭示接受率、分布质量和评价可靠性的差异，尚未完成完整触觉Sim-to-Real迁移验收。
```

修改后（第10行起）：

```latex
针对仿真与真实触觉存在分布差异且缺少认证同步物理配对的问题，本文利用真实统计先验与仿真探针构建伪配对，分别学习独立双域VAE表示，再由GRU Translator建立潜变量映射。在此基础上研究共享参数向量场、对抗优化及多视角判别与残差修正，并结合分布距离、邻域覆盖、输入依赖和判别器跨Session可靠性开展综合评价。这些工作使用单独登记并冻结的触觉子集，沿用前述数据管理与质量规范，但不将LeRobot导出直接作为跨域模型的训练输入，也不认证两套数据的一一对应。由此形成统计监督、跨域重建与质量诊断的方法探索链路，不意味着全部平台数据均参与训练。现有实验获得了限定指标下的局部改善，同时揭示接受率、分布质量和评价可靠性的差异，尚未完成完整触觉Sim-to-Real迁移验收。
```

## 31. A05｜thesis/body/ch07_sections/section_7_2_main_research_conclusions.tex:4

修改原因：依据第三章的实际分层检查收紧总结，避免全轨迹碰撞认证含义。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第4行起）：

```latex
本文的仿真实现与指定案例表明，统一机器人模型、控制接口、任务接口和触觉观测合同，能够为多任务操作提供可复用的实验基础。抓取候选仍需经过可达性、碰撞约束、连续轨迹及任务执行的分层检验：几何质量不能替代IK可达性，可达性不能替代任务成功，任务成功也不能替代Strict示教质量。各层判断针对不同条件，当前证据仅支持已有模型、任务及案例范围，不能外推为通用抓取能力或实机有效性。
```

修改后（第4行起）：

```latex
本文的仿真实现与指定案例表明，统一机器人模型、控制接口、任务接口和触觉观测合同，能够为多任务操作提供可复用的实验基础。候选转化为轨迹时，可达性筛选、桌面几何与局部接近约束、连续执行监测及尾段Strict检查承担不同层次的检验：几何质量不能替代IK可达性，可达性不能替代任务成功，任务成功也不能替代Strict示教质量。各层判断针对不同条件，当前证据仅支持已有模型、任务及案例范围，不能外推为通用抓取能力或实机有效性。
```

## 32. A09｜thesis/body/ch07_sections/section_7_2_main_research_conclusions.tex:8

修改原因：以交叉引用保留修订后开发协议背景，原披露未改。

对研究含义的影响：无新增实验事实；仅澄清或限定原有含义。

原文（修改前第8行起）：

```latex
跨域实验显示，局部指标改善尚不足以证明完整迁移。C27在原冻结Validation的11个实例中仅2个联合通过；$F_0$相对Original的SWD下降49.77\%，仅为指定合同下的边际改善。多轮对抗终点在$D_3$下FR达100\%，不能据此认定域不可辨识；多视角残差终点$F_{50}$虽Joint FR为100\%，分布距离却明显退化。P51因未满足既定Real接受要求对$D_{50}$判定NO-GO，该判别器不能作为当前实验的最终可靠评价器，该结论不外推$D_3$或其他方法。由此可见，接受率不等于分布相似，边际相似不保证输入输出对应，统计依赖不等于物理语义，梯度可达也不保证优化有效。评价器可靠性同样是迁移结论的前提。本文完成了统计伪监督、潜变量映射、共享参数向量场及多视角评价的系列探索，但现有证据尚不能证明完整触觉Sim-to-Real迁移成功。
```

修改后（第8行起）：

```latex
跨域实验显示，局部指标改善尚不足以证明完整迁移。C27在原冻结Validation的11个实例中仅2个联合通过；$F_0$相对Original的SWD下降49.77\%，仅为指定合同下的边际改善。在第\ref{sec:ch06_multi_round_adversarial}节披露的修订后开发协议下，多轮对抗终点在$D_3$下FR达100\%，不能据此认定域不可辨识；多视角残差终点$F_{50}$虽Joint FR为100\%，分布距离却明显退化。P51因未满足既定Real接受要求对$D_{50}$判定NO-GO，该判别器不能作为当前实验的最终可靠评价器，该结论不外推$D_3$或其他方法。由此可见，接受率不等于分布相似，边际相似不保证输入输出对应，统计依赖不等于物理语义，梯度可达也不保证优化有效。评价器可靠性同样是迁移结论的前提。本文完成了统计伪监督、潜变量映射、共享参数向量场及多视角评价的系列探索，但现有证据尚不能证明完整触觉Sim-to-Real迁移成功。
```

## 完整统一Diff

```diff
--- thesis/body/ch01_introduction.tex (before)
+++ thesis/body/ch01_introduction.tex (after)
@@ -95,7 +95,7 @@
 \section{主要研究内容与创新点}
 \label{sec:research_contents}
 
-本文围绕仿真操作、示教数据和触觉重建开展研究。主要工作包括平台与观测集成、适应具体机器人约束的示教流程，以及有限统计监督下的跨域方法探索。三者分别提供实验条件、可追溯数据和重建分析依据，其贡献需要结合实现与验证范围说明，平台集成、开源适配和迁移效果不作同一层次的创新判断。
+本文围绕仿真操作、示教数据和触觉重建开展研究。主要工作包括平台与观测集成、适应具体机器人约束的示教流程，以及有限统计监督下的跨域方法探索。三者分别提供实验条件、可追溯数据和重建分析依据，其贡献需要结合实现与验证范围说明，平台集成、开源适配和迁移效果不作同一层次的创新判断。上述三条主线对应四项工作：平台与触觉观测、抓取候选与专家示教、多模态示教数据与质量控制、有限监督触觉重建与评价。其中第二、三项同属专家示教与数据主线，第七章据此分项总结。
 
 在平台与触觉建模方面，构建RM75B机械臂与途见科技自研欠驱动灵巧手的仿真组合，将任务环境、控制接口和曲面触觉阵列纳入操作流程。围绕感知单元的空间布局，建立原生触觉读取及后续信号处理链，获得712维标量观测，并在抓取抬升、抓取放置、堆叠、推动和螺母装配任务中形成开发验证记录。这部分工作的特色在于连接机器人运动、局部接触观测与任务记录，为不同操作阶段提供一致的仿真接口；其验证范围是模型与记录链路的功能，不等同于真实传感器物理标定或实机部署。
 
--- thesis/body/ch02_simulation_platform.tex (before)
+++ thesis/body/ch02_simulation_platform.tex (after)
@@ -293,7 +293,7 @@
   \fbox{\begin{minipage}{0.91\textwidth}\centering
   曲面采样 $\rightarrow$ box site与原生touch $\rightarrow$ 地址绑定\\[5pt]
   sensordata读取 $\rightarrow$ 死区与截断 $\rightarrow$ 幂响应\\[5pt]
-  $\rightarrow$ IIR滤波 $\rightarrow$ Patch内邻域串扰\\[5pt]
+  $\rightarrow$ IIR滤波 $\rightarrow$ Patch内邻域扩散\\[5pt]
   $\rightarrow$ 可选高斯滤波与噪声（指定profile关闭）\\[5pt]
   $\rightarrow$ 归一化 $\rightarrow$ float32的712维processed观测
   \end{minipage}}
@@ -312,9 +312,9 @@
  \end{aligned}
  \label{eq:ch02_processing}
 \end{equation}
-其中，$t$为连续调用的观测索引，$\mathcal C_c$为下述Patch内串扰算子，$\boldsymbol y_t$为processed向量。末行适用于当前关闭高斯滤波和噪声、开启归一化的profile：先完成串扰及范围限制，再除以饱和尺度$s$，最后转换为float32。滤波初态由处理器reset路径决定，不能假定任何数据片段都以相同方式独立初始化。指定profile使用$d=0.002$、$s=1$、$p=0.7$、$\alpha=0.45$，并开启归一化。死区和饱和设置改变了弱响应及尾部范围，幂变换与时间滤波又改变幅值和相邻观测关系，因此处理后的值不再保留原始模拟力的直接单位含义。
-
-邻域串扰在每个Patch恢复的二维阵列中进行。系数$c=0.12$表示每个源单元保留$1-c$的响应，其余部分按可用的8邻域数量平均分配。边界单元的邻居少于内部单元，程序按源单元归一化，不能写成所有接收单元统一除以8。串扰只作用于同一Patch，数组中相邻但属于不同Patch的两个索引不会据此交换响应。当前profile的gaussian\_sigma和noise\_std均为0，高斯滤波与随机噪声属于可选能力，并未在该配置中启用。
+其中，$t$为连续调用的观测索引，$\mathcal C_c$为下述Patch内邻域扩散算子，$\boldsymbol y_t$为processed向量。末行适用于当前关闭高斯滤波和噪声、开启归一化的profile：先完成邻域扩散及范围限制，再除以饱和尺度$s$，最后转换为float32。滤波初态由处理器reset路径决定，不能假定任何数据片段都以相同方式独立初始化。指定profile使用$d=0.002$、$s=1$、$p=0.7$、$\alpha=0.45$，并开启归一化。死区和饱和设置改变了弱响应及尾部范围，幂变换与时间滤波又改变幅值和相邻观测关系，因此处理后的值不再保留原始模拟力的直接单位含义。
+
+邻域扩散在每个Patch恢复的二维阵列中进行。该仿真后处理操作的实现参数名为crosstalk，不表示已完成真实电子皮肤的物理串扰标定。系数$c=0.12$表示每个源单元保留$1-c$的响应，其余部分按可用的8邻域数量平均分配。边界单元的邻居少于内部单元，程序按源单元归一化，不能写成所有接收单元统一除以8。邻域扩散只作用于同一Patch，数组中相邻但属于不同Patch的两个索引不会据此交换响应。当前profile的gaussian\_sigma和noise\_std均为0，高斯滤波与随机噪声属于可选能力，并未在该配置中启用。
 
 \begin{table}[htbp]
   \centering\small
@@ -327,7 +327,7 @@
     空间布局 & 16个Patch、712个site & 定义感知位置、方向和体积\\
     原生touch & 每site一个非负标量 & 累加满足感知条件的接触法向力\\
     原始读取 & 712维float32副本 & 按sensor\_adr读取，顺序固定\\
-    信号处理 & Patch方阵及时间状态 & 死区、截断、幂响应、IIR及串扰\\
+    信号处理 & Patch方阵及时间状态 & 死区、截断、幂响应、IIR及邻域扩散\\
     processed输出 & 712维float32 & 指定profile归一化，无量纲响应\\
     \hline
   \end{tabular}
@@ -393,6 +393,6 @@
 
 本章围绕多任务灵巧手操作的执行与感知需求，介绍了RM75B机械臂与途见科技灵巧手仿真平台的模型、控制、触觉、任务和记录接口。当前机械臂采用七个转动关节及位置执行器，手部采用十二个转动关节、六个固定腱和六个位置驱动通道。通过区分物理关节、传动坐标与控制通道，明确了欠驱动结构的实现范围，避免将历史命名中的推杆与当前机构混同。
 
-控制链路将关节目标或末端IK求解与底层位置伺服、物理积分分开，在指定配置下按20~Hz控制和1000~Hz积分执行。触觉阵列由皮肤STL派生，通过指节与指尖外壳拟合、手掌参数域投影及局部坐标构造形成16个Patch，再由712个原生touch感知区读取标量响应。死区、幂响应、时间滤波和邻域串扰进一步产生无量纲processed观测，其含义不同于经过标定的真实压力。
+控制链路将关节目标或末端IK求解与底层位置伺服、物理积分分开，在指定配置下按20~Hz控制和1000~Hz积分执行。触觉阵列由皮肤STL派生，通过指节与指尖外壳拟合、手掌参数域投影及局部坐标构造形成16个Patch，再由712个原生touch感知区读取标量响应。死区、幂响应、时间滤波和邻域扩散进一步产生无量纲processed观测，其含义不同于经过标定的真实压力。
 
 已有五条历史任务记录提供了触觉数组有限、非负和规则时间索引的功能证据，但与当前模型的完整版本一致性、独立Root关系及实物映射仍有限制。基于这些边界，下一章讨论如何将几何抓取候选转化为满足机械臂可达性、连续运动和任务要求的专家示教。
--- thesis/body/ch04_multimodal_data.tex (before)
+++ thesis/body/ch04_multimodal_data.tex (after)
@@ -289,4 +289,4 @@
 
 已有数据核验确认了指定Sugar导出字段与源记录的一致性，也发现旧PickPlace命令连续性和旧Stack字段缺失问题。五任务v4开发包及Sink历史回放对照提供了版本化质量控制的有限依据，但不构成服务器全量验收、独立Root认证或全局无泄漏证明。保留来源、Hash、验证层级和失败记录，使这些局限能够被明确识别，而非通过统一文件名掩盖。
 
-在上述数据条件下，后续研究需要进一步区分真实触觉与仿真触觉的分布及监督关系。下一章将基于已保存的真实统计信息和冻结触觉子集讨论跨域重建，不将规范化存储等同于两域已经具有同步物理对应。
+在上述数据条件下，后续研究需要进一步区分真实触觉与仿真触觉的分布及监督关系。本章的数据组织、来源追踪与质量控制为后续研究提供管理基础；下一章使用单独登记并冻结的Real、Natural Sim与Paired Sim触觉子集，本章讨论的LeRobot示教导出并非其初始跨域模型直接读取的训练输入。两套体系在仿真触觉观测和质量规范上相关，但没有已认证的一一数据对应，服务器示教库存也不能与触觉窗口数混为同一规模。
--- thesis/body/ch05_sections/section_5_5.tex (before)
+++ thesis/body/ch05_sections/section_5_5.tex (after)
@@ -1,4 +1,4 @@
-% Standalone section draft; not included in the official thesis main file.
+% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
 \section{基于伪配对监督的跨域潜变量映射}
 \label{sec:ch05_latent_translator}
 
--- thesis/body/ch05_sections/section_5_4.tex (before)
+++ thesis/body/ch05_sections/section_5_4.tex (after)
@@ -1,8 +1,8 @@
-% Standalone section fragment. Not included in thesis.tex.
+% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
 \section{基于双域变分自编码器的触觉表示学习}
 \label{sec:ch05_dual_vae_representation}
 
-统计伪配对为两域提供有限对应，但触觉窗口仍同时包含精确零值、正值幅度和跨帧变化。为在映射前提取各域表示，本文采用变分自编码器\cite{kingma2014vae}，分别建立Real VAE与Sim VAE，如图\ref{fig:ch05_dual_vae}所示。两者结构相同，Encoder与Decoder参数独立，通过各自输入的自重建学习表示；均输出$4\times64$潜变量，但维数一致并不赋予共同语义坐标。Real分支仅学习真实窗口，Sim分支同时接收自然任务仿真和伪配对仿真，重建目标仍为各自的Sim输入，对应Real窗口不作为Sim自重建标签。
+统计伪配对为两域提供有限对应，但触觉窗口仍同时包含精确零值、正值幅度和跨帧变化。为在映射前提取各域表示，本文采用变分自编码器\cite{kingma2014vae}，分别建立Real VAE与Sim VAE，如图\ref{fig:ch05_dual_vae}所示。该双域表示与下一节Translator组成的初始方案记为C3。两者结构相同，Encoder与Decoder参数独立，通过各自输入的自重建学习表示；均输出$4\times64$潜变量，但维数一致并不赋予共同语义坐标。Real分支仅学习真实窗口，Sim分支同时接收自然任务仿真和伪配对仿真，重建目标仍为各自的Sim输入，对应Real窗口不作为Sim自重建标签。
 
 \begin{figure}[htbp]
  \centering
@@ -56,4 +56,4 @@
 
 Real训练池为5153窗，Sim训练池包含4049个Natural窗与1200个Paired窗，Sim采用分层混合批次，每步从Natural Sim和Paired Sim各取32窗，合成64窗，每轮127步。较小的Paired组重复访问，实现批次等量混合，而非按原始规模比例采样。Real每轮81步，两域总优化步数分别为3645和5715，重复访问不增加独立样本。两域分别拟合尺度，Real与Sim的输入尺度分别约为0.14118和0.04089，正值尺度分别约为0.0050025和$10^{-7}$。Sim正值尺度触及下限，其拟合按合并训练元素进行，与训练时两类窗口等量采样的权重不同，上述尺度均由传感器处理后的无量纲触觉数值计算，不对应实际压力或力学单位。采用AdamW、学习率$10^{-3}$、批量64，两域均训练45轮，仅在第30至45轮按各自验证目标选择检查点，最终均选第30轮，属于限定候选区间内的选择，而非全部轮次的无约束最优。验证损失取后验均值，但仍使用真实Mask条件，不能替代自由重建检验。Real按验证重建项与KL加权和选取权重，Sim则对Natural与Paired验证目标等权汇合；两套模型均未依据域分类器成绩选择。
 
-上述为C3基础模型。后续C13仅调整Real Decoder的当前帧Mask条件嵌入与幅值头，其余Encoder、Mask预测及循环骨干冻结；C23复用其自重建输出诊断，C24仅在相同可训练范围增加时序幅值约束，具体结果留待实验分析。各域重建目标只要求潜表示能够服务于本域读出，不约束同一维度表示相同接触特征。即使采用相同先验，也不能直接将两套坐标中的近邻解释为物理对应。独立自重建尚未建立两域语义对应，下一节以有限伪配对监督研究潜变量映射，而不预设同维表示已完成对齐。
+上述为C3基础模型。后续C13仅调整Real Decoder的当前帧Mask条件嵌入与幅值头，其余Encoder、Mask预测及循环骨干冻结；C23是复用C13输出的Real自重建读出诊断；C24是同一可训练范围内增加时序幅值约束的修复版本，具体结果留待实验分析。各域重建目标只要求潜表示能够服务于本域读出，不约束同一维度表示相同接触特征。即使采用相同先验，也不能直接将两套坐标中的近邻解释为物理对应。独立自重建尚未建立两域语义对应，下一节以有限伪配对监督研究潜变量映射，而不预设同维表示已完成对齐。
--- thesis/body/ch05_sections/section_5_6.tex (before)
+++ thesis/body/ch05_sections/section_5_6.tex (after)
@@ -1,8 +1,8 @@
-% Independent section draft; no official main-file integration.
+% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
 \section{跨域触觉重建实验结果与分析}
 \label{sec:ch05_reconstruction_results}
 
-为检验统计伪配对与双域表示是否支持完整触觉转换，本节采用迁移前训练并冻结的C0评价器。其由原Real Train与Natural Sim Train训练，包括Logistic、kNN及MLP、GRU、Transformer各Seed 11、12、13，共11个实例。权重、Scaler、log1p处理、通道顺序与0.5阈值保持不变，标签方向为Real=1、Sim=0。跨域评价比较780个Real Validation窗口与同划分905个Natural Sim窗口的转换结果，分别来自4个Session和10个Root。这些是开发验证数据，不是新独立盲测；后训练的Probe不能替代该协议。
+为检验统计伪配对与双域表示是否支持完整触觉转换，本节沿用第\ref{sec:ch05_unpaired_problem}节定义的迁移前冻结域分类评价基线C0。其由原Real Train与Natural Sim Train训练，包括Logistic、kNN及MLP、GRU、Transformer各Seed 11、12、13，共11个实例。权重、Scaler、log1p处理、通道顺序与0.5阈值保持不变，标签方向为Real=1、Sim=0。跨域评价比较780个Real Validation窗口与同划分905个Natural Sim窗口的转换结果，分别来自4个Session和10个Root。这些是开发验证数据，不是新独立盲测；后训练的Probe不能替代该协议。
 
 \begin{figure}[htbp]
  \centering
@@ -19,7 +19,7 @@
 
 记两类召回率为$R_r$与$R_s$，则$\mathrm{BA}=(R_r+R_s)/2$，$D=1-2|\mathrm{BA}-0.5|$。每个实例必须同时满足BA在$[0.45,0.55]$、AUC在$[0.475,0.525]$及$D\ge0.90$，且11个实例全部满足才算联合通过。BA与D门槛数学等价，不是两项独立证据；AUC检验分数排序，不能在小于0.5时反转后参加验收。点估计按窗口计算，不应误称Session或Root等权宏平均。原始未转换数据上，11个C0的Validation BA均超过0.98，表明其已能区分原始两域；但冻结后的测试只覆盖这些模型学到的判别特征，即使通过也不能替代触觉结构与输入响应保持检查。
 
-表\ref{tab:ch05_transfer_summary}汇总同角色跨域结果。C3采用原双域VAE与GRU Translator，其Validation结果取已有冻结链路复现，而非历史开发Test。虽然D全部达标，仅kNN同时满足AUC条件。表\ref{tab:ch05_validation_instances}中，C3的MLP分数仍保持正向排序差异，GRU和部分Transformer则出现反向排序；两种情况均不满足域不可辨识要求。例如MLP Seed11将905个迁移窗口全部判Real，Real中776/780判对，BA约0.497436；但AUC为0.591571。固定阈值下的高D因此可能来自类别预测偏向，不能证明分布已接近。
+表\ref{tab:ch05_transfer_summary}汇总C3、C13与C27的同角色跨域结果。C27指保留原C3双域VAE、改进Translator训练的方案，未与C13的Decoder修复合并。C3采用原双域VAE与GRU Translator，其Validation结果取已有冻结链路复现，而非历史开发Test。虽然D全部达标，仅kNN同时满足AUC条件。表\ref{tab:ch05_validation_instances}中，C3的MLP分数仍保持正向排序差异，GRU和部分Transformer则出现反向排序；两种情况均不满足域不可辨识要求。例如MLP Seed11将905个迁移窗口全部判Real，Real中776/780判对，BA约0.497436；但AUC为0.591571。固定阈值下的高D因此可能来自类别预测偏向，不能证明分布已接近。
 
 \begin{table}[htbp]
  \centering\small
--- thesis/body/ch05_sections/section_5_3.tex (before)
+++ thesis/body/ch05_sections/section_5_3.tex (after)
@@ -1,4 +1,4 @@
-% CH05-5.3 only: standalone section fragment, not included in thesis.tex.
+% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
 \section{GMM统计先验引导的仿真触觉伪配对构建}
 \label{sec:ch05_gmm_pseudopairs}
 
--- thesis/body/ch05_sections/section_5_1.tex (before)
+++ thesis/body/ch05_sections/section_5_1.tex (after)
@@ -1,8 +1,8 @@
-% Standalone section draft; not included in thesis.tex.
+% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
 \section{真实与仿真触觉差异分析}
 \label{sec:ch05_real_sim_difference}
 
-真实与仿真触觉具有相同数组维度，并不意味着能够直接用于同一操作表示。本文采用的观测按16个Patch组织，共712个标量通道，仿真阵列覆盖五指及手掌。其传感器原生raw读数来自MuJoCo native touch，对感知区域内的法向接触力累加\cite{mujoco2026xml}；仿真processed观测则经过死区、限幅、非线性响应、时间滤波及邻域扩散处理。Real processed观测来自实机自身采集处理流程，因此两域processed均应视为处理后的数值响应，不能直接称为真实压力或共同力学单位。对这种观测进行迁移，需要检查保留了哪些接触信息，而不只是核对输出形状。
+真实与仿真触觉具有相同数组维度，并不意味着能够直接用于同一操作表示。本文采用的观测按16个Patch组织，共712个标量通道，仿真阵列覆盖五指及手掌。其传感器原生raw读数来自MuJoCo native touch，对感知区域内的法向接触力累加\cite{mujoco2026xml}；仿真processed观测则经过死区、限幅、非线性响应、时间滤波及Patch内邻域扩散处理（实现参数名为crosstalk）。Real processed观测来自实机自身采集处理流程，因此两域processed均应视为处理后的数值响应，不能直接称为真实压力或共同力学单位。对这种观测进行迁移，需要检查保留了哪些接触信息，而不只是核对输出形状。
 
 P37旧开发数据的Real Train/Validation为5153/1408窗，Sim为1824/406窗；P43当前主线中，Real Train/Validation分别为5153/780窗，Natural Sim Train/Validation分别为4049/905窗，均采用$4\times712$窗口。P37统计不外推至P43，后者规模仅描述当前数据划分，不代表差异复现。两者不混合，窗口不作独立采集主体。
 
@@ -10,7 +10,7 @@
 
 幅值分析仅取$X>0$的响应。旧训练集汇总正值中位数，Real约为0.00500250，未转换Sim约为$1.03974\times10^{-5}$；P95分别约1.50101与0.459485。这些是各域全部正值池的统计，不是逐Patch分位数的平均，也不是接触力误差。幅值分解以每帧正值的P10、P25、P50、P75、P90构成有效$4\times5$表示，其余通道仅补零以复用网络接口；不输入Taxel位置、正值数量或空Patch标志。对包含至少一帧全零响应帧的窗口，按照预定规则予以排除，故该表示只刻画有响应条件下的幅值，不能覆盖完全无接触状态。
 
-表\ref{tab:ch05_domain_feature_difference}列出旧P37数据上既有转换输出经幅值校准后的表示分解诊断。Full与分位数先作log1p，各表示分别用Train拟合标准化，Probe由Logistic、kNN各1个及MLP、GRU、Transformer各Seed11、12、13组成，共11个；Full复用P38在相同输出、划分和预处理上的Probe拟合，不新增训练。此处为迁移后Probe诊断，非后文冻结C0验收。定义$D=1-2|\mathrm{BA}-0.5|$，只度量BA偏离0.5的程度；表中三个表示的最小D均低于0.90，表明至少存在Probe未满足固定阈值下的BA等价门槛；其排序判别能力仍需结合AUC判断。因此，不能仅凭这些结果认定支持或正值摘要已经实现域混淆。幅值调整保持Mask不变，不代表完整触觉已匹配。
+表\ref{tab:ch05_domain_feature_difference}列出旧P37数据上既有转换输出经幅值校准后的表示分解诊断。Full与分位数先作log1p，各表示分别用Train拟合标准化，Probe由Logistic、kNN各1个及MLP、GRU、Transformer各Seed11、12、13组成，共11个；Full复用P38在相同输出、划分和预处理上的Probe拟合，不新增训练。此处为迁移后Probe诊断，非后文迁移前冻结的域分类评价基线（C0）验收。定义$D=1-2|\mathrm{BA}-0.5|$，只度量BA偏离0.5的程度；表中三个表示的最小D均低于0.90，表明至少存在Probe未满足固定阈值下的BA等价门槛；其排序判别能力仍需结合AUC判断。因此，不能仅凭这些结果认定支持或正值摘要已经实现域混淆。幅值调整保持Mask不变，不代表完整触觉已匹配。
 
 \begin{table}[htbp]
  \centering\small
--- thesis/body/ch05_sections/section_5_2.tex (before)
+++ thesis/body/ch05_sections/section_5_2.tex (after)
@@ -1,10 +1,10 @@
-% Independent section fragment; not included in thesis.tex.
+% Active section fragment; included through body/ch05_dual_domain_reconstruction.tex.
 \section{非配对条件下的跨域触觉重建问题定义}
 \label{sec:ch05_unpaired_problem}
 
 前节表明，真实与仿真触觉即使具有相同观测维度，其激活支持、幅值及短时序关系也不能直接互换。本文以自然任务仿真产生的processed触觉为输入，研究向Real域风格观测的转换。单个窗口包含四帧、每帧712个通道，输入与输出均保留既有通道及帧序。目标输出应为完整$4\times712$的有限非负数值，保留精确零值与非零激活支持，并具有Real域processed统计特征，不将其直接称为真实传感器输出，也不以低维统计摘要代替完整观测。保留帧序不要求两域采样间隔相同，帧间变化也不直接表示物理时间导数。
 
-研究使用三类数据：Real提供真实域观测，用于域内表示学习，并提供统计先验及伪配对目标侧信息；Natural Sim来自自然任务仿真，与Paired Sim共同参与Sim域表示学习，也是最终转换对象；Paired Sim则由真实统计先验指导仿真物理探针生成，按登记关系与Real窗口构成统计伪配对，为潜变量映射提供有限监督。任意Real与Sim窗口并无已知共同接触状态，不能直接互作真实标签。现有1200个训练窗口与240个验证窗口也不代表1440个独立Root；登记关联不构成同步物理接触真值，来源复用与覆盖限制仍须保留。Natural参与Sim表示学习，不等于其具有对应Real目标，亦不等于直接参与原始配对映射训练。
+本章使用第\ref{sec:data_quality_summary}节所区分的单独登记并冻结的触觉子集，不直接读取LeRobot示教导出。其中三类数据分别承担不同角色：Real提供真实域观测，用于域内表示学习，并提供统计先验及伪配对目标侧信息；Natural Sim来自自然任务仿真，与Paired Sim共同参与Sim域表示学习，也是最终转换对象；Paired Sim则由真实统计先验指导仿真物理探针生成，按登记关系与Real窗口构成统计伪配对，为潜变量映射提供有限监督。任意Real与Sim窗口并无已知共同接触状态，不能直接互作真实标签。现有1200个训练窗口与240个验证窗口也不代表1440个独立Root；登记关联不构成同步物理接触真值，来源复用与覆盖限制仍须保留。Natural参与Sim表示学习，不等于其具有对应Real目标，亦不等于直接参与原始配对映射训练。
 
 记Natural Sim窗口为$X_S$，Real域风格输出为$\widehat X_R$，待学习映射为$F_\theta$，问题定义为
 \begin{equation}
--- thesis/body/ch06_sections/section_6_1_shared_vector_field.tex (before)
+++ thesis/body/ch06_sections/section_6_1_shared_vector_field.tex (after)
@@ -1,8 +1,8 @@
-% Standalone draft, 2026-11-26; not included in the thesis main file.
+% Active section fragment; included through body/ch06_adversarial_evaluation.tex.
 \section{共享参数向量场迁移方法}
 \label{sec:ch06_shared_vector_field}
 
-第五章利用统计伪配对监督Translator，在独立训练的双域VAE之间学习直接潜变量映射，再由Real Decoder生成目标域触觉。该方法通过GRU内部状态建模四帧序列，但未显式构造潜变量的积分更新路径；配对潜变量误差也不能直接说明Natural Sim输出的域分布是否接近Real。为在保留既有表示与读出模块的条件下研究可调整的迁移路径，本文将原映射扩展为源条件共享参数向量场，通过数值积分获得目标潜状态。这里的扩展是迁移计算形式的改变，并未重新建立公共物理坐标，也不预设两域潜变量具有物理等价性。
+第五章利用统计伪配对监督Translator，在独立训练的双域VAE之间学习直接潜变量映射，再由Real Decoder生成目标域触觉。该方法通过GRU内部状态建模四帧序列，但未显式构造潜变量的积分更新路径；配对潜变量误差也不能直接说明Natural Sim输出的域分布是否接近Real。为在保留既有表示与读出模块的条件下研究可调整的迁移路径，本文将原映射扩展为源条件共享参数向量场，通过数值积分获得目标潜状态。本节将该共享参数向量场实验记为P44。这里的扩展是迁移计算形式的改变，并未重新建立公共物理坐标，也不预设两域潜变量具有物理等价性。
 
 记Natural Sim输入为$X_s\in\mathbb R_{\ge0}^{4\times712}$，沿用第五章的记号，以冻结Sim Encoder的后验均值$Z_s=\mu_s(X_s)\in\mathbb R^{4\times64}$作为源条件。积分状态$H(\tau)$同样具有$4\times64$形状，位于Real Decoder接收的潜变量坐标中。与直接令输出等于$T_\theta(Z_s)$不同，向量场同时考虑源条件、当前积分状态和积分坐标，其定义为
 \begin{equation}
--- thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex (before)
+++ thesis/body/ch06_sections/section_6_2_discriminator_gradient.tex (after)
@@ -1,4 +1,4 @@
-% Standalone 6.2 draft, not included in the official thesis main file.
+% Active section fragment; included through body/ch06_adversarial_evaluation.tex.
 \section{判别器设计与梯度优化机制}
 \label{sec:ch06_discriminator_gradient}
 
@@ -18,7 +18,7 @@
 \end{equation}
 其中$\operatorname{sg}$表示停止梯度，$\sigma$为sigmoid。前向仍以硬Mask乘Gamma混合幅值均值，反向借用sigmoid导数，解决的是Decoder内部从logit到输出的离散路径。生成触觉再次进入判别器时，后者重新计算$M_D=\mathbf1[Y>0]$；该比较在实现中不保留输入梯度，数学上阈值函数几乎处处导数为零。因此，Decoder的STE不会自动穿过判别器端的再次二值化，但连续幅值支路仍可提供梯度，不能将这一断点写成生成器完全无法训练。
 
-为核查输入表示与梯度机制，设置表\ref{tab:ch06_discriminator_comparison}的三种配置。C组在判别输入端另加STE：
+为核查输入表示与梯度机制，在P44基础上开展编号为P45的结构对照，设置表\ref{tab:ch06_discriminator_comparison}的三种配置。C组在判别输入端另加STE：
 \begin{equation}
  \begin{aligned}
  q(Y)&=\sigma\!\left((Y-\epsilon)/\tau_m\right),\\
--- thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex (before)
+++ thesis/body/ch06_sections/section_6_3_multi_round_adversarial.tex (after)
@@ -1,8 +1,8 @@
-% Standalone 6.3 draft; historical CSV analysis only, not included in thesis.tex.
+% Active section fragment; included through body/ch06_adversarial_evaluation.tex.
 \section{多轮对抗优化实验}
 \label{sec:ch06_multi_round_adversarial}
 
-前述对照区分了判别输入与梯度路径，但尚未回答：判别器适应已有生成结果后，再更新生成场，能否持续改善迁移质量。为建立可比较的起点，先对冻结候选开展统一分布评价。以未转换Natural Sim为Original，将上一节A组训练终点按$K=1$读出的生成器记为$F_0$；该权重本身采用四种$K$调度训练，并非仅用单步训练或重新随机初始化。统一评价未新增生成器训练，其组等权SWD由Original的0.411343降至$F_0$的0.206609，相对下降49.77\%，定义$R_W=(\mathrm{SWD}_{\mathrm{Original}}-\mathrm{SWD}_{F_0})/\mathrm{SWD}_{\mathrm{Original}}$，本次为0.4977202817，95\%组Bootstrap区间约为$[0.408514,0.566743]$，如图\ref{fig:ch06_f0_swd_baseline}所示。这只支持指定距离合同下的改善，不能解释为迁移成功率。
+前述对照区分了判别输入与梯度路径，但尚未回答：判别器适应已有生成结果后，再更新生成场，能否持续改善迁移质量。本节将这一多轮实验记为P49，$F_i$与$D_i$分别表示生成器与判别器的阶段状态，后者不同于第五章由BA定义的域混淆指标$D$。为建立可比较的起点，先对冻结候选开展统一分布评价。以未转换Natural Sim为Original，将上一节A组训练终点按$K=1$读出的生成器记为$F_0$；该权重本身采用四种$K$调度训练，并非仅用单步训练或重新随机初始化。统一评价未新增生成器训练，其组等权SWD由Original的0.411343降至$F_0$的0.206609，相对下降49.77\%，定义$R_W=(\mathrm{SWD}_{\mathrm{Original}}-\mathrm{SWD}_{F_0})/\mathrm{SWD}_{\mathrm{Original}}$，本次为0.4977202817，95\%组Bootstrap区间约为$[0.408514,0.566743]$，如图\ref{fig:ch06_f0_swd_baseline}所示。这只支持指定距离合同下的改善，不能解释为迁移成功率。
 
 \begin{figure}[htbp]
  \centering
--- thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex (before)
+++ thesis/body/ch06_sections/section_6_4_multiview_residual_refiner.tex (after)
@@ -1,7 +1,7 @@
 \section{多视角判别与残差重建探索}
 \label{sec:ch06_multiview_residual_refiner}
 
-多轮优化提高了当前判别器下的接受率，却未维持固定距离评价中的质量。为此，本节探索条件残差Refiner与Global、Spatial、Temporal多视角判别约束。实验从$F_0$重新开始，而非续接上一节$F_3$，采用Seed 5001、单轮训练；两套VAE冻结，共享向量场与Refiner可训练。$D_{50}$是实验标识，不表示训练50轮。
+多轮优化提高了当前判别器下的接受率，却未维持固定距离评价中的质量。为此，本节探索条件残差Refiner与Global、Spatial、Temporal多视角判别约束。本实验记为P50，从$F_0$重新开始，而非续接上一节$F_3$，采用Seed 5001、单轮训练；两套VAE冻结，共享向量场与Refiner可训练。$F_{50}$表示本实验生成器的终点结果，$D_{50}$表示多视角判别器；下标50对应实验编号，不表示训练50轮。
 
 Refiner逐帧拼接$\log(1+Y_0)$与$z_s$，输入776维，经Linear投影至64维、GELU及隐藏维数64的GRU，再由零初始化Linear输出712维残差。其实际读出为
 \begin{equation}
--- thesis/body/ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex (before)
+++ thesis/body/ch06_sections/section_6_6_discriminator_reliability_session_generalization.tex (after)
@@ -1,7 +1,7 @@
 \section{判别器可靠性及跨Session泛化分析}
 \label{sec:ch06_discriminator_reliability}
 
-前文区分了边际分布、邻域覆盖、输入依赖和判别接受情况。进一步作出迁移质量判断，还需审查评价器能否稳定接受真实触觉。本节依据冻结$D_{50}$的历史只读审计，分析真实域接受能力、生成分数及跨Session校准差异，不继续优化生成器。本次P51只读审计未执行训练或参数优化，$D_{50}$权重保持冻结，未改变类别方向；本节仅复核已有记录，不重复历史推理。
+前文区分了边际分布、邻域覆盖、输入依赖和判别接受情况。进一步作出迁移质量判断，还需审查评价器能否稳定接受真实触觉。本节依据编号为P51的冻结$D_{50}$历史只读审计，分析真实域接受能力、生成分数及跨Session校准差异，不继续优化生成器。本次P51只读审计未执行训练或参数优化，$D_{50}$权重保持冻结，未改变类别方向；本节仅复核已有记录，不重复历史推理。
 
 历史模型文件SHA256与审计登记一致，重放记录给出的最大logit差为0，分类结果相同。Real fit为24个Session、4217窗，inner为8个Session、936窗，后者用于包络校准；开发评价使用另外4个Real Session、780窗及10个Sim Root、905窗。分组无交叉，但外层已有开发暴露，不能称为独立盲测。固定Real为1、Sim为0，概率阈值仍为0.5，即logit阈值0；Joint要求三支同时接受，连续Joint分数取三支logit最小值。
 
--- thesis/body/ch07_sections/section_7_1_research_work_summary.tex (before)
+++ thesis/body/ch07_sections/section_7_1_research_work_summary.tex (after)
@@ -1,10 +1,10 @@
 \section{主要研究工作总结}
 \label{sec:ch07_research_work_summary}
 
-针对灵巧手操作中机器人运动、接触感知与任务执行需要协同组织的问题，本文构建了RM75B机械臂与途见科技自研欠驱动灵巧手的多任务仿真平台。在MuJoCo环境中集成机器人模型、控制器、任务逻辑和触觉记录模块，结合曲面几何布局建立16个Patch、712维processed触觉观测体系，明确感知区域、原生读取与信号处理之间的关系。通过多任务接口及指定历史案例的观测检查，形成支撑后续抓取和触觉研究的仿真基础。这部分工作侧重系统集成与感知建模，其验证范围为所列模型实现和记录，不等同于实机触觉标定或真实机器人部署。
+以下按第\ref{sec:research_contents}节的四项具体工作总结，其中抓取示教与数据质量控制共同对应专家示教与数据主线。针对灵巧手操作中机器人运动、接触感知与任务执行需要协同组织的问题，本文构建了RM75B机械臂与途见科技自研欠驱动灵巧手的多任务仿真平台。在MuJoCo环境中集成机器人模型、控制器、任务逻辑和触觉记录模块，结合曲面几何布局建立16个Patch、712维processed触觉观测体系，明确感知区域、原生读取与信号处理之间的关系。通过多任务接口及指定历史案例的观测检查，形成支撑后续抓取和触觉研究的仿真基础。这部分工作侧重系统集成与感知建模，其验证范围为所列模型实现和记录，不等同于实机触觉标定或真实机器人部署。
 
-针对几何抓取候选不能直接转化为机器人可执行操作的问题，本文适配GraspQP官方抓取评价指标，参考DexEvolve设计实现进化搜索，并结合腕部姿态扩展形成多来源候选。通过机械臂逆运动学、碰撞约束和连续轨迹求解，将所选候选连接为分阶段专家示教轨迹，再结合执行反馈、任务结果与Strict检查验证轨迹，区分操作完成与示教质量。局部扰动轨迹和失败分类为流程的适用范围提供了历史依据，并形成后续数据记录的轨迹来源。该工作体现为现有工具的适配整合与验证流程构建，不将第三方算法本身作为原创，也不把候选质量、Strict通过与实际任务成功合并为同一结论。
+针对几何抓取候选不能直接转化为机器人可执行操作的问题，本文适配GraspQP官方抓取评价指标，参考DexEvolve设计实现进化搜索，并结合腕部姿态扩展形成多来源候选。通过机械臂逆运动学和连续轨迹求解连接分阶段专家示教，并按第\ref{sec:grasp_reachability}节所述，结合桌面几何惩罚、局部接近约束、执行监测和尾段Strict检查验证轨迹与任务结果，区分操作完成与示教质量。局部扰动轨迹和失败分类为流程的适用范围提供了历史依据，并形成后续数据记录的轨迹来源。该工作体现为现有工具的适配整合与验证流程构建，不将第三方算法本身作为原创，也不把候选质量、Strict通过与实际任务成功合并为同一结论。
 
 在专家示教基础上，本文围绕数据语义、采集时序和来源追踪建立多源数据组织与质量控制体系。通过区分策略观测与完整仿真状态、位置命令与逆运动学目标，明确状态和动作合同，检查动作命令与执行前后观测的时序配对关系，并完成NPZ、Manifest与LeRobot之间的转换组织。质量检查覆盖数值有效性、连续性、完整性、版本和重复关系，既核验了指定修复数据的字段映射，也保留旧记录中的命令不连续与状态字段缺失问题。由此形成从采集、转换到训练准入的分层检查依据，但指定样本修复不代表全量生产认证，文件Hash也不能替代来源独立性或全局泄漏审计。
 
-针对仿真与真实触觉存在分布差异且缺少认证同步物理配对的问题，本文利用真实统计先验与仿真探针构建伪配对，分别学习独立双域VAE表示，再由GRU Translator建立潜变量映射。在此基础上研究共享参数向量场、对抗优化及多视角判别与残差修正，并结合分布距离、邻域覆盖、输入依赖和判别器跨Session可靠性开展综合评价。这些工作以指定触觉子集为研究对象，完成了统计监督、跨域重建与质量诊断的方法探索链路，不意味着全部平台数据均参与训练。现有实验获得了限定指标下的局部改善，同时揭示接受率、分布质量和评价可靠性的差异，尚未完成完整触觉Sim-to-Real迁移验收。
+针对仿真与真实触觉存在分布差异且缺少认证同步物理配对的问题，本文利用真实统计先验与仿真探针构建伪配对，分别学习独立双域VAE表示，再由GRU Translator建立潜变量映射。在此基础上研究共享参数向量场、对抗优化及多视角判别与残差修正，并结合分布距离、邻域覆盖、输入依赖和判别器跨Session可靠性开展综合评价。这些工作使用单独登记并冻结的触觉子集，沿用前述数据管理与质量规范，但不将LeRobot导出直接作为跨域模型的训练输入，也不认证两套数据的一一对应。由此形成统计监督、跨域重建与质量诊断的方法探索链路，不意味着全部平台数据均参与训练。现有实验获得了限定指标下的局部改善，同时揭示接受率、分布质量和评价可靠性的差异，尚未完成完整触觉Sim-to-Real迁移验收。
--- thesis/body/ch07_sections/section_7_2_main_research_conclusions.tex (before)
+++ thesis/body/ch07_sections/section_7_2_main_research_conclusions.tex (after)
@@ -1,8 +1,8 @@
 \section{主要研究结论}
 \label{sec:ch07_main_research_conclusions}
 
-本文的仿真实现与指定案例表明，统一机器人模型、控制接口、任务接口和触觉观测合同，能够为多任务操作提供可复用的实验基础。抓取候选仍需经过可达性、碰撞约束、连续轨迹及任务执行的分层检验：几何质量不能替代IK可达性，可达性不能替代任务成功，任务成功也不能替代Strict示教质量。各层判断针对不同条件，当前证据仅支持已有模型、任务及案例范围，不能外推为通用抓取能力或实机有效性。
+本文的仿真实现与指定案例表明，统一机器人模型、控制接口、任务接口和触觉观测合同，能够为多任务操作提供可复用的实验基础。候选转化为轨迹时，可达性筛选、桌面几何与局部接近约束、连续执行监测及尾段Strict检查承担不同层次的检验：几何质量不能替代IK可达性，可达性不能替代任务成功，任务成功也不能替代Strict示教质量。各层判断针对不同条件，当前证据仅支持已有模型、任务及案例范围，不能外推为通用抓取能力或实机有效性。
 
 示教数据的可用性取决于语义和记录关系，而不只取决于文件能否读取或转换。策略观测与完整仿真状态、位置命令与IK目标需要区分，动作与观测的时序配对、数值有效性、版本来源和重复关系也应分别审查。指定修复与检查支持相应样本满足对应条件，历史命令不连续和状态字段缺失则说明格式合法不足以认定训练资格。文件Hash、Manifest和规模只能提供部分追踪依据，不能单独证明来源独立、全局无泄漏或物理真实性，也不能将局部修复扩大为全量生产认证。
 
-跨域实验显示，局部指标改善尚不足以证明完整迁移。C27在原冻结Validation的11个实例中仅2个联合通过；$F_0$相对Original的SWD下降49.77\%，仅为指定合同下的边际改善。多轮对抗终点在$D_3$下FR达100\%，不能据此认定域不可辨识；多视角残差终点$F_{50}$虽Joint FR为100\%，分布距离却明显退化。P51因未满足既定Real接受要求对$D_{50}$判定NO-GO，该判别器不能作为当前实验的最终可靠评价器，该结论不外推$D_3$或其他方法。由此可见，接受率不等于分布相似，边际相似不保证输入输出对应，统计依赖不等于物理语义，梯度可达也不保证优化有效。评价器可靠性同样是迁移结论的前提。本文完成了统计伪监督、潜变量映射、共享参数向量场及多视角评价的系列探索，但现有证据尚不能证明完整触觉Sim-to-Real迁移成功。
+跨域实验显示，局部指标改善尚不足以证明完整迁移。C27在原冻结Validation的11个实例中仅2个联合通过；$F_0$相对Original的SWD下降49.77\%，仅为指定合同下的边际改善。在第\ref{sec:ch06_multi_round_adversarial}节披露的修订后开发协议下，多轮对抗终点在$D_3$下FR达100\%，不能据此认定域不可辨识；多视角残差终点$F_{50}$虽Joint FR为100\%，分布距离却明显退化。P51因未满足既定Real接受要求对$D_{50}$判定NO-GO，该判别器不能作为当前实验的最终可靠评价器，该结论不外推$D_3$或其他方法。由此可见，接受率不等于分布相似，边际相似不保证输入输出对应，统计依赖不等于物理语义，梯度可达也不保证优化有效。评价器可靠性同样是迁移结论的前提。本文完成了统计伪监督、潜变量映射、共享参数向量场及多视角评价的系列探索，但现有证据尚不能证明完整触觉Sim-to-Real迁移成功。
```
