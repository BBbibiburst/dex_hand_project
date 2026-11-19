# P45架构对照与受控范围

共同生成器始终为P44初始G：冻结Es→唯一时间条件共享field（原Translator初始化）→冻结Dr。Dr的硬Mask、训练期ST反向、Gamma幅值结构均未改变。h0仍为固定零状态，不是Sim latent。

|组|D输入|前向|G侧反向|输入层参数|
|---|---|---|---|---|
|A|原完整连续log1p触觉/固定scale，2848维|不计算Mask|连续路径|原P44首层仅保留幅值半列|
|B|硬Mask+连续，5696维|原P44|Mask分支为0|原P44，历史结果|
|C|硬Mask+连续，5696维|逐值等于B|Mask采用sigmoid代理导数|原P44全部权重|

A不是删去触觉零值，也不是删去生成器的Mask。零和正值均原样进入完整连续712D。

C的epsilon=0，tau为Real Train全部正值中位数；hard+(soft-soft.detach())保证前向精确硬Mask。只在G反传中启用Mask代理；Real侧R1用mask_backward=False，保持P44原正则化路径，避免另改正则项。

A/C均从同一P44训练前Generator加载。C与B的初始D全部参数相同；A的公共层相同、首层拷贝原幅值列，因此A有559809参数而B/C有1106625，不能声称总参数完全相同。

每组只训练一个G/D，四个K共享同一组权重，K不等于触觉四帧时间轴。
