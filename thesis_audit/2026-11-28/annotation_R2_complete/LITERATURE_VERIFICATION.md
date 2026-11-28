# 新增文献核验

新增4条，均取得原始PDF并核对相关方法段落。PDF及逐页提取文本仅保存于本审计目录，未将他人图表复制进论文。原有Bib条目保留原字节前缀，不回填未知DOI、页码或地址。

|引用键|原件与出版身份|核对位置及正文用途|
|---|---|---|
|zhu2020robosuite|Zhu、Wong、Mandlekar、Martín-Martín，2020，arXiv:2009.12293v1|明确锁定四作者v1，不混用2025九作者v3。读取原文系统模块、任务与环境配置，用于说明MuJoCo与操作框架的分工。以online条目适配现有bst，未虚构会议出版。|
|xiang2020sapien|Xiang等，CVPR 2020，CVF官方论文页与PDF|原文平台构成、PhysX/资产/渲染及关节物体操作。只说明平台侧重点，不得推导其他引擎无法欠驱动。|
|mu2021maniskill|Mu等，NeurIPS Datasets and Benchmarks，2021，卷1|核对原PDF首页九名作者；官网摘要页重复列出Xuanlin Li，按原PDF与arXiv交叉核对后仅列一次。读取任务、视觉观测及示教用途，不引用其数据量来充本文成果。|
|qin2023anyteleop|Qin等，RSS 2023，DOI 10.15607/RSS.2023.XIX.015|RSS官方书目与PDF一致。绪论、系统及Failure Modes支持重定向、部署和自遮挡困难；作者个人操作经验另以用户批注为定性来源，不归给该论文。|

来源：[robosuite v1](https://arxiv.org/abs/2009.12293v1)、[SAPIEN官方页](https://openaccess.thecvf.com/content_CVPR_2020/html/Xiang_SAPIEN_A_SimulAted_Part-Based_Interactive_ENvironment_CVPR_2020_paper.html)、[ManiSkill原PDF](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/eda80a3d5b344bc40f3bc04f65b7a357-Paper-round2.pdf)、[AnyTeleop正式书目](https://www.roboticsproceedings.org/rss19/p015.html)。下载URL、SHA256、字节数见literature/sources.json。

这4篇针对具体批注补充背景，不等于长期计划中每个研究点5–6篇直接相关论文的系统调研已完成；没有宣称外部算法已在本文平台公平复现或比较。
