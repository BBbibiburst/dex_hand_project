# 5.6 V1 TeXcount

实测**1535**，目标1500–1750，中心1605。正文/图表文字1486、标题13、图表题36；10个行内数学另计，显示公式0组。图1、表2，共3浮动体。

复用项目Counter、原scripts/texcount_rules.tex，TeXcount3.1.1中文UTF-8模式，未改规则。输出全部保存在本目录wordcount/；raw/section_5_6.txt为原始统计，wordcount_audit.html为逐字解析，commands.json为实际命令。

中文按字、英文按词；表内数字和术语按TeXcount规则纳入正文，引用标签/注释/数学符号不计。没有为了中心目标追加冗余叙述。独立小节未接入主文件，此值不是全第五章累计统计。
