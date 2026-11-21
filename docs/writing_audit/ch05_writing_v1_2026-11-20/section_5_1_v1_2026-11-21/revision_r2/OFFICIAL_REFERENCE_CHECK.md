# Native touch官方引用核查

本地reference.bib已有mujoco2026xml（MuJoCo Contributors，MuJoCo Documentation: XML Reference，URL https://mujoco.readthedocs.io/en/stable/XMLreference.html，既存citedate=2026-11-20）。本轮复用条目，不改作者、日期或URL。

本轮2026-11-21访问官方[sensor/touch](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-touch)条目：说明site定义感知体积，满足接触点/法向射线条件的接触纳入读数，输出为非负标量，由纳入接触的标量法向力相加形成。因此支持正文native touch法向力累加的概括。页面也说明同body等纳入条件，正文是机制简述，不声称任意空间范围内所有力都被累计。

官方文档仅支持MuJoCo原生传感器语义，不替项目证明712布局、后处理参数、真实Taxel配准或实机串扰标定。项目实际使用native touch仍以既有P40代码/模型审计为依据；本轮不重新实例化MuJoCo。

检索方式：官方XMLreference页面open与find，相关条目sensor/touch。未新增文献或引用未经核验来源。
