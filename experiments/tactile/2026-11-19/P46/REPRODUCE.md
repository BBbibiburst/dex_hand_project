# P46 独立复现

在项目根目录运行。依赖 numpy、scipy、torch、matplotlib、PyYAML；原始注册产物必须存在。无需 sklearn。

为避免覆盖此次结果，将 `code/` 完整复制至一个**全新的**输出目录下的 `code/`。`common.py` 以自身所在目录推导输出位置。不要删除现有完成标记强行重跑。

按顺序用 mujoco 环境 Python 执行新目录中的：

1. `prepare.py`
2. `compute_distances.py`
3. `c2st.py`
4. `report.py`
5. `finalize.py`

设置 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。每步日志分别保存为 run.log、statistics.log、c2st.log、report.log；最后一步汇总 run.log 并生成 integrity.json。所有代码运行目录保持项目根目录。

准备阶段逐行校验允许读取的 Real 数值角色及 Natural 文件/窗口 Hash。不会加载 Test/Holdout 数值数组。完整历史文件 Hash 校验仅用于保护性完整性核对。

`integrity.json` 的 outputs 可逐项重算 SHA256；不要把时间耗时、日志和运行环境信息要求逐字节相同。固定数据、随机种子和 CPU 实现下的主要距离、组划分和分类结果应一致，其他软件版本需考虑浮点误差。

当前 C2ST 的 100 次 LBFGS 是固定预算；收敛警告完整保存。不为改善成绩追加优化。
