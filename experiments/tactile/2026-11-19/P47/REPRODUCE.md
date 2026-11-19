# P47 复现

在原项目根目录、mujoco Python 环境运行。依赖 numpy/scipy/matplotlib/PyYAML，无需训练工具或sklearn。

为避免覆盖，先把本目录的 code/ 和 references/ 复制到一个全新结果目录。common.py 以新目录推导输出路径；历史P44–P46注册数据路径仍相对原项目根目录。

设置 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。

顺序运行新目录 code/ 下：design.py、prdc_eval.py、dependence.py、rr_bootstrap.py、report.py、verify_controls.py、finalize.py。

将stdout/stderr分别保存到run.log、prdc.log、dependence.log、rr_bootstrap.log、report.log、control_checks.log、finalize.log。不要删除已有完成标记强行覆盖本结果。可参考run_commands.json的实际Python解释器、代码Hash和环境。

所有拟合只涉及固定Train描述子统计；HSIC按预声明规则计算开发数据自身中位数带宽，是开发诊断，不是盲验。没有生成器/C0推理或新网络训练。

PRDC主抽样100次与300次聚类贡献Bootstrap是不同不确定性来源，不能混用；后者固定邻域。完整DataFrame包含全部k和Root，不要只读取报告示例k5作为验收k。

历史日志保留了可选参考许可证下载404与初次启动的前置完成标记检查失败；未影响指标定义、数据或训练。统计前修正Root均值的数值求和方式，使常量输入严格保持相同均值，未依成绩调整规则。恒定描述子的Spearman未定义记录在undefined_statistics.json，不作0分或有效关联。

integrity.json覆盖所有最终产物（自身及尚在写的finalize.log除外）和历史受保护输入；可逐项重算SHA256。计时/日志在不同机器上不要求逐字节复现。
