# 快速生成与服务器运行

日期：2026-11-06。按当前要求，允许复用、重复和降低图像质量；100种物体按五任务合计统计，不要求每任务100种。

## 已生成的数据

`outputs/fast_collection_2026-11-06/expanded_dataset/` 已保存443份历史源轨迹及127,526条采样索引，覆盖101个独立资产、五个任务各约200小时，合计1000.02346小时扩容样本。目录可整体迁移；源数组只保存一次，索引允许重复读取。

原始来源合计约4.02893小时，按历史帧数除以20Hz计算，原文件没有可核验的逐帧时钟。新增仿真为0小时。这里没有RGB，也没有补造历史缺失的触觉；数组保留原始布局。这是可读取的重复采样数据集，不是1000小时全模态新仿真数据。

```python
from source.data.dataset_expansion import ExpandedTrajectoryDataset
data = ExpandedTrajectoryDataset('outputs/fast_collection_2026-11-06/expanded_dataset')
sample = data[0]
actions = sample['arrays']['action']
```

需要重新构建时，使用新的输出目录：

```bash
PYTHONPATH=. python apps/expand_existing_dataset.py --output outputs/expanded_copy --hours 1000 --minimum-assets 100
```

## 新采集配置

若启动时出现 `Template dependency changed`，先停止采集，在项目目录运行 `python apps/restore_production_inputs.py --archive five_task_production.zip`。它先核验迁移包，再备份并恢复不一致的模板/资产；不修改代码或迁移包以外的采集结果。恢复后用新的 `serve --output` 控制目录启动，不要修改清单哈希来绕过错误。

`configs/five_task_speed.json` 将RGB改为640×360、5Hz、x264 ultrafast CRF28；状态、动作和触觉仍按20Hz记录。每张图像保存对应控制行索引，中间动作不会因图像降频而删除。合法失败轨迹允许计入时长，但不会标成成功。新采集链路仍保留去重；允许重复扩容使用上面的独立入口。

服务器无需预先填写硬件规格。程序检测CPU配额、可用内存并限制并发启动；内存不足会等待。默认每worker预留2GiB，最多64个，子进程计算线程默认2个。这是资源准入控制，不保证所有硬件上都能达到最佳吞吐，GPU显存尚未自动调优。

Git只保存程序和配置，历史模板、轨迹和下载资产不在Git中。新服务器请将已校验的 `outputs/fast_collection_2026-11-06/server_delivery/five_task_production.zip` 解压到新目录，再同步到最新提交；只clone仓库还不能开始采集。扩容数据如需使用，另行复制整个 `expanded_dataset/` 目录。

迁移项目及其模板/资产后，安装依赖并在服务器指定真实数据目录：

```bash
export PYTHONPATH=.
export MUJOCO_GL=egl
export DEX_DATASET_ROOT=/data/dex_hand_dataset_speed
python -m pip install -e '.[production]'
python apps/five_task_production.py preflight --config configs/five_task_speed.json
python apps/five_task_production.py serve --config configs/five_task_speed.json
```

数据路径必须换成服务器实际数据盘。启动前检查会保留20%磁盘空间；模板中的旧项目绝对路径会优先解析到当前项目。旧1080p20配置保持独立。

## 一周工期

本机单条Push F2实测：31.3秒物理时长，626行20Hz控制数据，157帧640×360@5Hz图像，完整运行33.66秒，独立文件验收通过，任务成功。相同条件旧1080p测试约99.9秒；这是单条开发对照，不能代表五任务或服务器持续吞吐。

一周完成1000小时新仿真，需要整机持续有效吞吐至少5.95倍实时。没有目标服务器实测前不能保证。历史重复采样扩容已经完成；它不需要等待1000小时物理执行。如果需要RGB，应另做真实渲染导出，不能把本次无RGB扩容当成完成了视觉数据采集。
