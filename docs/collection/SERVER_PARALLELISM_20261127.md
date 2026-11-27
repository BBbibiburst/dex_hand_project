# 服务器并发恢复修正（2026-11-27）

本次只修正看门狗默认限流，不改变冻结模型、采集合同、成功/质量门槛和历史进度。

- 默认`initial_cpu_limit=0`：继承当前服务实际可用CPU，不额外限制为8个逻辑CPU。
- 默认`maximum_new_probes=0`：恢复候选批量按原生产资源规划的初始worker数确定，不固定为4。
- 48逻辑CPU、约64GB内存且每worker2线程的机器，原配置预检为24worker；这是资源规划上限，实际忙碌数量仍受可领取候选、任务配额、恢复冷却和内存影响，不保证一直24个任务同时执行。
- 已有默认v3状态首次启动时自动备份到`autorecovery_quality_v3/state.before-resource-adaptive-20261127.json`，只迁移这两个默认选项及默认CPU上限，保留sessions、重启时间戳、备份清单及队列恢复预算。显式自定义策略拒绝静默覆盖；已发生的内存降级保留。
- 内存不足、异常终止的有限自动恢复仍保留。旧硬盘离线包不会自动更新，请继续用Git仓库内`scripts/collection.sh`。

## 从停止旧服务到重新启动

“从零开始”指按完整步骤重新启动，不是清空或重建已有数据。必须在服务器原项目目录执行；不要删除SQLite、恢复状态或历史Episode。

一次粘贴以下整块，任一步失败都会停止。原服务器日志显示Conda位于`$HOME/miniconda3`，使用该环境：

```bash
bash <<'BASH'
set -eo pipefail
cd /home/ftcl/dex_hand_project
if systemctl --user cat dex-collection.service >/dev/null 2>&1; then
  systemctl --user stop dex-collection.service
fi

git fetch --no-tags origin main
git merge --ff-only origin/main
git log -1 --oneline

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mujoco
export COLLECTION_PYTHON="$(command -v python)"

# 仅首次尚未完成挂载迁移时执行；当前已采集的服务器自动跳过。
if [ ! -f outputs/five_task_speed_batch2_control/collection_deployment.json ]; then
  bash scripts/collection.sh deploy "/media/ftcl/外置存储1/dex_hand_dataset_batch2"
fi
bash scripts/collection.sh check
systemctl --user reset-failed dex-collection.service || true
bash scripts/collection.sh install-service
bash scripts/collection.sh status
BASH
```

随后查看日志：

```bash
journalctl --user -u dex-collection.service -n 60 --no-pager
journalctl --user -u dex-collection.service -f
```

正常默认状态应显示`cpu_limit: 0`及`cpu_limit_mode: inherit-server-affinity`；0代表不额外限CPU，并非零个worker。CHILD_STARTED事件列出`inherited_cpu_count`；原生产协调器仍自行按CPU/RAM及原配置决定worker池。状态内check.worker_plan是预检值，不能当作已执行任务数。

如果Git网络中断，整块会在fetch处停止；重跑整块即可。若Git提示本地修改冲突，保留本地修改，不用reset --hard强行覆盖。

看板仍在另一个终端运行：

```bash
cd /home/ftcl/dex_hand_project
conda activate mujoco
PYTHONPATH=. python apps/watch_collection.py \
  --config "/media/ftcl/外置存储1/dex_hand_dataset_batch2/config.json" --interval 5
```

## 后续必需修复

若worker池24但单模板仍只执行1条，参见[单模板串行瓶颈修复及必须执行的deploy步骤](SCHEDULER_PARALLELISM_FIX_20261127.md)。这次涉及冻结执行源码，不能跳过deploy。
