# 单模板串行瓶颈修复（2026-11-27）

## 根因与修复

资源规划24worker仅给出了上限。旧enqueue_next只要看到某模板active>0就跳过；当前NutAssembly仅一个可用模板，导致该任务最多1条执行。清除CPU限流、扩大看门狗恢复批量都没有解除这个调度限制。

生产协调器现将实际worker池大小传给调度器，允许同模板不同candidate_index并发，仍受整个worker池和内存准入约束。候选按原固定采样合同生成，job_id独立，索引从原last_candidate_index继续，不覆写历史任务。没有把重复几何条件说成新独立物体。

保持不变：Strict验收、独立回放、失败/重复隔离阈值、每模板恢复预算、历史进度、Seed、训练/数据合同。Lift因FINAL Strict失败被隔离是另一项质量问题，不能靠并发修复宣称已解决或降低门槛。

## 必须重新部署冻结代码指纹

本次修改了production_scheduler.py和production_coordinator.py，属于冻结执行代码。不能仅pull后直接启动，也不能手动覆盖run_contract的Hash。

新增configs/collection_scheduler_patch_20261127.json明确记录旧源码、上一质量修复版及新源码的允许指纹。deploy工具合并基础质量修复与这两项调度变化，备份原SQLite和元数据，仅批准这些已登记变化；环境、物理资产、其他源文件仍严格核对。历史原manifest不改写。此迁移不重建数据库、不清除失败计数。

## 服务器更新并续采

```bash
bash <<'BASH'
set -eo pipefail
cd /home/ftcl/dex_hand_project
systemctl --user stop dex-collection.service

git fetch --no-tags origin main
git merge --ff-only origin/main
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mujoco
export COLLECTION_PYTHON="$(command -v python)"

# 本次必须执行：批准两项已登记源码更新，备份并更新冻结合同。
bash scripts/collection.sh deploy "/media/ftcl/外置存储1/dex_hand_dataset_batch2"
bash scripts/collection.sh check
systemctl --user reset-failed dex-collection.service || true
bash scripts/collection.sh install-service
sleep 3
bash scripts/collection.sh status
BASH
```

任何关键步骤失败都会停止；不reset --hard、不删除原队列和恢复状态。准备结束进入COLLECTING后再看worker数，资源上限不是任务数承诺。

```bash
journalctl --user -u dex-collection.service -n 40 --no-pager
```

## 验证边界

- SQLite回归检查：单Nut模板可提供24条同时活跃、job_id及候选索引不同的任务，达到24后停止增加；失败隔离阈值继续生效；重新打开数据库后索引继续。
- 协调器集成检查：使用四个模拟worker及同步屏障，要求四个全启动后任何一个才可结束；确认旧单模板限制不再把worker池变成串行。不是物理仿真或采集性能实验。
- 部署回归检查：上一版已登记Hash可迁移、未登记源码或环境变化拒绝、既有进度及恢复状态保留。
- 服务器实际吞吐及Lift Strict失败原因仍需服务器结果，不以本地模拟测试冒充实机部署成功。

## 在线恢复后续修复

Lift空闲而Nut持续运行时，恢复探测不能等待整场会话结束。参见[在线任务恢复](LIVE_TASK_RECOVERY_20261127.md)，包含运行中补充、预算保护及联动测试说明。
