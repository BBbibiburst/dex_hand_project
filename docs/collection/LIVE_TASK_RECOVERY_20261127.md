# Lift空闲时的在线候选恢复（2026-11-27）

## 修复对象

此前有两个不同问题：单Nut模板的串行限制已经由8dd9500修复；Lift候选被隔离后，看门狗只在新会话准备阶段补充探测任务，Nut持续生产会使Lift长时间没有恢复机会。这次修复第二项，不改变Strict判定。

## 当前行为

- 看门狗在同一采集子进程运行期间，每60秒检查一次任务供给，不为了补Lift停止Nut。
- 只对尚未完成配额、且没有PENDING/RUNNING/VALIDATING任务的类别考虑恢复。Nut繁忙不再阻止Lift探测；Lift已有待执行或执行中任务时，不重复堆积。
- 每模板仍须有历史成功记录，保持原15分钟冷却和最多连续3次无新增成功探测预算。队列事务内分配新candidate_index，不能覆盖旧job、重置失败或重复计数。
- 每批恢复候选按worker池容量确定；候选排队不以已被运行worker占用后的剩余RAM重新缩小为一两个。实际进程数仍由原CPU/RAM准入机制控制，不增加总worker上限。
- 完成的任务不补采；预算耗尽明确记录NO_PROBE及isolated列表，而不是无限重试或偷偷清除失败隔离。
- 目录/数据集合同、模板及registry Hash在在线恢复中继续检查；不读取或更改模型权重。发现完整性异常仍停止，而非绕过。

正常生产调度器在worker释放后优先领取已入队的Lift任务。探测成功后，原成功登记逻辑允许该模板恢复正常调度；探测失败继续执行原隔离与预算，不保证物理上不可执行的模板会成功。

## 观察方式

```bash
journalctl --user -u dex-collection.service --since '10 minutes ago' --no-pager \
  | grep -E 'LIVE_RECOVERY|CHILD_STARTED|STATE_CHANGED'
bash scripts/collection.sh status
```

- LIVE_RECOVERY_PROBES_QUEUED：在线补入候选。
- LIVE_RECOVERY_STATUS：没有需要补充的空闲任务、仍在冷却，或预算已耗尽等状态。
- status中的live_recovery保存最近检查、starved_tasks、added、isolated及next_retry_at。看到NO_PROBE不应手工删状态文件骗取新预算。

## 更新

这次本身只改看门狗，不再修改冻结生产源码。但若服务器尚未部署8dd9500，必须同时安装其单模板并发修复。因此统一执行下列完整步骤，不清空历史进度：

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
bash scripts/collection.sh deploy "/media/ftcl/外置存储1/dex_hand_dataset_batch2"
bash scripts/collection.sh check
systemctl --user reset-failed dex-collection.service || true
bash scripts/collection.sh install-service
BASH
```

## 已执行的回归场景（模拟worker，不是物理采集实验）

1. Nut保持RUNNING，在线向30个候选Lift模板中的24个补入独立任务，Nut租约不变；再次检查不重复增加。
2. Nut持续RUNNING期间，Lift的15分钟冷却、三次失败预算及数据库重开后的预算持久化仍有效。
3. 一个真实测试子进程持续观察队列，看到Lift在线补入，确认Nut仍RUNNING、子进程未重启。
4. 协调器四个模拟Nut worker全部运行，在线加入Lift；释放一个Nut worker后，Lift在其余三个Nut仍RUNNING时被领取执行，整个协调器没有重启。
5. 修改模板Hash后在线补充被拒绝，不产生新Lift任务。

本地回归证明调度链路和边界生效，不冒充服务器已经满负荷运行，也不证明Lift轨迹质量通过。
