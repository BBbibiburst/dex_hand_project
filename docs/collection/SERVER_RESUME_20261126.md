# 原服务器恢复采集（Git 更新版）

代码仓库不包含原队列、历史示教、冻结模板和完整运行环境。续采请使用原服务器项目 `/home/ftcl/dex_hand_project`，不要用空 clone 替代它。保留原 `outputs/five_task_speed_batch2_control` 及所有冻结资源。

硬盘程序包目录是 `/media/ftcl/外置存储1/collection_quality_fix_20261126`，**数据目录**通常是其同级 `/media/ftcl/外置存储1/dex_hand_dataset_batch2`。以下命令使用 Git 仓库内脚本，无需调用硬盘包中的旧 start.sh。

```bash
# 如已安装服务，先停止；若不存在该服务，可以跳过这一条。
systemctl --user stop dex-collection.service
cd /home/ftcl/dex_hand_project
git pull --ff-only origin main
conda activate mujoco
# 显式固定为当前环境的 Python，尤其是环境不在 miniconda3 标准位置时。
export COLLECTION_PYTHON="$(command -v python)"
bash scripts/collection.sh deploy "/media/ftcl/外置存储1/dex_hand_dataset_batch2"
bash scripts/collection.sh check
# 只有 deploy/check 成功后再安装并启动。
bash scripts/collection.sh install-service
journalctl --user -u dex-collection.service -f
```

若 git pull 提示本地修改冲突，停止并保留修改，不要 reset --hard。deploy 不启动采集；它会锁定协调器、备份原 SQLite 及配置，并验证原冻结合同。只批准指定九个生产修复文件的指纹变化、挂载路径变化，以及对应恢复索引指针和看门狗路径。运行环境、物理资产、模板、配额及 Seed 不允许漂移。缺失原队列时拒绝创建新空队列。迁移失败恢复旧元数据；跨批次引用账本可能已幂等写入，但不改变历史时长。

路径迁移保留历史 Episode 原文件，旧记录中的路径作为来源记录保留。新进度从原 SQLite 接续，不按文件数猜进度。若检测到非预期指纹变化、源文件缺失、版本不符或数据库损坏，请保留错误和备份，不要手工改冻结 hash 绕过检查。

默认自动恢复有重启预算、退避、资源等待及有限候选探测；人工停止和数据完整性错误不会无限重启。配置完毕可用：

```bash
bash scripts/collection.sh status
systemctl --user status dex-collection.service --no-pager
systemctl --user stop dex-collection.service
```

用户服务默认登录后启动。服务器无人登录时的开机启动需要管理员为 ftcl 启用 linger（本工具不会执行 sudo 或更改系统配置）。本地修复和测试不代表已在服务器启动。
