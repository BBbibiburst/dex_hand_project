# 安装与服务器部署

本文给出从 Git clone 到服务器后台运行的完整流程。命令应在项目根目录执行。

## 系统要求

- Linux（本机 WSL 或服务器均可）
- Python 3.10+
- Git 与 Git LFS（仓库若使用 LFS 时）
- 充足的 CPU 和内存；全量流水线当前不依赖 Isaac Sim
- 无显示服务器可使用 MuJoCo headless 模式

## 从 Git 安装

```bash
git clone --recurse-submodules <repository-url> dex_hand_project
cd dex_hand_project

conda create -n mujoco python=3.10 -y
conda activate mujoco
python -m pip install --upgrade pip
python -m pip install -e ".[dev,assets]"
python -m pip install -e "deps/graspqp/graspqp[lite]" --no-build-isolation
```

如果 clone 时没有初始化 submodule：

```bash
git submodule update --init --recursive
```

`deps/graspqp` 不应通过 `.gitignore` 替代安装；它是仓库记录的 Git submodule，服务器必须
初始化并安装其中的 Python package。

## 物体资产

检查：

```bash
test -f assets/maniskill/manifest.json && echo "manifest OK"
find assets/maniskill/models -maxdepth 1 -type l | head
```

缺少资产时：

```bash
python tools/download_maniskill_objects.py
```

下载器会准备 YCB 和 EGAD，并生成 manifest。若通过完整压缩包部署且其中已有资产，不需要
重复下载。

## 部署验证

```bash
python -m pytest -q
python -m tools.run_smoke_checks --steps 2
python -m tools.grasping.benchmark_catalog \
  --full-pipeline --object-id ycb:003_cracker_box \
  --output configs/grasps/dex_hand/server_smoke_benchmark.json
```

单物体全流水线仍可能耗时较长；只需要确认依赖能够导入并开始搜索。

## 后台运行

首次运行：

```bash
mkdir -p logs
nohup python -m tools.grasping.benchmark_catalog \
  --full-pipeline \
  > logs/full_pipeline.log 2>&1 &
```

查看进程和日志：

```bash
pgrep -af "tools.grasping.benchmark_catalog"
tail -f logs/full_pipeline.log
```

程序自动推断并行度，启动日志类似：

```text
parallelism=auto available_cpus=32 available_memory=61.4GiB selected_jobs=8 selected_evolution_jobs=1
workers: objects=8 evolution_per_object=1 maximum_process_parallelism=8
```

推断会读取 CPU affinity 与 cgroup 内存限制，预留一个 CPU 和 1 GiB 内存，每个 worker
按约 2 GiB 预算，最大并行度为 8。显式参数优先：

```bash
python -m tools.grasping.benchmark_catalog \
  --full-pipeline --jobs 4 --evolution-jobs 1
```

`jobs × evolution_jobs` 不允许超过 8。

## 中断与恢复

每完成一个物体，benchmark 都通过临时文件原子更新。突然断电通常只会丢失当时正在运行
的物体。

普通恢复：

```bash
nohup python -m tools.grasping.benchmark_catalog \
  --full-pipeline --resume \
  >> logs/full_pipeline.log 2>&1 &
```

重试不完整物体：

```bash
nohup python -m tools.grasping.benchmark_catalog \
  --full-pipeline --resume --retry-incomplete \
  >> logs/full_pipeline.log 2>&1 &
```

`--resume` 要求报告 schema 和影响结果的参数完全一致。自动并行数不写入结果语义，因此可以
在不同服务器资源下恢复。

## 跨机器复制结果

复制整个目录最简单：

```bash
rsync -av user@server:/path/to/dex_hand_project/configs/grasps/ configs/grasps/
```

至少需要：

```text
configs/grasps/dex_hand/full_pipeline_benchmark.json
configs/grasps/dex_hand/dexevolve/
```

建议同时保留 `graspqp_seeds/` 便于排查。服务器绝对 mesh 路径会按路径中的 `assets/...`
部分重定位到当前项目。

## 常见错误

### `No module named 'graspqp'`

```bash
git submodule update --init --recursive
python -m pip install -e "deps/graspqp/graspqp[lite]" --no-build-isolation
```

### `Make sure to install proxsuite`

确认 GraspQP lite 依赖安装完整：

```bash
python -m pip install proxsuite
python -m tools.grasping.probe_graspqp_adapter --points-per-geom 40
```

### 恢复时报参数不一致

使用与首次运行相同的 `--full-pipeline`，不要更改搜索、进化或验证参数。若确实需要新参数，
使用新的 `--output`，不要覆盖旧报告。
