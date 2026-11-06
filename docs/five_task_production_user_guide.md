# 五任务数据程序使用说明

2026-11-06实际F盘的有限存储测试已验证：双进程登记同一内容只能保留1条唯一记录，重复副本进入隔离；写完文件但尚未rename、rename完成但尚未登记SQL这两个进程退出位置均可恢复，且独立文件审计通过。测试仅复用开发样本，不增加正式小时；尚未模拟断电或WSL重启。开发目录空间扫描遇到原子rename会进行有界重扫，权限或持续不稳定仍停止，不会按0字节放行。

2026-11-06在实际F盘完成单条存储事务验证：普通文件内容复制、hash、fsync、原子rename、SQLite登记及完整解码均通过。DrvFS可能不支持复制Linux权限和时间属性，当前打包器因此仅复制普通文件内容，拒绝symlink，仍保留全部内容校验。若 `/mnt/f` 未挂载，请在本机终端执行 `sudo mount -t drvfs F: /mnt/f` 后重新运行preflight；这只是临时挂载，未配置重启自动挂载。不要发送sudo密码，也不要在root fallback目录下启动生产。单条验证不代表F盘长时恢复或双worker并发已验收。

`status` 的 `runtime_statistics.remaining_time_projection` 提供条件工期预测和不可估原因。它使用实际会话墙钟校准 worker 时间，不按 worker 数直接乘吞吐。每个仍有缺口的任务至少有三个完整计时尝试且所需结果类型已有产出后，才估算小时配额时间。资产覆盖不足时，整体 `wall_clock_eta_seconds` 仍为空；小时部分可单独查看 `hours_only_wall_seconds_at_observed_mix`。进程丢失、历史计时缺口、并行度变更或索引重建后的未知历史均可使估算不可用。数值只是当前任务/结果分布下的条件点估计，不是完成日期承诺。

交付候选构建/校验入口为 `apps/build_production_delivery.py`：`--output <全新outputs目录>`构建，`--verify <zip>`逐文件校验，不启动仿真。归档含代码、配置、模板/注册表及所选资产输入，优先解压到新目录，禁止覆盖原项目。`RUNTIME_ENVIRONMENT.json`记录实际测试版本；迁移环境应先复现这些版本，再运行只读plan/preflight。候选包校验通过不等于生产准入或1000小时数据通过。

2026-11-05集成验证记录见 `docs/progress/2026-11-05_p44_integration.md`：五任务短程新执行均成功，完整文件复核通过；尚未完成小时耐久或正式生产验收。源码合同已补齐实际apps静态传递导入，当前覆盖9个执行依赖脚本；旧合同不能直接用修改后的源码续跑。

当前版本提供有界开发测试，并已实现持续生产的`run/resume`入口。持续入口尚未完成F盘上的端到端及小时级耐久验证，不能据此认定正式生产已就绪。以下先介绍检查配置、短程运行和核对数据。

## 环境与配置

打包阶段会将本次完整视频解码结果绑定到准确路径、帧数和文件SHA-256，仅在同一次调用内复用，避免写metadata前后对同一未变视频重复解码。解码前后和复用时都核对文件字节；变化就拒绝。不会读取JSON中的“已通过”标记代替解码。登记、恢复、定期审计和最终验收仍独立完整解码，不跨进程或跨rename复用结果。

分阶段耗时使用 `production-wall-profile-v1`。每次尝试的记录保存在 `runtime/profiles/`；worker结果保存输入校验、在线执行和独立回放/Strict；work中的result保存环境创建/reset、控制步、状态触觉采集、渲染、模型与数组写入；视频线程另保存管道写入、提交等待、收尾及解码时间。`report.json`的 `timing_diagnostics`按这些层级分别汇总。

这些是墙钟跨度。在线执行包含环境与渲染，worker包含在线执行，异步管道写入与渲染可能重叠，不能跨层直接相加。旧数据缺少计时就返回不可用，不从文件大小虚构耗时。`pilot`现在也可显式传入 `--wall-budget-seconds`，仍受1–5次attempt上限约束。

调度默认在任务缺口、并发负载和未覆盖资产之后，参考模板的实测有效物理秒/worker秒。至少 `efficiency_minimum_attempts=3` 次且所有已领取尝试都有计时才采用；重复和异常的耗时计入分母，不产生有效分子。成功/失败分别配额时，只计仍有缺口的结果类型。每 `efficiency_exploration_interval=10` 次任务规划保留一个探索槽，按较少采样的资产/模板排序，避免永久只用最快模板。未覆盖资产仍优先。

效率统计增量保存，不在每次调度时扫描所有历史attempt。选择依据写入 `SCHEDULER_SELECTED` 事件，并出现在dry-run结果。缺失计时或文件重建造成历史未知时停用效率分数。它是调度提示，不改变任何配额、任务成功或数据验收标准。

首次启动会在数据目录同级写入 `.<数据目录名>.bootstrap.json`，保存目录、索引位置、配置、seed和执行合同；然后原子写入数据目录内的配置/合同，建立队列并完成初始入队，最后写 `bootstrap_complete.json`。完成标记出现前不会启动worker。请连同数据和控制目录保留该启动意图文件。

若初始化被打断，`resume`或`serve`可依据同一启动意图补齐缺失步骤；配置/seed不能改变，初始任务不会重复添加。允许修复本程序已证明归属的空SQLite或缺表初始化，不会将任意已有目录或其他SQLite当成新队列。完成标记已存在却丢失配置、或未完成初始化却已有执行/登记数据时会拒绝自动重置。旧版本没有启动意图的历史目录继续保留审计，不会被新版本悄悄接管。

`status`及每次运行summary现在包含 `runtime_statistics`，按任务和独立资产hash汇总已登记物理时长、实际计时的尝试数、平均耗时、重复及异常消耗。`attempt_metrics`以job/attempt为唯一键，写入后不可改；历史或强制退出丢失的计时明确列为 `unmeasured_attempts`，不补成零。

`remaining_worker_seconds_at_observed_mix`只按已观察到的成功/失败组合估算剩余累计worker秒；存在未计时尝试、没有有效产出或缺少必需结果类型时返回null。成功/失败分开配额分别计算，采用当前产出组合下较大的需求。多个worker时间会重叠，该字段不能直接换算为整机工期，也不能除以worker数；`wall_clock_eta_seconds`在缺少可信整体吞吐预测时保持null。资产覆盖和长尾变化仍需单独评估。这些是SQL运行监控，不替代最终文件验收。

扰动范围示例见 `configs/five_task_production_perturbation_example.json`。可选 `perturbation_contract` 使用 `scene-perturbation-ranges-v1`，显式列出五任务已有场景/目标字段的 `[下界, 上界]`。省略时仍为原来的位置±2mm、角度±0.5度，采样顺序与旧默认相同。下界等于上界表示固定扰动。

范围仅允许已有字段，不能加入质量、摩擦、控制或成功阈值；绝对硬上限仍为位置1cm、角度2度。模板的 `perturbation_fields` 白名单继续生效，禁用字段保持零，不会因配置范围非零而自动开启。改变范围前应做有限canary验证，不保证现有模板在更大范围成功。

显式配置的范围会进入不可变job payload；worker执行前按sampling seed、候选序号和模板字段重新计算条件。与payload不符直接拒绝。已启动运行的配置与源码仍冻结，不能修改范围后当作原运行恢复。

正式运行与 canary 启动时会核对一条已有登记记录，随后按 `integrity_audit_interval_attempts`（默认10次尝试）轮巡核对一条完整 episode。检查包括完整视频解码、数组/hash、不可变任务 payload、结果和物理时长与 SQLite 的一致性。轮巡游标及失败状态持久保存；每轮固定当时最大记录序号，新增记录放到下一轮，避免持续新增使老记录永不复查。发现错误返回 `INTEGRITY_AUDIT_FAILED`、CLI退出码2，并停止领取任务。错误记录保存在 `runtime/integrity/`；审计本身执行异常保存在 `runtime/failures/`。

这是有界抽查，不是全库即时保证，不检查正在写入的事务目录。完整库、孤立目录与配额仍由 `verify/report` 和最终自动验收检查。大数据集一轮耗时会增长，应根据实际审计成本调整间隔；不能把一次抽查通过当作全库验收。旧冻结运行不能直接修改配置后续跑，恢复仍要求原合同一致。

执行失败按 `production-failure-policy-v1` 分类，保存在 `worker_result.json`、队列事件和运行 summary 中。输入合同、回放/Strict/打包校验错误不自动重试；超时或编码管道断开最多尝试2次；明确的临时 I/O 错误最多3次；未知执行错误隔离。内存不足、磁盘满、磁盘配额耗尽和运行保护触发会停止当前会话，保留剩余重试机会。任何类别都不能超过 job 原有的 `maximum_retries`。这些次数包含第一次执行，重启不会重置。

任务执行完整但任务判据失败的 episode 仍按原结果记录，是否贡献配额由成功/失败配置决定，不作为程序异常删除或重标。SIGTERM/会话预算耗尽作为会话中断处理，仍受原 job 总次数上限约束。

程序检测到队列配额完成后，会自动执行完整的文件验收，并把 JSON/Markdown 报告写入数据根目录旁的 `<数据目录名>_acceptance_reports/final_<时间戳>/`，避免将报告本身计入数据容量。只有文件验收通过才返回 `QUOTA_COMPLETE_FILE_VERIFIED`；队列配额完成而文件不足或不合法返回 `QUOTA_COMPLETE_FILE_REJECTED`；报告执行失败返回 `FINAL_REPORT_FAILED`。后两种情况 CLI 退出码为 2，不能作为生产成功。报告异常还会保存在 `runtime/failures/`。`quota_complete` 仍是队列统计，最终数据是否通过以 `final_file_acceptance` 和完整报告为准。

若 SQLite 索引损坏，可先从完整 episode 文件生成独立候选索引：

```sh
PYTHONPATH=. python apps/five_task_production.py rebuild-index --config configs/five_task_production.json --output outputs/index_recovery_candidate
```

输出目录必须不存在且位于数据目录之外。该手工命令逐条校验文件、视频、payload、lineage/content，再对候选数据库独立复核；不会替换原索引。报告列出无法恢复历史重试次数的项目和未完成 work 目录。手工候选没有自动采用资格，不要通过修改配置直接接入生产。旧数据库、失败现场和原数据均应保留。

正式运行或恢复入口默认启用 `auto_rebuild_inventory=true`。在配置、执行合同和独占协调锁检查通过后，若SQLite报告物理损坏，程序检查是否仍有该数据目录的worker/编码器进程存活。存活或无法确认时拒绝恢复。可恢复时，将新索引建在数据目录同级的 `<数据目录名>_index_recovery/`，完整独立验收后原子写入 `<原数据库名>.active.json`。原数据库及WAL现场不覆盖，status、dry-run、verify、report均跟随活动指针。

无法确定历史次数的未完成任务按预算已耗尽隔离，不重新授予重试；候选序号从已保存payload恢复。恢复后的尝试计数包含保守预算记账，不是实测执行次数，历史计时及ETA标为未知。payload缺失/冲突、错误资产或任务身份、重复/损坏文件、未完成事务等导致独立审计失败时，不激活新索引。无任何文件证据时拒绝重置为空队列。有效但不属于本程序的SQLite不会自动重建。

这里实现的是启动/`resume`时的自动索引恢复；运行中发生数据库异常仍会停止当前会话，需要通过恢复入口重新检查。它不能修复已损坏的episode文件，也不替代WSL重启与长时故障验证。开发canary的总attempt预算在历史重建后无法证明时，驱动会停止，不能借恢复重置开发测试上限。

在项目根目录执行，使用已安装MuJoCo的Python：

```bash
export PYTHONPATH=.
export MUJOCO_GL=egl
```

下列命令中的`python`应为`/home/zmy/miniconda3/envs/mujoco/bin/python`，或已激活对应conda环境。

`configs/five_task_production.json`使用当前五任务基础模板。`configs/five_task_production_catalogue.json`使用443条历史候选、101个独立候选资产；该目录尚不代表当前100物体成功覆盖。

配置保留五任务各200小时、至少100资产、成功Episode计时、最多2 workers、20%磁盘保留和每worker180秒上限。失败数据保留，但默认不计入成功时长。不要把物体数量、attempt数量或视频文件数当成有效小时数。

若需要成功/失败分别达标，可参考`configs/five_task_production_outcome_example.json`：每任务180小时成功、20小时失败，共900+100小时；必须显式开启失败计时。`task_outcome_hours`各任务之和应等于其`task_hours`。超出某一类的时长不能抵扣另一类缺口；原始成功/失败时长仍分别保留，配额贡献单独计算。未配置该字段时沿用原成功-only或成功+失败计时。

`coverage_requires_success`默认true，失败记录可以按配置计时，但不能因此满足100个成功资产覆盖。配额不保证当前物理条件一定能产生对应比例；不能修改标签或故意伪造失败来凑配额。`maximum_no_progress_attempts`默认1000，连续尝试未减少任何时长/资产缺口时以`QUOTA_STALLED`退出，计数跨重启保存。检查缺口后再决定下一步，不将“生成了更多文件”当成进展。

## 只读检查

```bash
python apps/five_task_production.py preflight
python apps/five_task_production.py plan
python apps/five_task_production.py plan --config configs/five_task_production_catalogue.json
```

`preflight`检查F盘身份、容量、依赖与编码器。`/mnt/f`落回WSL根分区时不允许正式生产；目录存在不代表已挂载。程序不修改系统挂载配置。

## 有界开发测试

为每次新测试使用独立输出目录：

```bash
python apps/five_task_production.py pilot --pilot-root outputs/my_five_task_pilot --max-episodes 1 --sampling-seed 32000
```

开发pilot每次只允许1至5个attempt，根目录必须位于项目`outputs/`。正式持续入口使用同一个调度核心，不要求用户逐批凑1000小时。

恢复已有任务，不自动增加候选：

```bash
python apps/five_task_production.py pilot --pilot-root outputs/my_five_task_pilot --max-episodes 1 --sampling-seed 32000 --resume
```

需要在同一有界预算内追加候选时，显式添加`--append-candidates`。恢复必须使用原配置；配置不一致时拒绝执行。候选编号保存在数据库中，不从零重置。

## 检查进度和文件

需要先检查历史候选在当前版本的兼容性时，可显式选择少量资产：

```bash
python apps/probe_production_compatibility.py --output outputs/my_asset_compatibility --case pick_place=egad:C1 --wall-budget-seconds 240
```

该入口一次最多五个case、单worker、零扰动，不修改原模板库。检查输出中的`compatibility_matrix.json`和`file_audit.json`；单条件通过后还需要扰动验证。不要用同一资产的多条轨迹代替独立资产数量。

```bash
python apps/five_task_production.py status --pilot-root outputs/my_five_task_pilot
python apps/five_task_production.py verify --pilot-root outputs/my_five_task_pilot
python apps/five_task_production.py report --pilot-root outputs/my_five_task_pilot --output outputs/my_five_task_file_report
```

`status`读取队列与已登记时长；`verify`重新读取Episode文件、检查哈希、动作因果、触觉、视频完整解码和时间戳，再与数据库比较。它会消耗解码时间，不能用数据库状态替代。

`report`同样重新校验文件，另外从文件独立计算五任务及成功/失败配额、独立资产、正式数据验收和实际存储。`--output`必须是数据根之外的新目录，生成`report.json`与`REPORT.md`。缺少某任务样本时，完整任务加权容量预测返回null；小样本平均外推不是正式容量批准。

运行记录在`runs/`，工作过程和日志在`work/`，登记数据在`episodes/`。重复产物进入`duplicates/`，不增加有效时长；不要未经检查删除目录或手工修改SQLite计数。中断后先恢复已有任务；不要同时启动两个协调器使用同一个根目录。

## 当前数据合同

每帧保存39D状态、13D实际关节位置动作、712D本次传感器触觉、完整对象位姿及时间戳；RGB为1920×1080、20Hz。新记录区分操作body、静态fixture和目标site，并保存无损压缩的实际编译模型。动作对应执行前观测，保留笛卡尔IK指令作为辅助字段。

最新在线合同为v4，另保留float64的MuJoCo完整执行前积分状态和完整实际ctrl，用于严格复算。常规学习字段不降维、不改变频率。旧v3数据仍可读取，但不能事后补造缺失精度；新v4回放只恢复首状态，然后逐步执行原始控制，不逐帧恢复参考位姿。回放辅助字段不增加内容去重的独立样本数量。

正式数据准入尚未完成。`development_verified_not_formal_production`只能表示开发流程验证通过。历史模板成功、格式检查通过或内部任务结果均不能自动替代完整正式验收。

## 正式启动前待完成

新增 `serve` 监督入口，仅在用户明确执行该命令时启动。它保留正式F盘、配置、源码和配额检查，并自动选择首次run或已有队列resume：

```bash
python apps/five_task_production.py serve --config configs/five_task_production_catalogue.json --output outputs/my_supervisor_001 --maximum-restarts 3 --wall-budget-seconds 3600
```

此命令仍属正式路径，当前F盘/耐久准入未完成时不要运行。开发没有执行上述生产命令。`--output`必须是数据目录之外的新控制目录；日志是`session_000.log`等及`supervisor.json`。后台运行可使用nohup，但外层日志应写到该新目录之外，避免先创建目录导致fresh-output检查拒绝。WSL实例关闭仍会结束Linux进程，重新启动后用相同配置和seed、新监督日志目录调用serve，不能把nohup当作跨WSL重启服务。

自动重启只接受明确的数据库损坏退出码75、SIGKILL/SIGSEGV/SIGABRT/SIGBUS异常退出。默认最多重启3次（最多10），不是无限重试。重启前等待旧dataset worker/FFmpeg退出，重新检查F卷/资源与配置/源码；仍有残留进程或合同变化时停止。新协调器使用独立PID，从原队列恢复，不重置attempt或候选编号。

人工停止监督进程（SIGINT/SIGTERM）会终止当前协调器且不重启；不要只杀子协调器来停止整个服务。普通退出、磁盘/资源保护、配额停滞、文件验收拒绝、一般配置错误都不自动重启。退出码0表示正常停止，不等于1000小时数据通过；仍看最终文件报告和`formal_dataset_accepted`。

百物体当前兼容验证、小时级故障测试、工作副本保留规则和含编译模型的新容量预算仍待完成。F盘挂载必须通过身份检查。

开发耐久使用独立`canary`入口，不能指向正式F盘根：

```bash
python apps/five_task_production.py canary --pilot-root outputs/my_bounded_canary --max-episodes 40 --sampling-seed 41000 --wall-budget-seconds 3600
```

单次最多40个attempt/一小时，可加`--resume`继续原合同，但仍受该次预算限制。它有本地测试卷与内存保护、源代码冻结和状态快照；产物仍为开发准入。不要在运行期间修改源代码或模板。带一次受控SIGTERM并自动恢复的外层测试驱动为`apps/run_production_canary.py`，整个测试另限制总attempt，不能用它绕过正式run的F卷检查。

持续入口格式如下，当前未在真实F盘上完成端到端验证；本轮开发没有执行这些生产命令：

```bash
python apps/five_task_production.py run --config configs/five_task_production_catalogue.json --sampling-seed 32000 --wall-budget-seconds 3600
python apps/five_task_production.py resume --config configs/five_task_production_catalogue.json --sampling-seed 32000 --wall-budget-seconds 3600
```

`dry-run`现在预览真实调度规则下的下一条候选，显示模板、资产、候选序号、seed及扰动条件。它只在临时 SQLite 副本上调度，不登记真实任务、不领取租约、不启动仿真；输出中的 `ENQUEUED` 指临时副本，`jobs_persisted=0`。预览不替代 F 盘与生产准入检查，也不保证候选成功。

`run`要求全新数据根目录和队列，`resume`要求已有队列、原配置、相同采样seed及模板/注册表文件哈希。省略会话墙钟预算会持续运行，直到五任务时长与资产配额完成或安全停止；开发阶段不要省略预算。`Ctrl-C`/SIGTERM停止领取并终止本次子进程，已有事务保留，后续恢复按队列处理。

每次领取前和worker运行期间检查F卷身份及安全空间；不满足时停止。所有候选被隔离时退出并保留配额缺口，不将失败任务配额挪给其他任务。单次汇总明确记录退出原因，持久队列和运行报告保留完整记录。正式Episode必须带模型、回放与对象身份信息；成功Lift另需原FINAL Strict通过证明。

任务续租与运行保护覆盖打包、解码和登记全过程。源码或执行包版本变化时停止，恢复前必须与冻结合同一致；不要在长期运行期间编辑同一份执行代码。停止时清理worker所在进程组，防止残留编码器继续运行。会话预算到达后停止领取与worker；已开始的单次文件检查可能需要在其有界超时内返回，保留的事务供下次恢复。

运行合同也包含机器人、手、场景、纹理等共享文件；改变这些文件后不能直接续跑旧合同。内存默认保留1 GiB，F盘在20%余量之外另留6GB工作空间。`status`中的`worker_snapshots`来自原子运行快照；关注更新时间，陈旧快照不证明进程仍存活，也不增加有效时长。
# 去重前后时长的查看

`status` 中的 `runtime_statistics` 按任务和资产区分候选时长与唯一登记时长。`pre_dedup_candidate_seconds` 在记录完整时给出候选总秒数；存在历史缺口时为 null，并报告 `unknown_candidate_attempts` 与已知部分。`post_dedup_registered_seconds` 为唯一登记成功与失败 Episode 时长之和，不等同于配置允许累计的配额。

恢复可能再次处理同一份完成输出，因此候选总秒数不能当作新执行仿真小时数。正式配额和最终验收仍按去重后合格文件计算。只读 status 不升级旧索引，旧记录不会自动补齐。
