"""File-derived acceptance and measured retained storage, with no production run."""

import json
from pathlib import Path
import stat

from source.data.production_acceptance import acceptance_from_audit
from source.data.production_inventory import audit_inventory
from source.data.production_index import active_database
from source.data.production_profile import summarize_profiles


def footprint(directory):
    logical = allocated = 0
    seen = set()
    categories = {}
    for path in directory.rglob('*'):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            continue
        logical += info.st_size
        identity = (info.st_dev, info.st_ino)
        if identity not in seen:
            allocated += info.st_blocks * 512
            seen.add(identity)
        name = path.name
        category = ('rgb' if name.endswith('.mp4') else 'compiled_model' if name in ('model.mjb', 'model.mjb.gz')
                    else 'arrays' if name.endswith('.npz') else 'metadata_logs_other')
        categories[category] = categories.get(category, 0) + info.st_size
    return {'logical_bytes': logical, 'allocated_unique_inode_bytes': allocated, 'categories_logical_bytes': categories}


def build_report(config):
    root = Path(config['dataset_root'])
    audit = audit_inventory(root, active_database(root, config['queue_database']))
    acceptance = acceptance_from_audit(config, audit)
    retained = footprint(root)
    registered = footprint(root / 'episodes')
    hours = acceptance['credited_hours']
    task_bytes = {task: 0 for task in config['task_hours']}
    for row in audit['episodes']:
        task_bytes[row['task']] += footprint(root / row['relative_path'])['logical_bytes']
    weighted = {task: task_bytes[task] / values['credited_hours'] * config['task_hours'][task] / 1e12
                if values['credited_hours'] else None for task, values in acceptance['tasks'].items()}
    complete_tasks = all(value is not None for value in weighted.values())
    storage = {'registered_directory': registered, 'entire_retained_root': retained,
               'credited_physical_hours': hours,
               'projection_TB_per_1000_credited_hours': {
                   'registered': registered['logical_bytes'] / hours * 1000 / 1e12 if hours else None,
                   'retained_root': retained['logical_bytes'] / hours * 1000 / 1e12 if hours else None},
               'task_weighted_projection': {'registered_TB_by_task_target': weighted,
                   'all_tasks_sampled': complete_tasks,
                   'registered_total_TB': sum(weighted.values()) if complete_tasks else None,
                   'retained_total_TB_using_current_root_multiplier': sum(weighted.values()) * retained['logical_bytes'] / registered['logical_bytes']
                   if complete_tasks and registered['logical_bytes'] else None},
               'limitations': 'Measured retained files, not a measured transient peak or throughput; small task/object sample'}
    return {'file_audit': audit, 'acceptance': acceptance, 'storage_measurement': storage,
            'timing_diagnostics': summarize_profiles(root),
            'configuration': config, 'production_started_by_report': False}


def save_report(report, output):
    if output.resolve().is_relative_to(Path(report['configuration']['dataset_root']).resolve()):
        raise ValueError('Report output must stay outside the measured dataset')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    acceptance = report['acceptance']
    lines = ['# 五任务数据文件验收', '',
             f"正式数据验收：{'通过' if acceptance['formal_dataset_accepted'] else '未通过'}。软件交付检查另见开发验收记录。", '',
             '| 任务 | 成功小时 | 失败小时 | 配额贡献小时 | 剩余小时 |', '|---|---:|---:|---:|---:|']
    for task, row in acceptance['tasks'].items():
        lines.append(f"| {task} | {row['successful_hours']:.6f} | {row['failed_hours']:.6f} | {row['credited_hours']:.6f} | {row['remaining_hours']:.6f} |")
    lines += ['', f"独立资产：{acceptance['independent_assets']}；仍缺{acceptance['remaining_assets']}。",
              f"文件完整性：{acceptance['file_integrity_passed']}；全部正式准入：{acceptance['all_credited_files_formally_admitted']}。", '',
              '存储实测及文件错误见report.json。容量外推只基于本样本和保留副本，不是实测峰值或1000小时生产批准。',
              '本报告不启动仿真或生产。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
