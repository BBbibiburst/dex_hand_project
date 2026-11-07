"""Read-only live collection dashboard; safe to start beside an existing run."""

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
import time


LABELS = {'lift': 'Lift', 'pick_place': 'PickPlace', 'stack': 'Stack',
          'push': 'Push', 'nut_assembly': 'NutAssembly'}


def process_alive(owner):
    if not owner:
        return False
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        fields = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and boot == owner['boot'] and fields[19] == owner['start_ticks']
    except (OSError, KeyError):
        return False


def snapshot(config, *, now=None):
    now = time.time() if now is None else now
    from source.data.production_index import active_database
    path = active_database(config['dataset_root'], config['queue_database'])
    if not path.exists():
        return {'available': False, 'reason': '等待采集数据库建立', 'database': str(path)}
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        totals = {r['task']: dict(r) for r in db.execute(
            'SELECT task,count(*) AS episodes,sum(success) AS successes,'
            'sum(CASE WHEN success=1 THEN seconds ELSE 0 END) AS success_seconds,'
            'sum(CASE WHEN success=0 THEN seconds ELSE 0 END) AS failure_seconds '
            'FROM episodes GROUP BY task')}
        states = {r['state']: r['n'] for r in db.execute('SELECT state,count(*) n FROM jobs GROUP BY state')}
        running = {r['task']: r['n'] for r in db.execute(
            "SELECT task,count(*) n FROM jobs WHERE state IN ('RUNNING','VALIDATING') "
            'AND lease_until>? GROUP BY task', (now,))}
        success_only = config.get('coverage_requires_success', True) or not config['count_failed_episodes']
        assets = db.execute('SELECT count(DISTINCT asset_hash) FROM episodes' +
                            (' WHERE success=1' if success_only else '')).fetchone()[0]
        sessions = [dict(r) for r in db.execute(
            "SELECT job_id,state,details,timestamp FROM events WHERE state IN "
            "('RUNTIME_SESSION_STARTED','RUNTIME_SESSION_FINISHED') ORDER BY id")]
        errors = [dict(r) for r in db.execute(
            'SELECT task,error FROM jobs WHERE error IS NOT NULL ORDER BY rowid DESC LIMIT 3')]
    starts = [r for r in sessions if r['state'] == 'RUNTIME_SESSION_STARTED']
    finished = {r['job_id'] for r in sessions if r['state'] == 'RUNTIME_SESSION_FINISHED'}
    live = [r for r in starts if r['job_id'] not in finished and process_alive(json.loads(r['details']).get('owner'))]
    tasks = {}
    for task, hours in config['task_hours'].items():
        row = totals.get(task, {'episodes': 0, 'successes': 0, 'success_seconds': 0., 'failure_seconds': 0.})
        seconds = row['success_seconds'] + (row['failure_seconds'] if config['count_failed_episodes'] else 0.)
        tasks[task] = {**row, 'credited_seconds': seconds, 'target_seconds': hours * 3600,
                       'remaining_seconds': max(0., hours * 3600 - seconds), 'running': running.get(task, 0)}
    return {'available': True, 'database': str(path), 'dataset_root': config['dataset_root'],
            'timestamp': now, 'tasks': tasks, 'states': states, 'coordinator_running': bool(live),
            'configured_workers': json.loads(live[-1]['details'])['workers'] if live else None,
            'elapsed_seconds_including_downtime': max(0., now - starts[0]['timestamp']) if starts else None,
            'independent_assets': assets, 'target_assets': config['minimum_independent_assets'],
            'count_failed_episodes': config['count_failed_episodes'], 'recent_errors': errors,
            'credited_seconds': sum(r['credited_seconds'] for r in tasks.values()),
            'target_seconds': sum(r['target_seconds'] for r in tasks.values())}


def projection(data, history):
    """Use observed aggregate wall time, never worker count as a multiplier."""
    seconds = {task: r['credited_seconds'] for task, r in data['tasks'].items()}
    now = data['timestamp']
    history.append((now, seconds))
    while len(history) > 2 and history[1][0] < now - 300:
        history.popleft()
    elapsed = data['elapsed_seconds_including_downtime']
    rates = {t: None for t in seconds}
    basis = '等待首条有效数据及至少30秒计时'
    if now - history[0][0] >= 30:
        period = now - history[0][0]
        # A rebuilt/changed index must not produce negative throughput or a false ETA.
        if all(seconds[t] >= history[0][1][t] for t in seconds):
            rates = {t: (v - history[0][1][t]) / period for t, v in seconds.items()}
            basis = f'最近{period:.0f}秒实测'
    elif elapsed is not None and elapsed >= 30 and data['credited_seconds'] > 0:
        rates = {t: v / elapsed for t, v in seconds.items()}
        basis = '自首次采集以来，包含停顿'
    measured = all(v is not None for v in rates.values())
    rate = sum(rates.values()) if measured else None
    remaining = sum(r['remaining_seconds'] for r in data['tasks'].values())
    quota_etas = [0. if r['remaining_seconds'] == 0 else r['remaining_seconds'] / rates[t]
                 if rates[t] is not None and rates[t] > 0 else None for t, r in data['tasks'].items()]
    active = data['coordinator_running']
    return {'basis': basis, 'rate': rate,
            'total_eta_seconds': (0. if remaining == 0 else remaining / rate if rate and active else None),
            'task_rates': rates,
            'task_quota_eta_seconds': max(quota_etas) if active and all(x is not None for x in quota_etas) else None}


def duration(seconds):
    if seconds is None:
        return '暂无估计'
    if seconds >= 86400:
        return f'{seconds / 86400:.1f}天'
    return f'{seconds / 3600:.1f}小时' if seconds >= 3600 else f'{seconds / 60:.1f}分钟'


def render(data, estimate):
    if not data['available']:
        return data['reason'] + '\n数据库：' + data['database']
    completed = data['credited_seconds'] / 3600
    target = data['target_seconds'] / 3600
    rate = estimate['rate']
    lines = [f"五任务采集看板 | {datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
        f"采集进程：{'运行中' if data['coordinator_running'] else '未检测到运行中的采集进程'}  "
        f"执行/验证中：{sum(r['running'] for r in data['tasks'].values())}  "
        f"并发上限：{data['configured_workers'] if data['configured_workers'] is not None else '--'}",
        f'有效时长：{completed:.4f} / {target:.0f} 小时  ({completed / target * 100:.4f}%)',
        '采集效率：' + (f'{rate:.3f}倍实时 = 每墙钟小时采集{rate:.3f}小时' if rate is not None else '等待计时') +
        f"  [{estimate['basis']}]",
        f"总时长目标预计剩余：{duration(estimate['total_eta_seconds'])}；"
        f"同时满足各任务配额：{duration(estimate['task_quota_eta_seconds'])}",
        f"物体覆盖：{data['independent_assets']} / {data['target_assets']}（全任务合计）",
        '', '任务          已采集小时 / 目标小时   进度       条数   成功/失败   执行中  效率(倍实时)']
    for task, row in data['tasks'].items():
        lines.append(f"{LABELS.get(task, task):<13} {row['credited_seconds']/3600:9.4f} / "
            f"{row['target_seconds']/3600:7.1f}   {100*row['credited_seconds']/row['target_seconds']:7.3f}%  "
            f"{row['episodes']:5d}   {row['successes']}/{row['episodes']-row['successes']}     {row['running']}      "
            + (f"{estimate['task_rates'][task]:.3f}" if estimate['task_rates'][task] is not None else '--'))
    lines += ['', '队列：' + '  '.join(f'{k}={v}' for k, v in sorted(data['states'].items())),
        '计入时长：已校验并登记的唯一Episode；' + ('成功和合法失败均计入' if data['count_failed_episodes'] else '仅成功计入'),
        'ETA按当前实测速率粗估；任务/物体覆盖、后续难度和停顿可能延长。不是最终全量文件复验。',
        f"数据目录：{data['dataset_root']}"]
    if data['recent_errors']:
        lines += ['最近任务错误（可能已恢复）：'] + [f"  {r['task']}: {r['error'][:160]}" for r in data['recent_errors']]
    lines.append('自动刷新；Ctrl+C只关闭看板，不停止采集。')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/five_task_speed.json'))
    parser.add_argument('--interval', type=float, default=5.)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least 1 second')
    config = json.loads(args.config.read_text())
    if config.get('storage_policy') == 'configured-volume-v1' and os.environ.get('DEX_DATASET_ROOT'):
        config['dataset_root'] = os.environ['DEX_DATASET_ROOT']
    history = deque()
    try:
        while True:
            try:
                data = snapshot(config)
                estimate = projection(data, history) if data['available'] else None
                if sys.stdout.isatty() and not args.once and not args.json:
                    print('\033[2J\033[H', end='')
                print(json.dumps({'data': data, 'estimate': estimate}, ensure_ascii=False) if args.json
                      else render(data, estimate), flush=True)
            except (OSError, sqlite3.Error, ValueError) as error:
                if args.once:
                    raise
                print(f'读取暂不可用，稍后重试：{error}', flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
