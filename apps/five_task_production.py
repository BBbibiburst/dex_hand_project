"""Production planning, queue inspection and explicitly bounded development pilots."""

import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

from source.data.production_queue import TASKS
from source.data.production_quotas import validate_outcome_hours


def load_config(path):
    config = json.loads(path.read_text())
    if config.get('storage_policy') == 'configured-volume-v1' and os.environ.get('DEX_DATASET_ROOT'):
        config['dataset_root'] = str(Path(os.environ['DEX_DATASET_ROOT']).expanduser().resolve())
    if config.get("schema") != "five-task-production-config-v1":
        raise ValueError("Unsupported production configuration schema")
    if set(config["task_hours"]) != set(TASKS) or any(
            not math.isfinite(v) or v <= 0 for v in config["task_hours"].values()):
        raise ValueError("Five positive finite task quotas are required")
    if sum(config["task_hours"].values()) < 1000:
        raise ValueError("Production target must cover at least 1000 physical hours")
    if config["minimum_independent_assets"] < 100:
        raise ValueError("Require at least 100 assets across the dataset")
    from source.data.production_parallel import validate_worker_config

    validate_worker_config(config)
    if config.get('storage_policy') not in (None, 'configured-volume-v1'):
        raise ValueError('Unsupported storage policy')
    if not isinstance(config["count_failed_episodes"], bool):
        raise ValueError("Failure accounting must be explicit boolean")
    if not isinstance(config.get('coverage_requires_success', True), bool):
        raise ValueError('coverage_requires_success must be boolean')
    validate_outcome_hours(config['task_hours'], config.get('task_outcome_hours'), config['count_failed_episodes'])
    from source.data.production_conditions import sampling_contract

    sampling_contract(config.get('perturbation_contract'))
    limit = config.get('maximum_no_progress_attempts', 1000)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('maximum_no_progress_attempts must be a positive integer')
    audit_interval = config.get('integrity_audit_interval_attempts', 10)
    if isinstance(audit_interval, bool) or not isinstance(audit_interval, int) or audit_interval < 1:
        raise ValueError('integrity_audit_interval_attempts must be a positive integer')
    if not isinstance(config.get('auto_rebuild_inventory', True), bool):
        raise ValueError('auto_rebuild_inventory must be boolean')
    for key in ('efficiency_minimum_attempts', 'efficiency_exploration_interval'):
        value = config.get(key, 3 if key == 'efficiency_minimum_attempts' else 10)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
            raise ValueError(f'{key} must be an integer between 1 and 1000')
    if not math.isfinite(config["episode_wall_timeout_seconds"]) or config["episode_wall_timeout_seconds"] < 1:
        raise ValueError("Episode execution requires a positive finite wall timeout")
    if isinstance(config['maximum_retries'], bool) or not isinstance(config["maximum_retries"], int) or config["maximum_retries"] < 1:
        raise ValueError("Retry budget must be a positive integer")
    if not .2 <= config["disk_reserve_fraction"] < 1:
        raise ValueError("Require at least 20 percent disk reserve")
    expected = {"physics_hz": 1000, "control_hz": 20, "state_dimension": 39,
                "action_dimension": 13, "tactile_dimension": 712}
    from source.data.production_capture import capture_from_config

    if capture_from_config(config) is None:
        expected.update(rgb_width=1920, rgb_height=1080, rgb_fps=20)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Configuration violates the frozen multimodal contract")
    return config


def inspect(config):
    from source.data.production_statistics import ReadOnlyQueue, production_statistics
    from source.data.production_index import active_database

    database = active_database(config['dataset_root'], config['queue_database'])
    if not database.exists():
        return {"queue_exists": False, "production_ready": False,
                "reason": "No production jobs registered", "target_hours": config["task_hours"]}
    status = ReadOnlyQueue(database).quota_status(
        config["task_hours"], min_assets=config["minimum_independent_assets"],
        count_failures=config["count_failed_episodes"], outcome_hours=config.get('task_outcome_hours'),
        coverage_success_only=config.get('coverage_requires_success', True))
    snapshots = []
    snapshot_error = None
    try:
        for path in sorted((Path(config['dataset_root']) / 'runtime').glob('worker_*.json')):
            snapshots.append(json.loads(path.read_text()))
    except OSError as error:
        snapshot_error = str(error)
    return {"queue_exists": True, "production_ready": False, 'worker_snapshots': snapshots,
            'worker_snapshot_error': snapshot_error,
            'runtime_statistics': production_statistics(config), 'active_database': str(database), **status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "plan", "dry-run", "status", "report", "inventory", "rebuild-index", "verify", "pilot", "canary", "run", "resume", "serve"))
    parser.add_argument("--config", type=Path, default=Path("configs/five_task_production.json"))
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--sampling-seed", type=int, default=32000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--append-candidates', action='store_true', help='Permit bounded new candidates during pilot resume')
    parser.add_argument('--wall-budget-seconds', type=float, help='Optional explicit session wall-clock budget for run/resume')
    parser.add_argument('--output', type=Path, help='Fresh output directory (report, rebuild-index or serve)')
    parser.add_argument('--maximum-restarts', type=int, help='serve only; default 3, maximum 10')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output is not None and args.command not in ('report', 'rebuild-index', 'serve'):
        raise ValueError('--output applies only to report, rebuild-index or serve')
    if args.maximum_restarts is not None and args.command != 'serve':
        raise ValueError('--maximum-restarts applies only to serve')
    if args.command in ('run', 'resume', 'serve') and (args.max_episodes is not None or args.resume or args.append_candidates):
        raise ValueError('Use --wall-budget-seconds for formal sessions; pilot-only flags are not accepted')
    if args.command not in ('run', 'resume', 'canary', 'serve', 'pilot') and args.wall_budget_seconds is not None:
        raise ValueError('--wall-budget-seconds applies only to run/resume/canary/serve/pilot')
    if args.command in ('run', 'resume', 'serve') and args.pilot_root is not None:
        raise ValueError('Formal run/resume cannot use --pilot-root')
    if args.pilot_root is not None and args.command not in ('pilot', 'canary'):
        config = {**config, "dataset_root": str(args.pilot_root),
                  "queue_database": str(args.pilot_root / "inventory.sqlite")}
    if args.command == "preflight":
        from source.data.production_preflight import preflight

        result = preflight(config)
    elif args.command == 'serve':
        from source.data.production_supervisor import supervise

        if args.output is None:
            raise ValueError('serve requires a fresh --output control directory')
        result = supervise(config, args.config, args.sampling_seed, args.output,
                           maximum_restarts=args.maximum_restarts if args.maximum_restarts is not None else 3,
                           wall_budget=args.wall_budget_seconds)
    elif args.command in ('run', 'resume'):
        from source.data.production_coordinator import production

        try:
            result = production(config, args.sampling_seed, resume=args.command == 'resume',
                                wall_budget=args.wall_budget_seconds)
        except sqlite3.DatabaseError as error:
            from source.data.production_index import is_database_corruption

            if not is_database_corruption(error):
                raise
            print(json.dumps({'termination': 'DATABASE_CORRUPTION', 'error': str(error)}), flush=True)
            raise SystemExit(75) from error
    elif args.command == 'canary':
        from source.data.production_coordinator import canary

        if args.pilot_root is None or args.append_candidates:
            raise ValueError('Canary needs an explicit --pilot-root and automatically appends within its finite budget')
        result = canary(config, args.pilot_root, args.max_episodes if args.max_episodes is not None else 40,
                        args.sampling_seed, resume=args.resume,
                        wall_budget=args.wall_budget_seconds if args.wall_budget_seconds is not None else 3600)
    elif args.command == "pilot":
        from source.data.production_coordinator import pilot

        if args.pilot_root is None:
            raise ValueError("pilot requires an explicit --pilot-root in project outputs")
        result = pilot(config, args.pilot_root, args.max_episodes if args.max_episodes is not None else 1, args.sampling_seed, args.resume,
                       args.append_candidates, wall_budget=args.wall_budget_seconds)
    elif args.command == 'rebuild-index':
        from source.data.production_rebuild import rebuild_inventory

        if args.output is None:
            raise ValueError('rebuild-index requires a fresh --output outside the dataset')
        result = rebuild_inventory(Path(config['dataset_root']), args.output)
    elif args.command == 'report':
        from source.data.production_report import build_report, save_report

        if args.output is not None and args.output.exists():
            raise FileExistsError(args.output)
        result = build_report(config)
        if args.output is not None:
            save_report(result, args.output)
    elif args.command in ("verify", "inventory"):
        from source.data.production_inventory import audit_inventory
        from source.data.production_index import active_database

        result = audit_inventory(Path(config["dataset_root"]), active_database(config['dataset_root'], config['queue_database']))
    elif args.command == 'dry-run':
        from source.data.production_preview import preview
        from source.data.production_templates import load_templates

        templates = load_templates(Path(config.get('template_library', 'configs/five_task_templates.json')),
                                   Path(config.get('object_registry', 'outputs/five_task_p31/object_registry.json')))
        result = preview(config, templates, args.sampling_seed)
    elif args.command == "plan":
        from source.data.production_templates import load_templates

        templates = load_templates(Path(config.get("template_library", "configs/five_task_templates.json")),
                                   Path(config.get("object_registry", "outputs/five_task_p31/object_registry.json")))
        result = {"configuration": config, "production_ready": False,
                  "template_count": len(templates), "template_asset_count": len({r["asset_hash"] for r in templates}),
                  "remaining_gate": "Current 100-asset compatibility, endurance and final storage/volume validation"}
    else:
        result = inspect(config)
    print(json.dumps(result, indent=2))
    if result.get('termination') in ('QUOTA_COMPLETE_FILE_REJECTED', 'FINAL_REPORT_FAILED', 'INTEGRITY_AUDIT_FAILED'):
        raise SystemExit(2)
    if result.get('termination', '').startswith('SUPERVISOR_') and result['termination'] not in (
            'SUPERVISOR_STOP_REQUESTED', 'SUPERVISOR_WALL_BUDGET'):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
