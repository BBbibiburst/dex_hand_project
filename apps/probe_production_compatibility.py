"""Bounded current execution checks for explicitly selected historical assets."""

import argparse
import json
from pathlib import Path

from apps.five_task_production import load_config
from source.data.production_coordinator import pilot
from source.data.production_templates import load_templates
from source.data.production_compatibility import compatibility_matrix
from source.data.production_inventory import audit_inventory


def select_cases(rows, cases):
    selected = []
    for case in cases:
        task, identity = case.split('=', 1)
        matches = [r for r in rows if r['task'] == task and r.get('object_id') == identity]
        if not matches:
            raise ValueError(f'No frozen historical template for {case}')
        row = min(matches, key=lambda r: (r.get('validation_status') != 'CURRENT_SINGLE_CONDITION_SUCCESS',
                                          r['template_id']))
        if row['template_id'] in {r['template_id'] for r in selected}:
            raise ValueError('Repeated compatibility case')
        selected.append({**row, 'perturbation_fields': []})
    return selected


def probe(config_file, output, cases, budget):
    project = Path(__file__).resolve().parents[1]
    if not output.resolve().is_relative_to(project / 'outputs') or not 1 <= len(cases) <= 5:
        raise ValueError('Select one to five cases in a new project outputs directory')
    if not 1 <= budget <= 600:
        raise ValueError('Compatibility probe session budget must be 1 to 600 seconds')
    config = load_config(config_file)
    rows = load_templates(Path(config['template_library']), Path(config['object_registry']))
    selected = select_cases(rows, cases)
    output.mkdir(parents=True, exist_ok=False)
    library = output / 'selected_templates.json'
    library.write_text(json.dumps({'schema': 'five-task-template-library-v1', 'templates': selected}, indent=2))
    config = {**config, 'workers': 1, 'maximum_retries': 1, 'template_library': str(library)}
    selection = {'cases': cases, 'zero_perturbation': True, 'maximum_attempts': len(selected),
                 'worker_wall_seconds': config['episode_wall_timeout_seconds'], 'session_wall_seconds': budget,
                 'scope': 'Current compatibility development probe, not formal production or 100-object coverage'}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    result = pilot(config, output / 'dataset', len(selected), 36000, wall_budget=budget)
    from source.data.production_index import active_database

    database = active_database(output / 'dataset', output / 'dataset/inventory.sqlite')
    audit = audit_inventory(output / 'dataset', database)
    (output / 'file_audit.json').write_text(json.dumps(audit, indent=2))
    (output / 'compatibility_matrix.json').write_text(json.dumps(
        compatibility_matrix(database, audit), indent=2))
    (output / 'probe_results.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/five_task_production_catalogue.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True, help='TASK=OBJECT_ID; at most five explicit cases')
    parser.add_argument('--wall-budget-seconds', type=float, default=450)
    args = parser.parse_args()
    print(json.dumps(probe(args.config, args.output, args.case, args.wall_budget_seconds), indent=2))
