"""Create an explicitly repeated sample dataset from existing frozen trajectories."""

import argparse
import json
from pathlib import Path

from apps.five_task_production import load_config
from source.data.dataset_expansion import inventory_sources, build_expanded_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/five_task_speed.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--hours', type=float, default=1000)
    parser.add_argument('--minimum-assets', type=int, default=100)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.plan:
        result = inventory_sources(config)
        result = {k:v for k,v in result.items() if k != 'sources'} | {'source_count': len(result['sources'])}
    else:
        if args.output is None:
            parser.error('--output is required for expansion')
        result = build_expanded_dataset(config, args.output, hours=args.hours, minimum_assets=args.minimum_assets)
    print(json.dumps(result, indent=2))
