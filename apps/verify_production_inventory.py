"""Audit an existing inventory and its files; never generate simulation data."""

import argparse
import json
from pathlib import Path

from source.data.production_inventory import audit_inventory
from source.data.production_index import active_database


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_inventory(args.root, active_database(args.root, args.database))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != 'episodes'}, indent=2))
