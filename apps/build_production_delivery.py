"""Build or verify a fresh delivery candidate; never run simulation."""

import argparse
import json
from pathlib import Path

from source.data.production_delivery import build_delivery, verify_archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.verify is None):
        parser.error('Choose exactly one of --output or --verify')
    result = verify_archive(args.verify) if args.verify else build_delivery(Path.cwd(), args.output)
    print(json.dumps(result, indent=2))
