"""Non-destructive P30 production preflight for the intended Windows F volume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


EXPECTED_MINIMUM_BYTES = 19_000_000_000_000


def check(path: Path) -> dict:
    resolved = path.resolve()
    stat = os.statvfs(resolved)
    total = stat.f_blocks * stat.f_frsize
    available = stat.f_bavail * stat.f_frsize
    payload = {
        "path": str(path),
        "resolved": str(resolved),
        "total_bytes": total,
        "available_bytes": available,
        "safe_new_bytes_after_20_percent_reserve": available - int(0.2 * total),
        "not_root_fallback": resolved != Path("/"),
        "expected_capacity_class": total >= EXPECTED_MINIMUM_BYTES,
    }
    payload["ready"] = bool(payload["not_root_fallback"] and payload["expected_capacity_class"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("/mnt/f"))
    args = parser.parse_args()
    payload = check(args.path)
    print(json.dumps(payload, indent=2))
    if not payload["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
