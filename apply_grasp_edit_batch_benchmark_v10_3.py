#!/usr/bin/env python3
"""Upgrade the v10.1 batch benchmark directly to v10.3 lattice-preflight + adaptive scheduling."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import subprocess
import sys
import zipfile
from pathlib import Path

TARGET = 'tools/grasping/batch_grasp_edit_v10.py'
V10_1_SHA256 = '42c9c0f44901f44e8f321ecb52eb13fd414c8ca021f0895ac4263888bb77860c'
V10_2_SHA256 = 'bb25e8e4b5bc8617908ff0cb092c80ed420d7082b6451b9fcb7f6d988daa2c54'
NEW_SHA256 = '3ff20f62df455475e6fb0d42f4731ad4f8a9ba712b008b540c572a5e3c0557b4'
PREREQUISITES = {'apps/train_grasp_edit_rl.py': '19f554b7cf8e99b30c297f55cdaf0c697ec9762e2651edc43fc426be111c623d', 'source/rl/grasp_edit_hybrid_ppo.py': '72603c11b266085a214f9e7b233445ab38c8f0aee79d7338e192a7a274a25387', 'source/rl/grasp_edit_templates.py': '84c34c7c4d97838735b764a51ebf30161ea0252c0e9df35e8e8c506acd6f9822', 'source/rl/mjwarp_grasp_edit_env.py': 'c1a2842ad2f642067ce624b32942e3727b060f949a733e357385180cb2c8a5f1'}
PAYLOAD = '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'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError("Run this script inside dex_hand_project.")
    return Path(proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = repo_root()

    for rel, expected in PREREQUISITES.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"v10 prerequisite missing: {path}")
        current = digest(path)
        if current != expected:
            raise RuntimeError(
                f"{rel} is not the expected applied v10 source "
                f"(sha256={current}). No files were changed."
            )

    target_path = root / TARGET
    if not target_path.is_file():
        raise FileNotFoundError(
            f"v10.1 benchmark runner is missing: {target_path}. "
            "Apply v10.1 before this updater."
        )
    current = digest(target_path)
    if current == NEW_SHA256:
        print("v10.3 benchmark is already installed.")
        return 0
    if current == V10_1_SHA256:
        state = "upgrade v10.1 -> v10.3 directly (v10.2 is NOT required)"
    elif current == V10_2_SHA256:
        state = "upgrade v10.2 -> v10.3"
    else:
        raise RuntimeError(
            f"{TARGET} is not the expected v10.1/v10.2 runner "
            f"(sha256={current}). No files were changed."
        )

    print("v10.3 benchmark plan:")
    print(f"  {state}")
    print("  Ultra success -> ULTRA_SUCCESS, no lattice/RL by default")
    print("  Ultra failure -> CPU DIRECT lattice preflight first")
    print("  preflight has successful template -> LATTICE_SUCCESS, 0 RL updates")
    print("  only preflight failures enter adaptive PPO: 5 -> 10 -> 15 updates")
    print("  adaptive stages resume checkpoint_final.pt instead of restarting")
    print("  obvious no-progress cases stop after 5; ambiguous cases get 10")
    print("  only still-promising cases receive the final 15-update budget")
    print("  LATTICE_SUCCESS lift reports lattice lift, not Ultra failed lift")
    print("  detailed preflight/child output stays in per-object logs")
    print("  default output: outputs/grasp_edit_benchmark_v10_3")
    if args.dry_run:
        print("dry-run OK; no files changed")
        return 0

    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD)))
    content = archive.read(TARGET)
    if hashlib.sha256(content).hexdigest() != NEW_SHA256:
        raise RuntimeError("Embedded v10.3 payload checksum failed.")

    target_path.write_bytes(content)
    subprocess.run([sys.executable, "-m", "py_compile", TARGET], cwd=root, check=True)
    subprocess.run(["git", "--no-pager", "diff", "--check"], cwd=root, check=True)

    print("v10.3 benchmark installed.")
    print("smoke:")
    print(
        "  python -m tools.grasping.batch_grasp_edit_v10 "
        "--dataset all --limit 3 --expect-count 0"
    )
    print("full:")
    print(
        "  python -m tools.grasping.batch_grasp_edit_v10 "
        "--dataset all --expect-count 127"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
