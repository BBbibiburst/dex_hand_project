"""Check out the pinned UltraDexGrasp reference repositories."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_PATH = PROJECT_ROOT / "deps" / "ultradexgrasp" / "versions.json"
CHECKOUT_PATHS = {
    "ultradexgrasp": PROJECT_ROOT / "deps" / "ultradexgrasp" / "upstream",
    "bodex_api": PROJECT_ROOT / "deps" / "ultradexgrasp" / "third_party" / "BODex_api",
    "curobo": PROJECT_ROOT / "deps" / "ultradexgrasp" / "third_party" / "curobo",
    "pytorch3d": PROJECT_ROOT / "deps" / "ultradexgrasp" / "third_party" / "pytorch3d",
}


def _run(*command: str, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=capture,
        text=True,
    )
    return result.stdout.strip() if capture else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone-missing", action="store_true")
    parser.add_argument("--update", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    versions = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
    failed = False
    for name, spec in versions.items():
        path = CHECKOUT_PATHS[name]
        if not (path / ".git").exists():
            if not args.clone_missing:
                print(f"[missing] {name}: {path}", flush=True)
                failed = True
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            _run("git", "clone", spec["url"], str(path))
            _run("git", "-C", str(path), "checkout", "--detach", spec["revision"])
        actual = _run("git", "-C", str(path), "rev-parse", "HEAD", capture=True)
        expected = spec["revision"]
        if actual != expected and args.update:
            dirty = _run("git", "-C", str(path), "status", "--porcelain", capture=True)
            if dirty:
                raise RuntimeError(f"Refusing to update dirty checkout: {path}")
            _run("git", "-C", str(path), "fetch", "origin", expected)
            _run("git", "-C", str(path), "checkout", "--detach", expected)
            actual = _run("git", "-C", str(path), "rev-parse", "HEAD", capture=True)
        ok = actual == expected
        failed = failed or not ok
        print(
            f"[{'ok' if ok else 'mismatch'}] {name}: expected={expected} actual={actual}",
            flush=True,
        )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
