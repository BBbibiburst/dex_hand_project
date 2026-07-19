"""Download and collect ManiSkill YCB plus the official EGAD evaluation set.

This tool downloads YCB through ManiSkill and EGAD from its official archive, then merges them into a
single output folder, recording exactly how many objects came from each
source. It never silently substitutes or pads with unrelated data: every
object in the final manifest is traceable back to "ycb" or "egad".

Target composition:
    - YCB:  all model directories provided by the installed ManiSkill release
    - EGAD: the official 7x7 complexity/difficulty evaluation subset
            (expected 49; IDs look like "A0" .. "G6")
    - Total: actual YCB count + 49 (usually 126 or 127)

Examples:

    python tools/download_maniskill_objects.py
    python tools/download_maniskill_objects.py --dry-run
    python tools/download_maniskill_objects.py --datasets ycb egad
    python tools/download_maniskill_objects.py --egad-selection docs/egad_eval49.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "assets" / "maniskill"
DEFAULT_OUTPUT_DIR = DEFAULT_CACHE_DIR / "models"
DEFAULT_MANIFEST = DEFAULT_CACHE_DIR / "manifest.json"
DEFAULT_LOCK_DIR = PROJECT_ROOT / "docs"

MODEL_SUFFIXES = {".dae", ".glb", ".gltf", ".obj", ".ply", ".stl", ".urdf", ".xml"}

EGAD_EVAL_IDS = tuple(f"{letter}{number}" for letter in "ABCDEFG" for number in range(7))
EGAD_EVAL_EXPECTED_COUNT = len(EGAD_EVAL_IDS)
EGAD_EVAL_URL = (
    "https://data.researchdatafinder.qut.edu.au/dataset/egad---evolved/"
    "resource/f01c0b75-aa6d-4af9-b2a9-5edfee823e03/download/egad_eval_set.zip"
)
YCB_CACHE_RELATIVE_ROOT = Path("ycb")
EGAD_CACHE_RELATIVE_ROOT = Path("egad")


@dataclass(frozen=True)
class DatasetSpec:
    name: str  # "ycb" or "egad"
    asset_uid: str | None  # the string passed to ManiSkill, if it provides the dataset
    # Candidate cache-relative roots to check, in priority order. ManiSkill's
    # on-disk layout has changed across versions/forks, so we probe a few
    # known possibilities and fall back to a recursive search by name.
    relative_roots: tuple[Path, ...]
    expected_count: int | None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "ycb": DatasetSpec(
        name="ycb",
        asset_uid="ycb",
        relative_roots=(
            YCB_CACHE_RELATIVE_ROOT,
            Path("data") / "assets" / "mani_skill2_ycb",
            Path("assets") / "mani_skill2_ycb",
            Path("data") / "assets" / "ycb",
        ),
        # ManiSkill releases have exposed 77 or 78 model directories. Select
        # every valid model and record the actual count instead of dropping one.
        expected_count=None,
    ),
    "egad": DatasetSpec(
        name="egad",
        asset_uid=None,
        relative_roots=(EGAD_CACHE_RELATIVE_ROOT,),
        expected_count=EGAD_EVAL_EXPECTED_COUNT,
    ),
}


@dataclass(frozen=True)
class ObjectRecord:
    dataset: str
    object_id: str
    project_path: str
    source_path: str
    file_count: int
    size_bytes: int
    model_files: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=["ycb", "egad"],
        help="Which datasets to download and merge. Default: ycb egad.",
    )
    parser.add_argument(
        "--ycb-selection",
        type=Path,
        help="Optional text file containing one YCB object ID per line.",
    )
    parser.add_argument(
        "--ycb-count",
        type=int,
        default=0,
        help="Optional maximum number of YCB objects; 0 selects all.",
    )
    parser.add_argument(
        "--egad-selection",
        type=Path,
        help=(
            "Optional text file containing one EGAD object ID per line. "
            "If omitted, the tool auto-selects the official 7x7 eval subset "
            "(IDs matching [A-G][0-6]) when present."
        ),
    )
    parser.add_argument(
        "--egad-count",
        type=int,
        default=0,
        help="Optional maximum number of EGAD objects; 0 selects the auto-detected set.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy" if os.name == "nt" else "symlink",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the local selection without downloading or copying.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Allow ManiSkill's downloader to replace an existing dataset cache.",
    )
    parser.add_argument(
        "--force-output",
        action="store_true",
        help="Replace existing collected object directories.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_mani_skill(cache_dir: Path) -> None:
    os.environ["MS_ASSET_DIR"] = str(cache_dir)
    if importlib.util.find_spec("mani_skill") is None:
        raise RuntimeError(
            "ManiSkill is required only for this asset tool. Install it with:\n"
            "  python -m pip install --upgrade mani_skill"
        )


def find_dataset_root(cache_dir: Path, spec: DatasetSpec) -> Path:
    """Return the on-disk root for a dataset, probing supported layouts."""
    for relative in spec.relative_roots:
        candidate = cache_dir / relative
        if candidate.exists():
            return candidate
    return cache_dir / spec.relative_roots[0]


def models_root(dataset_root: Path) -> Path:
    """Some ManiSkill layouts nest a 'models' folder; others don't. Probe
    both so object discovery works either way."""
    nested = dataset_root / "models"
    return nested if nested.is_dir() else dataset_root


def normalize_ycb_cache(cache_dir: Path) -> Path | None:
    """Move ManiSkill's version-dependent YCB tree into ``ycb/``."""
    canonical = cache_dir / YCB_CACHE_RELATIVE_ROOT
    if (canonical / "models").is_dir():
        return canonical
    spec = DATASET_SPECS["ycb"]
    candidates = [
        *(
            cache_dir / relative
            for relative in spec.relative_roots[1:]
            if (cache_dir / relative / "models").is_dir()
        ),
        PROJECT_ROOT / "assets" / "ycb",
    ]
    legacy = next((path for path in candidates if (path / "models").is_dir()), None)
    if legacy is None:
        return None
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if canonical.exists():
        raise RuntimeError(f"Cannot normalize YCB cache because destination exists: {canonical}")
    print(f"[organize] {legacy} -> {canonical}")
    shutil.move(str(legacy), str(canonical))
    return canonical


def download_egad(cache_dir: Path, *, force: bool) -> None:
    """Download and normalize the official flat EGAD archive to one directory per object."""
    target = cache_dir / EGAD_CACHE_RELATIVE_ROOT / "models"
    legacy_target = cache_dir / "egad" / "eval" / "models"
    if not target.exists() and legacy_target.is_dir():
        print(f"[organize] {legacy_target} -> {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_target), str(target))
    expected_paths = {
        object_id: target / object_id / f"{object_id}.obj" for object_id in EGAD_EVAL_IDS
    }
    if not force and all(path.is_file() for path in expected_paths.values()):
        print(f"[reuse] egad cache: {target}")
        return

    archive = cache_dir / "downloads" / "egad_eval_set.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.partial")
    print(f"[download] {EGAD_EVAL_URL}")
    try:
        request = urllib.request.Request(
            EGAD_EVAL_URL,
            headers={"User-Agent": "dex-hand-project-asset-downloader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)

    with zipfile.ZipFile(archive) as bundle:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in bundle.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() != ".obj":
                continue
            object_id = Path(info.filename).stem
            if object_id in EGAD_EVAL_IDS:
                if object_id in members:
                    raise RuntimeError(f"EGAD archive contains duplicate model {object_id}.obj")
                members[object_id] = info
        missing = sorted(set(EGAD_EVAL_IDS) - set(members))
        unexpected_count = len(members) != EGAD_EVAL_EXPECTED_COUNT
        if missing or unexpected_count:
            raise RuntimeError(
                f"Invalid EGAD evaluation archive: found {len(members)} expected OBJ files; "
                f"missing={missing}"
            )
        for object_id, info in members.items():
            destination = expected_paths[object_id]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    print(f"[ready] egad: {EGAD_EVAL_EXPECTED_COUNT} models at {target}")


def download_dataset(cache_dir: Path, spec: DatasetSpec, *, force: bool) -> None:
    if spec.name == "egad":
        download_egad(cache_dir, force=force)
        return
    if spec.asset_uid is None:
        raise RuntimeError(f"No downloader is configured for dataset '{spec.name}'.")
    if spec.name == "ycb":
        normalized = normalize_ycb_cache(cache_dir)
        target = normalized if normalized is not None else find_dataset_root(cache_dir, spec)
    else:
        target = find_dataset_root(cache_dir, spec)
    if (models_root(target)).is_dir() and contains_model_files(models_root(target)) and not force:
        if spec.name == "ycb":
            target = normalize_ycb_cache(cache_dir) or target
        print(f"[reuse] {spec.name} cache: {target}")
        return
    require_mani_skill(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MS_ASSET_DIR"] = str(cache_dir)
    env["MS_SKIP_ASSET_DOWNLOAD_PROMPT"] = "1"
    command = [
        sys.executable,
        "-m",
        "mani_skill.utils.download_asset",
        spec.asset_uid,
        "-y",
    ]
    print(f"[download] {' '.join(command)}")
    subprocess.run(command, check=True, env=env)
    if spec.name == "ycb":
        target = normalize_ycb_cache(cache_dir)
        if target is None:
            raise FileNotFoundError(f"Downloaded YCB models were not found under {cache_dir}")
    else:
        target = find_dataset_root(cache_dir, spec)
    if not models_root(target).exists():
        raise FileNotFoundError(
            f"ManiSkill reported success but no '{spec.name}' directory was found "
            f"under {cache_dir}. Inspect {cache_dir} manually and adjust "
            f"DATASET_SPECS['{spec.name}'].relative_roots if the layout changed."
        )


def contains_model_files(path: Path) -> bool:
    return any(item.is_file() and item.suffix.lower() in MODEL_SUFFIXES for item in path.rglob("*"))


def discover_object_ids(cache_dir: Path, spec: DatasetSpec) -> list[str]:
    root = models_root(find_dataset_root(cache_dir, spec))
    if not root.is_dir():
        raise FileNotFoundError(f"{spec.name} models directory not found: {root}")
    object_ids = [
        path.name for path in sorted(root.iterdir()) if path.is_dir() and contains_model_files(path)
    ]
    if not object_ids:
        raise RuntimeError(f"No {spec.name} model directories were found under {root}")
    return object_ids


def load_selection(path: Path, *, strip_prefix: str | None = None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Selection file not found: {path}")
    selected: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        object_id = line.split("#", maxsplit=1)[0].strip()
        if strip_prefix and object_id.startswith(strip_prefix):
            object_id = object_id.removeprefix(strip_prefix)
        if object_id and object_id not in seen:
            selected.append(object_id)
            seen.add(object_id)
    return selected


def choose_ycb_objects(available: list[str], selection: Path | None, count: int) -> list[str]:
    if count < 0:
        raise ValueError("--ycb-count must be non-negative.")
    if selection is None:
        selected = available.copy()
    else:
        selected = load_selection(selection, strip_prefix="ycb:")
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"Selected YCB object IDs do not exist: {missing}")
    if count:
        selected = selected[:count]
    if not selected:
        raise ValueError("The YCB selection is empty.")
    return selected


def choose_egad_objects(available: list[str], selection: Path | None, count: int) -> list[str]:
    if count < 0:
        raise ValueError("--egad-count must be non-negative.")

    if selection is not None:
        selected = load_selection(selection, strip_prefix="egad:")
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"Selected EGAD object IDs do not exist: {missing}")
    else:
        selected = list(EGAD_EVAL_IDS)
        missing = sorted(set(selected) - set(available))
        if missing:
            raise RuntimeError(
                f"Official EGAD evaluation cache is incomplete; missing IDs: {missing}"
            )

    if count:
        selected = selected[:count]
    if not selected:
        raise ValueError("The EGAD selection is empty.")
    return selected


def write_lock_file(path: Path, dataset: str, object_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by tools/download_maniskill_objects.py",
        f"# One stable {dataset.upper()} object ID per line.",
        *object_ids,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def transfer(source: Path, destination: Path, mode: str, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not force:
            print(f"[reuse] output: {destination.name}")
            return
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    if mode == "copy":
        shutil.copytree(source, destination)
        return
    try:
        destination.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create directory symlink {destination}. On Windows use --mode copy "
            "or enable Developer Mode."
        ) from exc


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def make_record(dataset: str, object_id: str, source: Path, destination: Path) -> ObjectRecord:
    files = [path for path in source.rglob("*") if path.is_file()]
    model_files = sorted(
        path.relative_to(source).as_posix()
        for path in files
        if path.suffix.lower() in MODEL_SUFFIXES
    )
    return ObjectRecord(
        dataset=dataset,
        object_id=object_id,
        project_path=relative_to_project(destination),
        source_path=relative_to_project(source),
        file_count=len(files),
        size_bytes=sum(path.stat().st_size for path in files),
        model_files=model_files,
    )


def write_manifest(path: Path, records: list[ObjectRecord], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_dataset_counts: dict[str, int] = {}
    for record in records:
        per_dataset_counts[record.dataset] = per_dataset_counts.get(record.dataset, 0) + 1
    payload = {
        "schema_version": 1,
        "datasets": ["ManiSkill YCB", "Official EGAD evaluation subset"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actual_count_total": len(records),
        "actual_count_by_dataset": per_dataset_counts,
        "transfer_mode": mode,
        "license_notice": (
            "ManiSkill code is Apache-2.0; downloaded assets have their own "
            "original licenses (YCB: CC BY 4.0; EGAD: CC BY 4.0). "
            "ManiSkill documents its distributed asset bundle as CC BY-NC 4.0. "
            "Verify original terms before commercial redistribution."
        ),
        "objects": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def object_output_name(dataset: str, object_id: str) -> str:
    # Namespaced with the source dataset to guarantee no collisions when
    # merging multiple datasets into one flat output folder.
    return f"{dataset}__{object_id}"


def migrate_legacy_output(output_dir: Path, manifest_path: Path) -> None:
    """Reuse output created by versions that wrote to ``assets/combined``."""
    legacy_root = PROJECT_ROOT / "assets" / "combined"
    legacy_output = legacy_root / "models"
    legacy_manifest = legacy_root / "manifest.json"
    if output_dir == DEFAULT_OUTPUT_DIR and not output_dir.exists() and legacy_output.is_dir():
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[organize] {legacy_output} -> {output_dir}")
        shutil.move(str(legacy_output), str(output_dir))
    if (
        manifest_path == DEFAULT_MANIFEST
        and not manifest_path.exists()
        and legacy_manifest.is_file()
    ):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[organize] {legacy_manifest} -> {manifest_path}")
        shutil.move(str(legacy_manifest), str(manifest_path))
    if legacy_root.is_dir() and not any(legacy_root.iterdir()):
        legacy_root.rmdir()


def main() -> int:
    args = parse_args()
    cache_dir = project_path(args.cache_dir)
    output_dir = project_path(args.output_dir)
    manifest_path = project_path(args.manifest)
    lock_dir = project_path(args.lock_dir)
    ycb_selection_path = project_path(args.ycb_selection) if args.ycb_selection else None
    egad_selection_path = project_path(args.egad_selection) if args.egad_selection else None
    migrate_legacy_output(output_dir, manifest_path)

    per_dataset_selected: dict[str, list[str]] = {}

    for dataset_name in args.datasets:
        spec = DATASET_SPECS[dataset_name]

        if args.dry_run and not find_dataset_root(cache_dir, spec).exists():
            raise RuntimeError(
                f"--dry-run never downloads files, but the local {dataset_name} cache "
                "is missing. Run once without --dry-run first."
            )
        if not args.dry_run:
            download_dataset(cache_dir, spec, force=args.force_download)

        available = discover_object_ids(cache_dir, spec)

        if dataset_name == "ycb":
            selected = choose_ycb_objects(available, ycb_selection_path, args.ycb_count)
        else:  # egad
            selected = choose_egad_objects(available, egad_selection_path, args.egad_count)

        per_dataset_selected[dataset_name] = selected
        print(
            f"[plan] {dataset_name}: selected {len(selected)} of {len(available)} available objects"
        )
        if spec.expected_count is not None and len(selected) != spec.expected_count:
            print(
                f"[note] {dataset_name} selected {len(selected)} objects, "
                f"expected {spec.expected_count}. This may be intentional "
                "(e.g. --ycb-count or --egad-selection), but double-check."
            )

    total_selected = sum(len(ids) for ids in per_dataset_selected.values())
    print(f"[plan] combined total across datasets: {total_selected}")

    if args.dry_run:
        for dataset_name, selected in per_dataset_selected.items():
            print(f"\n== {dataset_name} ({len(selected)}) ==")
            for index, object_id in enumerate(selected, start=1):
                print(f"{index:03d}. {object_id}")
        return 0

    records: list[ObjectRecord] = []
    for dataset_name, selected in per_dataset_selected.items():
        spec = DATASET_SPECS[dataset_name]
        write_lock_file(lock_dir / f"{dataset_name}_objects.lock.txt", dataset_name, selected)
        source_root = models_root(find_dataset_root(cache_dir, spec))
        for index, object_id in enumerate(selected, start=1):
            source = source_root / object_id
            destination = output_dir / object_output_name(dataset_name, object_id)
            print(f"[{dataset_name} {index:03d}/{len(selected):03d}] {object_id}")
            transfer(source, destination, args.mode, force=args.force_output)
            records.append(make_record(dataset_name, object_id, source, destination))

    write_manifest(manifest_path, records, args.mode)
    print(f"[done] combined objects folder: {output_dir}")
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] actual combined count: {len(records)}")
    for dataset_name, selected in per_dataset_selected.items():
        print(f"  - {dataset_name}: {len(selected)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
