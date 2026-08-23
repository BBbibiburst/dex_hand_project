"""Download a bounded Google Scanned Objects candidate set from Gazebo Fuel.

The tool never downloads all 13 GB implicitly. Pass a selection file (one Fuel
model name per line), or use ``--list`` to create one after inspecting the
official catalogue. Downloaded models are merged into the project manifest and
tagged as metre-scale CC BY 4.0 assets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUEL_API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
DEFAULT_ROOT = PROJECT_ROOT / "assets" / "maniskill" / "gso" / "models"
DEFAULT_MANIFEST = PROJECT_ROOT / "assets" / "maniskill" / "manifest.json"
UNDERACTUATED_NAME_TOKENS = frozenset(
    {
        "bottle",
        "bottles",
        "canister",
        "container",
        "cup",
        "cups",
        "flask",
        "jar",
        "mug",
        "mugs",
        "pitcher",
        "shaker",
        "tumbler",
        "vase",
    }
)


def _request_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "dex-hand-project/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def list_models() -> list[dict]:
    records: list[dict] = []
    page = 1
    while True:
        batch = _request_json(f"{FUEL_API}?page={page}&per_page=100")
        if not isinstance(batch, list) or not batch:
            break
        records.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return records


def _selection(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"Empty GSO selection: {path}")
    return values


def underactuated_candidates(catalogue: list[dict]) -> list[str]:
    """Use metadata only to bound downloads; mesh UAS makes the selection."""
    selected: list[str] = []
    for item in catalogue:
        categories = set(item.get("categories", []))
        tokens = {token.lower() for token in str(item["name"]).split("_")}
        if "Bottles and Cans and Cups" in categories or tokens.intersection(
            UNDERACTUATED_NAME_TOKENS
        ):
            selected.append(str(item["name"]))
    return selected


def _safe_extract(bundle: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for info in bundle.infolist():
        target = (destination / info.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe archive member: {info.filename}")
    bundle.extractall(destination)


def _download(name: str, destination: Path, *, force: bool) -> None:
    mesh = destination / "meshes" / "model.obj"
    if mesh.is_file() and not force:
        print(f"[reuse] gso:{name}")
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    quoted = urllib.parse.quote(name, safe="")
    url = f"{FUEL_API}/{quoted}/tip/{quoted}.zip"
    archive = destination.with_suffix(".zip.partial")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "dex-hand-project/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as bundle:
            _safe_extract(bundle, destination)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        archive.unlink(missing_ok=True)
    if not mesh.is_file():
        raise FileNotFoundError(f"GSO archive has no meshes/model.obj: {name}")
    print(f"[ready] gso:{name}")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def merge_manifest(path: Path, root: Path, names: list[str], metadata: dict[str, dict]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"objects": []}
    selected = set(names)
    retained = [
        item
        for item in payload.get("objects", [])
        if item.get("dataset") != "gso" or item.get("object_id") not in selected
    ]
    for name in names:
        directory = root / name
        files = [item for item in directory.rglob("*") if item.is_file()]
        meta = metadata[name]
        retained.append(
            {
                "dataset": "gso",
                "object_id": name,
                "project_path": _relative(directory),
                "source_path": _relative(directory),
                "file_count": len(files),
                "size_bytes": sum(item.stat().st_size for item in files),
                "model_files": ["meshes/model.obj"],
                "scale_to_meters": 1.0,
                "license": meta.get("license_name", "CC BY 4.0"),
                "source_url": f"{FUEL_API}/{urllib.parse.quote(name, safe='')}",
                "categories": meta.get("categories", []),
            }
        )
    counts: dict[str, int] = {}
    for item in retained:
        dataset = item["dataset"]
        counts[dataset] = counts.get(dataset, 0) + 1
    payload.update(
        schema_version=2,
        actual_count_total=len(retained),
        actual_count_by_dataset=counts,
        objects=retained,
    )
    datasets = list(payload.get("datasets", []))
    if "Google Scanned Objects" not in datasets:
        datasets.append("Google Scanned Objects")
    payload["datasets"] = datasets
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--underactuated-candidates",
        action="store_true",
        help="Download the bounded GSO bottle/can/cup candidate pool.",
    )
    parser.add_argument("--list", action="store_true", help="Print official names and exit.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    catalogue = list_models()
    metadata = {item["name"]: item for item in catalogue}
    if args.list:
        for item in catalogue:
            print(f"{item['name']}\t{','.join(item.get('categories', []))}\t{item.get('filesize', 0)}")
        return 0
    if args.selection is not None and args.underactuated_candidates:
        parser.error("Use either --selection or --underactuated-candidates, not both")
    if args.underactuated_candidates:
        names = underactuated_candidates(catalogue)
    elif args.selection is not None:
        names = _selection(args.selection)
    else:
        parser.error("--selection or --underactuated-candidates is required unless --list is used")
    if args.limit:
        names = names[: args.limit]
    missing = sorted(set(names) - set(metadata))
    if missing:
        raise ValueError(f"Unknown GoogleResearch Fuel models: {missing}")
    root = args.root.resolve()
    for index, name in enumerate(names, 1):
        print(f"[{index:03d}/{len(names):03d}] {name}")
        _download(name, root / name, force=args.force)
    merge_manifest(args.manifest.resolve(), root, names, metadata)
    print(f"[done] merged {len(names)} GSO objects into {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
