"""Build a diverse geometry candidate pool for subsequent physics validation."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import trimesh

from source.envs.manipulation.object_catalog import (
    MANIFEST_PATH,
    object_records,
    record_scale_to_meters,
)
from source.grasping.catalog import resolve_object_mesh
from source.grasping.affordance import geometry_affordance, simulate_initial_pose_stability


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CATEGORY_QUOTAS = {
    "cylinder": 27,
    "box": 9,
    "container": 4,
    "sphere": 6,
    "egad": 6,
    "regular": 32,
    "boundary": 16,
}

DEFAULT_DATASET_QUOTAS = None

BENCHMARK_FAMILY_LIMITS = {
    "supplement": 6,
    "fruit": 5,
    "ball": 3,
    "ramekin": 2,
    "building_block": 3,
    "ink_cartridge": 2,
    "pencil_case": 1,
    "catalog_variant": 1,
    "cup": 1,
    "haircare": 1,
    "toy_airplane": 1,
}


@dataclass(frozen=True)
class GeometryAssessment:
    object_id: str
    dataset: str
    geometry: dict
    adaptive_contact: None
    robustness: None
    uas: None
    eligible: bool
    reasons: list[str]


def semantic_category(item: GeometryAssessment) -> str:
    """Assign a coarse grasp family; names only disambiguate semantic geometry."""
    name = item.object_id.lower()
    tokens = set(re.findall(r"[a-z0-9]+", name.split(":", 1)[-1]))
    geometry = item.geometry
    if item.dataset == "egad":
        return "egad"
    if tokens.intersection(("ball", "apple", "orange", "lemon", "peach", "pear", "plum", "strawberry")):
        return "sphere"
    if tokens.intersection(("cup", "cups", "mug", "mugs", "bowl", "bowls", "ramekin", "planter", "container")):
        return "container"
    if tokens.intersection(("box", "block", "blocks", "cube", "lego", "brick")):
        return "box"
    if tokens.intersection(("bottle", "bottles", "can", "cans", "canister", "jar", "flask", "shaker", "tumbler")):
        return "cylinder"
    extents = sorted(float(value) for value in geometry.get("extents_m", (1.0, 1.0, 1.0)))
    transverse_balance = extents[0] / max(extents[1], 1e-9)
    if (
        transverse_balance >= 0.78
        and 1.25 <= float(geometry.get("axis_ratio", 1.0)) <= 3.5
        and float(geometry.get("rotational_symmetry", 0.0)) >= 0.50
    ):
        return "cylinder"
    if float(geometry.get("axis_ratio", 1.0)) > 3.0 or float(geometry.get("convexity", 1.0)) < 0.65:
        return "boundary"
    return "regular"


def shape_signature(item: GeometryAssessment) -> tuple:
    """Quantize scale and shape so near-identical catalogue scans share a bucket."""
    geometry = item.geometry
    extents = sorted(float(value) for value in geometry["extents_m"])
    return (
        semantic_category(item),
        round(extents[0] / 0.008),
        round(extents[1] / 0.008),
        round(extents[2] / 0.015),
        round(float(geometry["sphericity"]) / 0.05),
        round(float(geometry["convexity"]) / 0.10),
    )


def product_family(item: GeometryAssessment) -> str:
    """Group catalogue variants from the same branded or procedural series."""
    local_id = item.object_id.split(":", 1)[-1]
    if item.dataset == "ycb":
        number, _, name = local_id.partition("_")
        number = number.split("-", 1)[0]
        return f"ycb:{number}:{name.replace('colored_', '')}"
    if item.dataset == "egad":
        return f"egad:{local_id[:1]}"
    if item.dataset != "gso":
        return item.object_id
    tokens = re.findall(r"[A-Za-z0-9]+", local_id)
    generic = {"3d", "object", "mens", "womens"}
    useful = [token.lower() for token in tokens if token.lower() not in generic]
    return useful[0] if useful else item.object_id.lower()


def benchmark_family(item: GeometryAssessment) -> str | None:
    """Group semantically interchangeable objects that geometry alone misses."""
    name = item.object_id.lower().replace("-", "_")
    tokens = set(re.findall(r"[a-z0-9]+", name.split(":", 1)[-1]))
    if tokens.intersection({"apple", "orange", "lemon", "peach", "pear", "plum", "strawberry"}):
        return "fruit"
    if "ball" in tokens:
        return "ball"
    if tokens.intersection({"ramekin"}):
        return "ramekin"
    if tokens.intersection({"lego", "duplo", "block", "blocks", "brick"}):
        return "building_block"
    if "ink" in tokens and "cartridge" in tokens:
        return "ink_cartridge"
    if "pencil" in tokens and "case" in tokens:
        return "pencil_case"
    if "cup" in tokens or "cups" in tokens:
        return "cup"
    if "airplane" in tokens:
        return "toy_airplane"
    if ("marc" in tokens and "anthony" in tokens) or "haircolor" in tokens:
        return "haircare"
    if (
        ("same" in tokens and "200" in tokens)
        or "labelworks" in tokens
        or "weston" in tokens
    ):
        return "catalog_variant"
    compact = "".join(sorted(tokens))
    supplement_fragments = {
        "5htp", "acid", "betaglucan", "bifidus", "blackcurrant", "borage", "caplets", "coq10", "creatine",
        "dophilus", "folic", "germanium", "glucosamine", "inositol", "jarrow", "jarro",
        "krill", "lactoferrin", "lutein", "masticgum", "mk7", "multivitamin", "natto", "omega", "pomegranate", "probiotic",
        "prostate", "quercetin", "saccharomyces", "supplement", "theanine", "tocopherol",
        "twinlab", "tyrosine", "whey",
    }
    if tokens.intersection({"dim", "multi"}) or any(
        fragment in compact for fragment in supplement_fragments
    ):
        return "supplement"
    return None


def diverse_selection(
    ranked: list[GeometryAssessment],
    *,
    count: int,
    maximum_near_duplicates: int = 2,
    quotas: dict[str, int] | None = None,
    dataset_quotas: dict[str, int] | None = None,
    excluded_ids: set[str] | None = None,
) -> list[GeometryAssessment]:
    """Stratify grasp families, then fill remaining slots without clone fallback."""
    quotas = dict(DEFAULT_CATEGORY_QUOTAS if quotas is None else quotas)
    excluded_ids = set() if excluded_ids is None else set(excluded_ids)
    selected: list[GeometryAssessment] = []
    selected_ids: set[str] = set()
    signature_counts: dict[tuple, int] = {}
    dataset_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    benchmark_family_counts: dict[str, int] = {}

    def take(item: GeometryAssessment) -> bool:
        signature = shape_signature(item)
        if (
            item.object_id in excluded_ids
            or item.object_id in selected_ids
            or signature_counts.get(signature, 0) >= maximum_near_duplicates
        ):
            return False
        family = product_family(item)
        if family_counts.get(family, 0) >= 2:
            return False
        benchmark_group = benchmark_family(item)
        if (
            benchmark_group is not None
            and benchmark_family_counts.get(benchmark_group, 0)
            >= BENCHMARK_FAMILY_LIMITS[benchmark_group]
        ):
            return False
        category = semantic_category(item)
        if category_counts.get(category, 0) >= quotas.get(category, count):
            return False
        if dataset_quotas is not None and dataset_counts.get(item.dataset, 0) >= dataset_quotas.get(item.dataset, count):
            return False
        selected.append(item)
        selected_ids.add(item.object_id)
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        dataset_counts[item.dataset] = dataset_counts.get(item.dataset, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        if benchmark_group is not None:
            benchmark_family_counts[benchmark_group] = (
                benchmark_family_counts.get(benchmark_group, 0) + 1
            )
        category_counts[category] = category_counts.get(category, 0) + 1
        return True

    for category, quota in quotas.items():
        admitted = 0
        for item in ranked:
            if semantic_category(item) == category and take(item):
                admitted += 1
                if admitted >= quota or len(selected) >= count:
                    break
    if dataset_quotas is not None:
        for dataset, quota in dataset_quotas.items():
            for item in ranked:
                if item.dataset == dataset:
                    take(item)
                if dataset_counts.get(dataset, 0) >= quota:
                    break
    for item in ranked:
        if take(item) and len(selected) >= count:
            break
    return selected[:count]


def assess(object_id: str) -> GeometryAssessment:
    record = object_records()[object_id]
    mesh = trimesh.load(resolve_object_mesh(object_id), force="mesh", process=True)
    geometry = geometry_affordance(mesh, scale_to_meters=record_scale_to_meters(record))
    return GeometryAssessment(
        object_id=object_id,
        dataset=record["dataset"],
        geometry=geometry.to_dict(),
        adaptive_contact=None,
        robustness=None,
        uas=None,
        eligible=geometry.eligible,
        reasons=list(geometry.reasons),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--minimum-prior",
        type=float,
        default=0.50,
        help="Minimum geometry prior for the diverse prefilter. Default: 0.50.",
    )
    parser.add_argument(
        "--maximum-near-duplicates",
        type=int,
        default=2,
        help="Maximum objects in one quantized shape/size bucket before fallback filling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs/underactuated_top100.json",
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        help="Optional JSON list (or object with an 'objects' list) of object ids to exclude.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/underactuated_geometry_report.json",
    )
    parser.add_argument(
        "--input-report",
        type=Path,
        help="Reuse assessments from an existing report instead of rescanning every mesh.",
    )
    parser.add_argument(
        "--skip-initial-stability",
        action="store_true",
        help="Skip MuJoCo free-settling validation (geometry diagnostics only).",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if not 0.0 <= args.minimum_prior <= 1.0:
        parser.error("--minimum-prior must lie in [0, 1]")
    if args.maximum_near_duplicates <= 0:
        parser.error("--maximum-near-duplicates must be positive")
    assessments: list[GeometryAssessment] = []
    excluded_ids: set[str] = set()
    if args.exclude_file is not None:
        payload = json.loads(args.exclude_file.read_text(encoding="utf-8"))
        values = payload.get("objects", []) if isinstance(payload, dict) else payload
        excluded_ids = {
            str(value.get("object_id", value.get("id"))) if isinstance(value, dict) else str(value)
            for value in values
        }
    if args.input_report is not None:
        cached = json.loads(args.input_report.read_text(encoding="utf-8"))
        assessments = [GeometryAssessment(**item) for item in cached["assessments"]]
        print(f"[cache] loaded {len(assessments)} assessments from {args.input_report}")
    else:
        for index, object_id in enumerate(object_records(), 1):
            try:
                item = assess(object_id)
            except Exception as exc:
                record = object_records()[object_id]
                item = GeometryAssessment(
                    object_id, record["dataset"], {}, None, None, None,
                    False, [f"mesh_error:{type(exc).__name__}"],
                )
            assessments.append(item)
            prior = float(item.geometry.get("geometry_prior", 0.0))
            print(
                f"[{index:03d}/{len(object_records()):03d}] {object_id}: "
                f"prior={prior:.3f} {item.reasons}"
            )
    ranked = sorted(
        (
            item
            for item in assessments
            if item.eligible
            and float(item.geometry["geometry_prior"]) >= args.minimum_prior
        ),
        key=lambda x: (-float(x.geometry["geometry_prior"]), x.object_id),
    )
    stability_results: dict[str, dict] = {}
    if not args.skip_initial_stability:
        physically_stable: list[GeometryAssessment] = []
        for index, item in enumerate(ranked, 1):
            try:
                stability = simulate_initial_pose_stability(item.object_id)
                result = stability.to_dict()
            except Exception as exc:
                result = {
                    "stable": False,
                    "settled": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                result["error"] = None
            stability_results[item.object_id] = result
            if result["stable"]:
                physically_stable.append(item)
            print(
                f"[stability {index:03d}/{len(ranked):03d}] {item.object_id}: "
                f"{'stable' if result['stable'] else 'unstable'}"
            )
        ranked = physically_stable
    selected = diverse_selection(
        ranked,
        count=args.count,
        maximum_near_duplicates=args.maximum_near_duplicates,
        dataset_quotas=DEFAULT_DATASET_QUOTAS,
        excluded_ids=excluded_ids,
    )
    report_payload = {
        "schema_version": 1,
        "source_manifest": str(MANIFEST_PATH),
        "requested_count": args.count,
        "minimum_geometry_prior": args.minimum_prior,
        "eligible_count": len(ranked),
        "initial_stability_validated": not args.skip_initial_stability,
        "initial_stability": stability_results,
        "assessments": [asdict(item) for item in assessments],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    if len(selected) < args.count:
        raise RuntimeError(
            f"Only {len(selected)} objects meet geometry and prior thresholds; "
            f"add candidates before selecting {args.count}."
        )
    lock_payload = {
        "schema_version": 1,
        "name": "Underactuated-hand geometry candidate pool",
        "selection_stage": "geometry_prefilter",
        "selection_method": "stratified_underactuated_geometry_prior_v3",
        "minimum_geometry_prior": args.minimum_prior,
        "category_quotas": DEFAULT_CATEGORY_QUOTAS,
        "dataset_quotas": DEFAULT_DATASET_QUOTAS,
        "maximum_near_duplicates": args.maximum_near_duplicates,
        "requires_physics_validation": True,
        "initial_stability_validated": not args.skip_initial_stability,
        "excluded_object_ids": sorted(excluded_ids),
        "objects": [
            asdict(item)
            | {
                "category": semantic_category(item),
                "initial_pose_stability": stability_results.get(item.object_id),
            }
            for item in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[done] selected {len(selected)} objects -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
