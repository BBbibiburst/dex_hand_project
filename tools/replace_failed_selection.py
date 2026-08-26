"""Replace selected failure classes while preserving benchmark diversity."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from source.envs.manipulation.object_catalog import object_records
from source.grasping.affordance import simulate_initial_pose_stability
from tools.rank_underactuated_candidates import (
    BENCHMARK_FAMILY_LIMITS,
    GeometryAssessment,
    assess,
    benchmark_family,
    product_family,
    semantic_category,
    shape_signature,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assessment(item: dict) -> GeometryAssessment:
    return GeometryAssessment(
        object_id=item["object_id"],
        dataset=item["dataset"],
        geometry=item["geometry"],
        adaptive_contact=item.get("adaptive_contact"),
        robustness=item.get("robustness"),
        uas=item.get("uas"),
        eligible=bool(item["eligible"]),
        reasons=list(item.get("reasons", ())),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--failure-classes", nargs="+", default=["initial_pose_unstable", "no_strict_grasp"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-prior", type=float, default=0.50)
    args = parser.parse_args()

    payload = json.loads(args.selection.read_text(encoding="utf-8"))
    failure = json.loads(args.failure_report.read_text(encoding="utf-8"))["categories"]
    removed_ids = {
        object_id
        for key in args.failure_classes
        for object_id in failure[key]["object_ids"]
    }
    original = payload["objects"]
    removed = [_assessment(item) for item in original if item["object_id"] in removed_ids]
    retained = [_assessment(item) for item in original if item["object_id"] not in removed_ids]
    signature_counts = Counter(shape_signature(item) for item in retained)
    product_counts = Counter(product_family(item) for item in retained)
    benchmark_counts = Counter(
        group for item in retained if (group := benchmark_family(item)) is not None
    )

    candidates = []
    unavailable = (
        {item.object_id for item in retained}
        | removed_ids
        | set(payload.get("excluded_object_ids", ()))
    )
    for index, object_id in enumerate(object_records(), 1):
        if object_id in unavailable:
            continue
        try:
            item = assess(object_id)
        except Exception:
            continue
        size_only_reasons = set(item.reasons).issubset(
            {"grasp_span_too_large", "overall_size_too_large"}
        )
        if (item.eligible or size_only_reasons) and float(item.geometry["geometry_prior"]) >= args.minimum_prior:
            candidates.append(item)
        if index % 200 == 0:
            print(f"[geometry] {index}/{len(object_records())} eligible={len(candidates)}")
    candidates.sort(key=lambda item: (-float(item.geometry["geometry_prior"]), item.object_id))

    replacements: list[GeometryAssessment] = []
    stability_rows: dict[str, dict] = {}
    # Replacement objects must be regular enough for passive enclosure. The
    # original geometry prior alone can overrate complex toys via their convex
    # hull, so reject boundary/procedural shapes and low-convexity bodies here.
    for item in candidates:
        if len(replacements) >= len(removed):
            break
        if item in replacements:
            continue
        geometry = item.geometry
        name = item.object_id.lower()
        name_tokens = set(re.findall(r"[a-z0-9]+", name.split(":", 1)[-1]))
        suitable_tokens = {
            "apple", "box", "bottle", "can", "cans", "canister", "jar",
            "cup", "cups", "mug", "bowl", "container", "planter", "pot",
            "carton", "package", "pack", "candy", "crayon", "crayons",
            "xylitol", "case", "cleanser",
        }
        if (
            item.dataset == "egad"
            or not name_tokens.intersection(suitable_tokens)
            or float(geometry["axis_ratio"]) > 5.0
            or float(geometry["power_grasp_suitability"]) < 0.50
        ):
            continue
        signature = shape_signature(item)
        family = product_family(item)
        benchmark = benchmark_family(item)
        if signature_counts[signature] >= 5 or product_counts[family] >= 2:
            continue
        if benchmark is not None and benchmark_counts[benchmark] >= BENCHMARK_FAMILY_LIMITS[benchmark]:
            continue
        stability = simulate_initial_pose_stability(item.object_id)
        stability_rows[item.object_id] = stability.to_dict()
        print(
            f"[fallback] {item.object_id} prior={item.geometry['geometry_prior']:.3f} "
            f"{'stable' if stability.stable else 'rejected'}"
        )
        if not stability.stable:
            continue
        replacements.append(item)
        signature_counts[signature] += 1
        product_counts[family] += 1
        if benchmark is not None:
            benchmark_counts[benchmark] += 1
    if len(replacements) != len(removed):
        raise RuntimeError(f"Only found {len(replacements)}/{len(removed)} replacements")

    replacement_map: dict[str, GeometryAssessment] = {}
    replacement_iter = iter(replacements)
    new_objects = []
    for old in original:
        if old["object_id"] not in removed_ids:
            new_objects.append(old)
            continue
        new = next(replacement_iter)
        replacement_map[old["object_id"]] = new
        new_objects.append(
            {
                **new.__dict__,
                "category": semantic_category(new),
                "initial_pose_stability": stability_rows[new.object_id],
            }
        )
    output = dict(payload)
    output["name"] = "Underactuated-hand Top100 with A/B failures replaced"
    output["selection_method"] = "preserve_92_replace_failure_classes_A_B"
    output["replacements"] = {
        old: new.object_id for old, new in replacement_map.items()
    }
    output["objects"] = new_objects
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {len(new_objects)} objects -> {args.output}")
    for old, new in replacement_map.items():
        print(f"  {old} -> {new.object_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
