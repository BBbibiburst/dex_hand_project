"""Rank a geometry-only object pool for grasp generation and dynamics validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import trimesh

from source.envs.manipulation.object_catalog import (
    MANIFEST_PATH,
    object_records,
    record_scale_to_meters,
)
from source.grasping.catalog import resolve_object_mesh
from source.grasping.affordance import geometry_affordance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        default=0.70,
        help="Minimum geometry UAS prior for inclusion. Default: 0.70.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/underactuated_geometry_candidates.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "outputs/underactuated_geometry_report.json",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if not 0.0 <= args.minimum_prior <= 1.0:
        parser.error("--minimum-prior must lie in [0, 1]")
    assessments: list[GeometryAssessment] = []
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
        print(f"[{index:03d}/{len(object_records()):03d}] {object_id}: prior={prior:.3f} {item.reasons}")
    ranked = sorted(
        (
            item
            for item in assessments
            if item.eligible
            and float(item.geometry["geometry_prior"]) >= args.minimum_prior
        ),
        key=lambda x: (-float(x.geometry["geometry_prior"]), x.object_id),
    )
    selected = ranked[: args.count]
    report_payload = {
        "schema_version": 1,
        "source_manifest": str(MANIFEST_PATH),
        "requested_count": args.count,
        "minimum_geometry_prior": args.minimum_prior,
        "eligible_count": len(ranked),
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
        "selection_method": "underactuated_affordance_geometry_prior_v1",
        "minimum_geometry_prior": args.minimum_prior,
        "requires_physics_validation": True,
        "objects": [asdict(item) for item in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[done] selected {len(selected)} objects -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
