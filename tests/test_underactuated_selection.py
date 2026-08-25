from tools.rank_underactuated_candidates import (
    GeometryAssessment,
    benchmark_family,
    diverse_selection,
    semantic_category,
)
from tools.download_gso_objects import _selection


def item(object_id: str, dataset: str, prior: float, extents=(0.04, 0.05, 0.09)):
    return GeometryAssessment(
        object_id=object_id,
        dataset=dataset,
        geometry={
            "geometry_prior": prior,
            "extents_m": extents,
            "axis_ratio": max(extents) / min(extents),
            "sphericity": 0.8,
            "convexity": 0.9,
        },
        adaptive_contact=None,
        robustness=None,
        uas=None,
        eligible=True,
        reasons=[],
    )


def test_semantic_category_separates_common_grasp_families() -> None:
    assert semantic_category(item("ycb:005_tomato_soup_can", "ycb", 0.9)) == "cylinder"
    assert semantic_category(item("ycb:065-a_cups", "ycb", 0.9)) == "container"
    assert semantic_category(item("ycb:013_apple", "ycb", 0.9)) == "sphere"
    assert semantic_category(item("egad:A0", "egad", 0.9)) == "egad"


def test_benchmark_family_groups_semantic_clones_geometry_cannot_detect() -> None:
    assert benchmark_family(item("gso:QAbsorb_CoQ10", "gso", 0.9)) == "supplement"
    assert benchmark_family(item("ycb:013_apple", "ycb", 0.9)) == "fruit"
    assert benchmark_family(item("gso:Horse_Dreams_Pencil_Case", "gso", 0.9)) == "pencil_case"


def test_diverse_selection_honours_family_quota_before_score_fill() -> None:
    ranked = [
        item("gso:bottle_a", "gso", 0.99),
        item("gso:bottle_b", "gso", 0.98),
        item("gso:bottle_c", "gso", 0.97),
        item("ycb:013_apple", "ycb", 0.80, (0.06, 0.06, 0.065)),
        item("ycb:004_sugar_box", "ycb", 0.79, (0.04, 0.06, 0.10)),
    ]
    selected = diverse_selection(
        ranked,
        count=3,
        maximum_near_duplicates=1,
        quotas={"sphere": 1, "box": 1, "cylinder": 1},
    )
    assert {semantic_category(value) for value in selected} == {"sphere", "box", "cylinder"}


def test_diverse_selection_never_relaxes_duplicate_or_family_limits_to_fill() -> None:
    ranked = [
        item("gso:brand_a", "gso", 0.99),
        item("gso:brand_b", "gso", 0.98),
        item("gso:brand_c", "gso", 0.97),
    ]
    selected = diverse_selection(
        ranked,
        count=3,
        maximum_near_duplicates=1,
        quotas={"cylinder": 3},
    )
    assert len(selected) == 1


def test_diverse_selection_honours_physics_exclusions() -> None:
    ranked = [item("gso:bad", "gso", 0.99), item("gso:good", "gso", 0.90)]
    selected = diverse_selection(
        ranked,
        count=1,
        quotas={"cylinder": 1},
        excluded_ids={"gso:bad"},
    )
    assert [value.object_id for value in selected] == ["gso:good"]


def test_gso_downloader_accepts_shared_namespaced_json_selection(tmp_path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        '{"objects": ['
        '{"object_id": "ycb:013_apple"},'
        '{"object_id": "gso:First_Model"},'
        '{"object_id": "gso:Second_Model"}'
        ']}',
        encoding="utf-8",
    )

    assert _selection(path) == ["First_Model", "Second_Model"]
