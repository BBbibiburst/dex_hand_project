from tools.rank_underactuated_candidates import (
    GeometryAssessment,
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
