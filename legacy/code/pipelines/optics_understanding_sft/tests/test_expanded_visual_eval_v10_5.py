from optics_understanding_sft.build_expanded_visual_eval_v10_5 import (
    missing_classes,
    selected_sources,
)


def test_selected_sources_ignores_modality_and_uses_explicit_replay_names():
    rows = [
        {
            "example_id": "case_diag",
            "group_id": "group",
            "modality": "text",
            "prompt_inputs": {},
            "task_type": "diagnosis",
        },
        {
            "example_id": "case_setup",
            "group_id": "group",
            "modality": "text",
            "prompt_inputs": {},
            "task_type": "setup_interpretation",
        },
    ]
    selected = selected_sources(rows)
    assert len(selected) == 1
    assert selected[0]["modality"] == "visual"
    assert selected[0]["prompt_inputs"]["images"] == [
        "images/synthetic/case_diag_baseline.png",
        "images/synthetic/case_diag_observed.png",
    ]
    assert rows[0]["modality"] == "text"


def test_missing_classes_requires_union_complete_coverage():
    complete = {
        "state": {
            "centroid_horizontal_region": {
                "left_of_center": 1,
                "centered": 1,
                "right_of_center": 1,
            },
            "centroid_vertical_region": {
                "above_center": 1,
                "centered": 1,
                "below_center": 1,
            },
            "sigma_x_band": {"narrow": 1, "medium": 1, "wide": 1},
            "sigma_y_band": {"narrow": 1, "medium": 1, "wide": 1},
        },
        "pair": {
            field: {"decrease": 1, "no_change": 1, "increase": 1}
            for field in ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity")
        },
    }
    assert missing_classes(complete) == {"state": {}, "pair": {}}
    del complete["pair"]["sigma_y"]["increase"]
    assert missing_classes(complete)["pair"]["sigma_y"] == ["increase"]
