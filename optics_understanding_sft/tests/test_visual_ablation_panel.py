from __future__ import annotations

from optics_understanding_sft.build_visual_ablation_panel import build_panels


def test_visual_ablation_removes_only_pixel_channel() -> None:
    target = {"status": "answerable", "answer": {"direction": "right"}}
    source = {
        "example_id": "visual-1",
        "group_id": "group-1",
        "modality": "visual",
        "prompt": (
            "Input data:\n{\n  \"observation_format\": \"before image followed by after image\"\n}"
            "\n\nReturn only strict JSON"
        ),
        "prompt_inputs": {
            "images": ["before.png", "after.png"],
            "observation_format": "before image followed by after image",
            "action": {"lens_x_delta_mm": 0.03},
        },
        "provenance": {"dataset_version": "test"},
        "split": "val",
        "target": target,
        "task_type": "causal_effects",
    }

    full, withheld = build_panels([source])

    assert full[0] == source
    assert withheld[0]["example_id"] == source["example_id"]
    assert withheld[0]["target"] == target
    assert withheld[0]["prompt_inputs"]["action"] == source["prompt_inputs"]["action"]
    assert withheld[0]["prompt_inputs"]["images"] == []
    assert withheld[0]["modality"] == "text"
    assert "Controlled ablation" in withheld[0]["prompt"]
    assert "before image followed by after image" not in withheld[0]["prompt"]

