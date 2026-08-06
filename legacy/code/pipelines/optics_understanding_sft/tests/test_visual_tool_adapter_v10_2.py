from __future__ import annotations

from pathlib import Path

import pytest

from optics_understanding_sft.visual_tool_adapter_v10_2 import (
    PAIR_TOOL,
    STATE_TOOL,
    normalize_source_map,
    run_mapped_visual_tool,
)


def test_visual_state_tool_rejects_invented_source_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact prompt-visible path"):
        run_mapped_visual_tool(
            STATE_TOOL,
            {"image_path": "visible.png"},
            {"image_path": "private.png"},
            image_root=tmp_path,
            state_calibration={},
            pair_calibration={},
        )


def test_pair_map_removes_only_exact_optional_difference_reference() -> None:
    visible = {
        "first_image_path": "first.png",
        "second_image_path": "second.png",
        "signed_difference_reference_path": "difference.png",
    }
    normalized = normalize_source_map(PAIR_TOOL, visible, dict(visible))
    assert normalized == {
        "first_image_path": "first.png",
        "second_image_path": "second.png",
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"signed_difference_reference_path": "invented.png"},
        {"unknown_path": "difference.png"},
    ],
)
def test_pair_map_rejects_invented_or_unknown_optional_extras(extra: dict) -> None:
    visible = {
        "first_image_path": "first.png",
        "second_image_path": "second.png",
        "signed_difference_reference_path": "difference.png",
    }
    with pytest.raises(ValueError):
        normalize_source_map(
            PAIR_TOOL,
            visible,
            {
                "first_image_path": "first.png",
                "second_image_path": "second.png",
                **extra,
            },
        )
