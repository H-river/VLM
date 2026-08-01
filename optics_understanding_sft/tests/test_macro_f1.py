from __future__ import annotations

from optics_understanding_sft.evaluate_intermediate_evidence_v6 import macro_f1
from optics_understanding_sft.evaluate_visual_evidence_v10 import macro_f1 as visual_macro_f1


def test_macro_f1_penalizes_predicted_only_class() -> None:
    score = macro_f1(["no_change", "increase"], ["decrease", "increase"])
    assert score == (0.0 + 1.0 + 0.0) / 3.0
    assert visual_macro_f1(["no_change", "increase"], ["decrease", "increase"]) == score
