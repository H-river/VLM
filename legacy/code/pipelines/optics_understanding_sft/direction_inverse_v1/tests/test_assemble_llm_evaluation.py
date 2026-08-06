import pytest

from optics_understanding_sft.direction_inverse_v1.assemble_llm_evaluation import assemble


def test_assemble_requires_named_splits_and_preserves_selection() -> None:
    panel = {"selected_checkpoint": {"checkpoint": 200}, "panel": [{"checkpoint": 200}]}
    output = assemble(panel, {"split": "eval_iid"}, {"split": "eval_ood"})
    assert output["selected_checkpoint"]["checkpoint"] == 200
    assert output["eval_iid"]["split"] == "eval_iid"

    with pytest.raises(ValueError, match="IID"):
        assemble(panel, {"split": "val"}, {"split": "eval_ood"})
