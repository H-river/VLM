import pytest

from qwen_reasoning_plan_selector_candidate.protocol import PLAN_NAMES
from qwen_reasoning_plan_selector_candidate.selector_contract import (
    canonical_selector_text,
    parse_selector_output,
)
from qwen_reasoning_plan_selector_candidate.inference import parse_generated_text


def test_selector_contract_accepts_only_complete_executable_ranking() -> None:
    value = {"plan_ranking": list(PLAN_NAMES), "selected_plan": PLAN_NAMES[0]}
    text = canonical_selector_text(value)
    assert parse_selector_output(text).selected_plan == PLAN_NAMES[0]


@pytest.mark.parametrize(
    "text",
    [
        '{"selected_plan":"direct_all_five","plan_ranking":[]}',
        '{"plan_ranking":["direct_all_five"],"selected_plan":"direct_all_five"}',
        '{"plan_ranking":[],"selected_plan":"direct_all_five","gain":2}',
        '{"plan_ranking":[],"selected_plan":"unknown"}',
    ],
)
def test_selector_contract_rejects_partial_continuous_or_unknown_output(text: str) -> None:
    with pytest.raises(ValueError):
        parse_selector_output(text)


def test_inference_never_repairs_invalid_json() -> None:
    text = canonical_selector_text(
        {"plan_ranking": list(PLAN_NAMES), "selected_plan": PLAN_NAMES[0]}
    )
    assert parse_generated_text(text).valid_json
    invalid = parse_generated_text("prefix " + text)
    assert not invalid.valid_json
    assert invalid.parsed is None
