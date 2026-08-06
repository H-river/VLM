from __future__ import annotations

from pathlib import Path

from optics_understanding_sft.core import read_jsonl
from optics_understanding_sft.counterfactual_tool import simulate_paired_counterfactual


ROOT = Path(__file__).resolve().parents[2]


def test_all_dev_counterfactual_targets_replay_exactly() -> None:
    masters = read_jsonl(ROOT / "optics_understanding_sft/data/dev_v2/master/cases.jsonl")
    count = 0
    for master in masters:
        for item in master["records"]:
            record = item["record"]
            if record["task_type"] != "counterfactual_reasoning":
                continue
            specs = {spec["name"]: spec for spec in item["private_eval"]["replay_specs"]}
            actual = simulate_paired_counterfactual(
                specs["scenario_a_before"]["setup_config"],
                specs["scenario_b_before"]["setup_config"],
                record["prompt_inputs"]["shared_action"],
                record["prompt_inputs"]["changed_parameter"],
            )
            # IEEE signed zero is semantically identical for this sensor delta contract.
            assert actual == record["target"]["answer"], record["example_id"]
            count += 1
    assert count == 60
