from __future__ import annotations

from pathlib import Path

from optics_understanding_sft.core import read_jsonl
from optics_understanding_sft.evaluate_intermediate_evidence_v6 import value_equal
from optics_understanding_sft.forward_prediction_tool import (
    FORWARD_SOURCE_MAP,
    materialize_forward_answer,
    run_mapped_forward_tool,
)


ROOT = Path(__file__).resolve().parents[1]


def test_forward_tool_replays_public_development_targets() -> None:
    rows = read_jsonl(ROOT / "data" / "dev_v2" / "canonical" / "val.jsonl")
    masters = read_jsonl(ROOT / "data" / "dev_v2" / "master" / "cases.jsonl")
    registry = {
        item["record"]["example_id"]: item["private_eval"]["replay_specs"][0]["setup_config"]
        for master in masters
        for item in master["records"]
        if item["record"]["task_type"] == "forward_prediction"
    }
    forward = [row for row in rows if row["task_type"] == "forward_prediction"]
    assert len(forward) == 60
    for row in forward:
        visible = {
            "setup_state_handle": row["example_id"],
            "action": row["prompt_inputs"]["action"],
        }
        result = run_mapped_forward_tool(visible, FORWARD_SOURCE_MAP, registry)
        assert value_equal(materialize_forward_answer(result), row["target"])
