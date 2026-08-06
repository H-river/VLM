from __future__ import annotations

from pathlib import Path

from optics_understanding_sft.core import load_yaml, read_jsonl
from optics_understanding_sft.evaluate_path_mapping_v7 import evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_v7_oracle_predictions_pass_all_development_gates() -> None:
    records = read_jsonl(ROOT / "data" / "path_mapping_v7" / "canonical" / "dev.jsonl")
    predictions = [
        {"example_id": record["example_id"], "parsed_json": record["target"]}
        for record in records
    ]
    details, summary, gates = evaluate(
        load_yaml(ROOT / "configs" / "path_mapping_v7.yaml"), records, predictions
    )

    assert gates["passed"]
    assert summary["end_to_end_exact_match"] == 1.0
    assert summary["mapped_tool_execution_match"] == 1.0
    assert all(detail["target_exact"] for detail in details)


def test_v7_evaluator_rejects_invented_mapping_shortcut() -> None:
    records = read_jsonl(ROOT / "data" / "path_mapping_v7" / "canonical" / "dev.jsonl")
    record = next(row for row in records if row["stage"] == "tool_source_mapping")
    prediction = {
        "example_id": record["example_id"],
        "parsed_json": {
            "tool_name": record["target"]["tool_name"],
            "source_map": {"action_trial_index": "candidate_action_trials.0"},
        },
    }

    details, summary, _ = evaluate(
        load_yaml(ROOT / "configs" / "path_mapping_v7.yaml"), [record], [prediction]
    )

    assert not details[0]["source_mapping_exact"]
    assert not details[0]["mapped_tool_execution_match"]
    assert summary["source_mapping_exact_match"] == 0.0
