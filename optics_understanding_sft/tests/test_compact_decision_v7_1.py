from __future__ import annotations

from pathlib import Path

from optics_understanding_sft.core import load_yaml, read_jsonl
from optics_understanding_sft.evaluate_compact_decision_v7_1 import evaluate


ROOT = Path(__file__).resolve().parents[1]


def test_compact_v7_1_oracle_passes_every_gate() -> None:
    records = read_jsonl(ROOT / "data" / "compact_decision_v7_1" / "canonical" / "dev.jsonl")
    predictions = [
        {"example_id": record["example_id"], "parsed_json": record["target"]}
        for record in records
    ]
    details, summary, gates = evaluate(
        load_yaml(ROOT / "configs" / "compact_decision_v7_1.yaml"), records, predictions
    )

    assert gates["passed"]
    assert summary["compact_decision_exact_match"] == 1.0
    assert summary["end_to_end_exact_match"] == 1.0
    assert all(row["target_exact"] for row in details)
