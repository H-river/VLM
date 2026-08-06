#!/usr/bin/env python3
"""Analyze final physical probe-coupling interventions and Qwen plan flips."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluate_selector import borda
from .protocol import PLAN_NAMES


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"
SEEDS = (2026080401, 2026080402, 2026080403)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    evaluation = json.loads((ARTIFACT / "confirmation_evaluation.json").read_text())
    robust = json.loads((ARTIFACT / "confirmation_13seed_evaluation.json").read_text())
    groups = read_jsonl(ARTIFACT / "probe_coupling_confirmation/groups.jsonl")
    pairs = read_jsonl(ARTIFACT / "probe_coupling_confirmation/pairs.jsonl")
    rows = read_jsonl(ARTIFACT / "probe_coupling_confirmation/rollout_results.jsonl")
    predictions = []
    for seed in SEEDS:
        seed_rows = read_jsonl(ARTIFACT / f"probe_coupling_confirmation/predictions_seed_{seed}.jsonl")
        if len(seed_rows) != len(groups) or any(int(row["seed"]) != seed for row in seed_rows):
            raise RuntimeError(f"incomplete probe-coupling prediction bundle for {seed}")
        predictions.extend(seed_rows)
    fallback = evaluation["best_fixed_chosen_without_confirmation"]
    qwen_right = borda(predictions, fallback)
    by = defaultdict(list)
    for row in rows:
        by[(row["group_id"], row["plan_name"])].append(row)
    if any(len(by[(group["group_id"], plan)]) != 13 for group in groups for plan in PLAN_NAMES):
        raise RuntimeError("every probe-coupling group-plan requires 13 physical CEM seeds")
    outcomes = {}
    for key, values in by.items():
        if len({row["fixed_controller_config_hash"] for row in values}) != 1:
            raise RuntimeError("controller config drift in probe-coupling outcomes")
        outcomes[key] = {"strict_success_rate": float(np.mean([row["strict_all_five_success"] for row in values])), "terminal_normalized_error_mean": float(np.mean([row["terminal_normalized_error"] for row in values])), "steps_mean": float(np.mean([row["steps"] for row in values])), "boundary_risk_cost_mean": float(np.mean([row["boundary_risk_cost"] for row in values]))}
    oracle_right = {group["group_id"]: sorted(PLAN_NAMES, key=lambda plan: (-outcomes[(group["group_id"], plan)]["strict_success_rate"], outcomes[(group["group_id"], plan)]["terminal_normalized_error_mean"], outcomes[(group["group_id"], plan)]["steps_mean"], outcomes[(group["group_id"], plan)]["boundary_risk_cost_mean"], PLAN_NAMES.index(plan)))[0] for group in groups}
    details = []
    for pair in pairs:
        left, right = pair["left_group_id"], pair["right_group_id"]
        qwen_pair = [evaluation["qwen_selection"][left], qwen_right[right]]
        oracle_pair = [robust["oracle_13seed_selection"][left], oracle_right[right]]
        details.append({**pair, "qwen_plan_pair": qwen_pair, "oracle_plan_pair": oracle_pair, "qwen_flipped": qwen_pair[0] != qwen_pair[1], "oracle_flipped": oracle_pair[0] != oracle_pair[1], "exact_plan_transition_agreement": qwen_pair == oracle_pair, "right_hand_oracle_plan_match": qwen_pair[1] == oracle_pair[1]})
    qwen_selected_values = [outcomes[(group["group_id"], qwen_right[group["group_id"]])] for group in groups]
    output = {
        "candidate_only": True,
        "physical_probe_coupling_intervention": True,
        "same_dominant_error_component_verified": all(row["dominant_error_component"] == row["right_dominant_error_component"] for row in pairs),
        "groups": len(groups),
        "pairs": len(details),
        "physical_episodes": len(rows),
        "prediction_valid_json_rate": float(np.mean([row["valid_json"] for row in predictions])),
        "qwen_plan_flip_rate": float(np.mean([row["qwen_flipped"] for row in details])),
        "oracle_plan_flip_rate": float(np.mean([row["oracle_flipped"] for row in details])),
        "flip_presence_agreement_rate": float(np.mean([row["qwen_flipped"] == row["oracle_flipped"] for row in details])),
        "exact_plan_transition_agreement_rate": float(np.mean([row["exact_plan_transition_agreement"] for row in details])),
        "right_hand_oracle_plan_accuracy": float(np.mean([row["right_hand_oracle_plan_match"] for row in details])),
        "qwen_right_closed_loop": {"strict_success_rate": float(np.mean([row["strict_success_rate"] for row in qwen_selected_values])), "terminal_normalized_error_mean": float(np.mean([row["terminal_normalized_error_mean"] for row in qwen_selected_values])), "steps_mean": float(np.mean([row["steps_mean"] for row in qwen_selected_values])), "boundary_risk_cost_mean": float(np.mean([row["boundary_risk_cost_mean"] for row in qwen_selected_values]))},
        "qwen_right_selection": qwen_right,
        "oracle_right_selection": oracle_right,
        "pair_details": details,
        "frozen_or_protected_enabled": False,
    }
    (ARTIFACT / "probe_coupling_intervention_audit.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
