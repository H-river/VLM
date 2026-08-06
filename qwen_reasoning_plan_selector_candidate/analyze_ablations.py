#!/usr/bin/env python3
"""Score Qwen ablations using already executed real confirmation outcomes."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .evaluate_selector import ARTIFACT, outcome_table, read_jsonl, selection_metrics


def main() -> None:
    evaluation = json.loads((ARTIFACT / "confirmation_evaluation.json").read_text())
    audit = json.loads((ARTIFACT / "final_outcome_audit.json").read_text())
    ids = sorted(evaluation["qwen_selection"])
    fallback = evaluation["best_fixed_chosen_without_confirmation"]
    rows = read_jsonl(ARTIFACT / "ablations/predictions_seed_2026080401.jsonl")
    by_kind = defaultdict(dict)
    for row in rows:
        by_kind[row["ablation"]][row["group_id"]] = row["parsed"]["selected_plan"] if row["valid_json"] else fallback
    outcomes = outcome_table(ARTIFACT / "plan_outcomes.csv")
    summaries = {kind: {"decision_flip_rate_vs_normal_3seed": float(np.mean([selection[gid] != evaluation["qwen_selection"][gid] for gid in ids])), **selection_metrics(selection, outcomes, ids)} for kind, selection in by_kind.items()}
    pairs = read_jsonl(ARTIFACT / "matched_interventions.jsonl")
    robust_path = ARTIFACT / "confirmation_13seed_evaluation.json"
    robust = json.loads(robust_path.read_text()) if robust_path.exists() else None
    oracle_selection = robust["oracle_13seed_selection"] if robust else {group_id: audit["group_rankings"][group_id]["selected_oracle_plan"] for group_id in ids}
    matched = {}
    pair_types = {
        "target_swap": "same_image_and_state_change_target",
        "boundary_swap": "same_image_and_target_change_legal_boundary",
        "image_family_swap": "same_physical_state_and_target_change_image_family",
    }
    for kind, recorded_type in pair_types.items():
        usable = [row for row in pairs if row["type"] == recorded_type and row["left_group_id"] in ids and row["right_group_id"] in ids]
        qwen_pairs = [(evaluation["qwen_selection"][row["left_group_id"]], evaluation["qwen_selection"][row["right_group_id"]]) for row in usable]
        oracle_pairs = [(oracle_selection[row["left_group_id"]], oracle_selection[row["right_group_id"]]) for row in usable]
        qwen_flip = [left != right for left, right in qwen_pairs]
        oracle_flip = [left != right for left, right in oracle_pairs]
        oracle_flip_indices = [index for index, value in enumerate(oracle_flip) if value]
        matched[kind] = {
            "pairs": len(usable),
            "oracle_source": "13seed_physical_plan_ranking" if robust else "3seed_physical_plan_ranking",
            "qwen_plan_flip_rate": None if not usable else float(np.mean(qwen_flip)),
            "oracle_plan_flip_rate": None if not usable else float(np.mean(oracle_flip)),
            "flip_presence_agreement_rate": None if not usable else float(np.mean(np.asarray(qwen_flip) == np.asarray(oracle_flip))),
            "exact_plan_transition_agreement_rate": None if not usable else float(np.mean([qwen_pair == oracle_pair for qwen_pair, oracle_pair in zip(qwen_pairs, oracle_pairs, strict=True)])),
            "right_hand_oracle_plan_accuracy": None if not usable else float(np.mean([qwen_pair[1] == oracle_pair[1] for qwen_pair, oracle_pair in zip(qwen_pairs, oracle_pairs, strict=True)])),
            "qwen_flip_recall_on_oracle_flips": None if not oracle_flip_indices else float(np.mean([qwen_flip[index] for index in oracle_flip_indices])),
            "pair_details": [{"pair_id": row["pair_id"], "qwen": list(qwen_pair), "oracle": list(oracle_pair)} for row, qwen_pair, oracle_pair in zip(usable, qwen_pairs, oracle_pairs, strict=True)],
        }
    output = {"candidate_only": True, "ablation_checkpoint_seed": 2026080401, "normal_selector_is_three_training_seed_borda": True, "image_and_probe_ablation": summaries, "matched_physical_interventions": matched, "probe_shuffle_is_input_corruption_not_a_physical_causal_intervention": True, "frozen_or_protected_enabled": False}
    (ARTIFACT / "intervention_ablation_audit.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
