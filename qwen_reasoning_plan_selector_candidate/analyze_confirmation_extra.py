#!/usr/bin/env python3
"""Aggregate final confirmation selectors over 13 physical CEM seeds."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .protocol import PLAN_NAMES


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    evaluation = json.loads((ARTIFACT / "confirmation_evaluation.json").read_text())
    groups = [row for row in read_jsonl(ARTIFACT / "split_manifests/groups.jsonl") if row["split"] == "candidate_confirmation"]
    ids = [row["group_id"] for row in groups]
    original = [row for row in read_jsonl(ARTIFACT / "rollout_results.jsonl") if row["group_id"] in ids]
    extra = read_jsonl(ARTIFACT / "confirmation_extra_10seed_results.jsonl")
    rows = original + extra
    by = defaultdict(list)
    for row in rows:
        by[(row["group_id"], row["plan_name"])].append(row)
    if any(len(by[(gid, plan)]) != 13 for gid in ids for plan in PLAN_NAMES):
        raise RuntimeError("every confirmation group-plan requires 13 seeds")
    outcomes = {}
    for key, values in by.items():
        outcomes[key] = {"strict_success_rate": float(np.mean([row["strict_all_five_success"] for row in values])), "terminal_normalized_error_mean": float(np.mean([row["terminal_normalized_error"] for row in values])), "steps_mean": float(np.mean([row["steps"] for row in values])), "boundary_risk_cost_mean": float(np.mean([row["boundary_risk_cost"] for row in values]))}
    def score(selection: dict[str, str]) -> dict:
        values = [outcomes[(gid, selection[gid])] for gid in ids]
        counts = Counter(selection[gid] for gid in ids)
        probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(ids)
        return {"groups": len(ids), "physical_episodes": len(ids) * 13, "strict_success_rate": float(np.mean([row["strict_success_rate"] for row in values])), "terminal_normalized_error_mean": float(np.mean([row["terminal_normalized_error_mean"] for row in values])), "steps_mean": float(np.mean([row["steps_mean"] for row in values])), "boundary_risk_cost_mean": float(np.mean([row["boundary_risk_cost_mean"] for row in values])), "plan_distribution": dict(sorted(counts.items())), "plan_selection_entropy_bits": float(-np.sum(probabilities * np.log2(probabilities)))}
    selectors = {name: score(selection) for name, selection in evaluation["selector_selections"].items()}
    oracle_13seed_selection = {
        gid: sorted(
            PLAN_NAMES,
            key=lambda plan: (
                -outcomes[(gid, plan)]["strict_success_rate"],
                outcomes[(gid, plan)]["terminal_normalized_error_mean"],
                outcomes[(gid, plan)]["steps_mean"],
                outcomes[(gid, plan)]["boundary_risk_cost_mean"],
                PLAN_NAMES.index(plan),
            ),
        )[0]
        for gid in ids
    }
    selectors["oracle_13seed"] = score(oracle_13seed_selection)
    for name, metrics in selectors.items():
        if name in {"oracle", "oracle_13seed"}:
            continue
        metrics["oracle_strict_success_regret"] = selectors["oracle_13seed"]["strict_success_rate"] - metrics["strict_success_rate"]
        metrics["oracle_terminal_error_regret"] = metrics["terminal_normalized_error_mean"] - selectors["oracle_13seed"]["terminal_normalized_error_mean"]
    qwen = selectors["qwen_borda_3seed"]
    qwen["beats_best_fixed_strict_success"] = qwen["strict_success_rate"] > selectors["best_fixed"]["strict_success_rate"]
    qwen["beats_visual_lookup_strict_success"] = qwen["strict_success_rate"] > selectors["visual_diagnosis_lookup"]["strict_success_rate"]
    qwen["beats_numerical_only_strict_success"] = qwen["strict_success_rate"] > selectors["numerical_only_mlp"]["strict_success_rate"]
    per_visual_family = {}
    for family in sorted({str(group["visual_family"]) for group in groups}):
        family_ids = [group["group_id"] for group in groups if group["visual_family"] == family]
        family_outcomes = {key: value for key, value in outcomes.items() if key[0] in family_ids}
        def family_score(selection: dict[str, str]) -> dict:
            values = [family_outcomes[(gid, selection[gid])] for gid in family_ids]
            return {"groups": len(family_ids), "strict_success_rate": float(np.mean([row["strict_success_rate"] for row in values])), "terminal_normalized_error_mean": float(np.mean([row["terminal_normalized_error_mean"] for row in values])), "steps_mean": float(np.mean([row["steps_mean"] for row in values])), "boundary_risk_cost_mean": float(np.mean([row["boundary_risk_cost_mean"] for row in values]))}
        selections = {**evaluation["selector_selections"], "oracle_13seed": oracle_13seed_selection}
        per_visual_family[family] = {name: family_score(selection) for name, selection in selections.items()}
    output = {"candidate_only": True, "confirmation_groups": len(ids), "seeds_per_group_plan": 13, "all_plans_physically_executed_episodes": len(ids) * len(PLAN_NAMES) * 13, "selectors": selectors, "per_visual_family": per_visual_family, "oracle_13seed_selection": oracle_13seed_selection, "frozen_or_protected_enabled": False}
    (ARTIFACT / "confirmation_13seed_evaluation.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
