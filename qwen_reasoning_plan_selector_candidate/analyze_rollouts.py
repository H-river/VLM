#!/usr/bin/env python3
"""Rank real plan outcomes, audit collapse/value, and enforce fixed-gain equality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from continuous_control_v12.contracts import ACTION_FIELDS, OUTPUT_FIELDS, stable_seed

from .core import FIXED_CONTROLLER_CONFIG, PLAN_NAMES, build_plan_bank, canonical_hash


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPOSITORY_ROOT / "artifacts/qwen_reasoning_plan_selector"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rank_key(outcome: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -float(outcome["strict_success_rate"]),
        float(outcome["terminal_normalized_error_mean"]),
        float(outcome["steps_mean"]),
        float(outcome["boundary_risk_cost_mean"]),
        PLAN_NAMES.index(str(outcome["plan_name"])),
    )


def _seed_rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -float(bool(row["strict_all_five_success"])),
        float(row["terminal_normalized_error"]),
        float(row["steps"]),
        float(row["boundary_risk_cost"]),
        PLAN_NAMES.index(str(row["plan_name"])),
    )


def _actions_signature(row: Mapping[str, Any]) -> str:
    actions = [
        [round(float(action[field]), 12) for field in ACTION_FIELDS]
        for action in row["complete_actuator_actions"]
    ]
    return canonical_hash(actions)


def _group_outcomes(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_plan: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_plan[str(row["plan_name"])].append(row)
    if set(by_plan) != set(PLAN_NAMES) or any(len(values) != 3 for values in by_plan.values()):
        raise RuntimeError("every formal group requires five plans x three seeds")
    outcomes = []
    for plan_name in PLAN_NAMES:
        values = sorted(by_plan[plan_name], key=lambda row: int(row["seed_index"]))
        outcomes.append(
            {
                "plan_name": plan_name,
                "strict_success_rate": float(np.mean([bool(row["strict_all_five_success"]) for row in values])),
                "terminal_normalized_error_mean": float(np.mean([row["terminal_normalized_error"] for row in values])),
                "terminal_normalized_error_std": float(np.std([row["terminal_normalized_error"] for row in values])),
                "steps_mean": float(np.mean([row["steps"] for row in values])),
                "boundary_risk_cost_mean": float(np.mean([row["boundary_risk_cost"] for row in values])),
                "boundary_violation_count": int(sum(int(row["boundary_violation_count"]) for row in values)),
                "seed_outcomes": [
                    {
                        "seed_index": int(row["seed_index"]),
                        "cem_seed": int(row["cem_seed"]),
                        "strict_all_five_success": bool(row["strict_all_five_success"]),
                        "terminal_normalized_error": float(row["terminal_normalized_error"]),
                        "steps": int(row["steps"]),
                        "boundary_risk_cost": float(row["boundary_risk_cost"]),
                        "action_signature": _actions_signature(row),
                    }
                    for row in values
                ],
            }
        )
    outcomes.sort(key=_rank_key)
    top, second = outcomes[:2]
    success_margin = float(top["strict_success_rate"] - second["strict_success_rate"])
    terminal_margin = float(second["terminal_normalized_error_mean"] - top["terminal_normalized_error_mean"])
    per_seed_top_sets = []
    for seed_index in range(3):
        seed_rows = [
            next(row for row in by_plan[plan_name] if int(row["seed_index"]) == seed_index)
            for plan_name in PLAN_NAMES
        ]
        seed_rows.sort(key=_seed_rank_key)
        best = seed_rows[0]
        acceptable = []
        for candidate in seed_rows:
            same_success = bool(candidate["strict_all_five_success"]) == bool(best["strict_all_five_success"])
            close_error = float(candidate["terminal_normalized_error"]) - float(best["terminal_normalized_error"]) <= 0.05
            if same_success and close_error:
                acceptable.append(str(candidate["plan_name"]))
        per_seed_top_sets.append(acceptable)
    winner_stable = all(str(top["plan_name"]) in values for values in per_seed_top_sets)
    close_top_two = success_margin == 0.0 and terminal_margin <= 0.05
    ambiguous_reasons = []
    if close_top_two:
        ambiguous_reasons.append("top_second_terminal_margin_le_0.05_at_equal_success")
    if not winner_stable:
        ambiguous_reasons.append("winner_not_in_all_per_seed_top_sets")
    decisive = not ambiguous_reasons
    ranking = {
        "selected_oracle_plan": str(top["plan_name"]),
        "decisive": decisive,
        "ambiguous_reasons": ambiguous_reasons,
        "success_margin": success_margin,
        "terminal_error_margin": terminal_margin,
        "per_seed_top_sets": per_seed_top_sets,
        "winner_stable": winner_stable,
    }
    return outcomes, ranking


def _feature_vector(group: Mapping[str, Any], representative: Mapping[str, Any]) -> np.ndarray:
    trace = representative["trace"]
    if not trace:
        observed = np.asarray([group["initial_clean_metrics"][field] for field in OUTPUT_FIELDS])
    else:
        observed = np.asarray([trace[0]["measured_metrics_before"][field] for field in OUTPUT_FIELDS])
    target = np.asarray([group["target_metrics"][field] for field in OUTPUT_FIELDS])
    position = np.asarray([group["initial_positions_mm"][field] for field in ("lens_x_mm", "lens_y_mm", "camera_x_mm", "camera_y_mm")])
    bounds = group["bounds"]
    lower = position - np.asarray(bounds["position_low"])
    upper = np.asarray(bounds["position_high"]) - position
    probe = representative["initial_probe_audit"]
    sensitivity = np.asarray([probe["sensitivity"][name] for name in ("lens_x", "lens_y", "camera_x", "camera_y")])
    uncertainty = np.asarray([row["mean_uncertainty"] for row in probe["rows"]])
    return np.concatenate([observed, target, position, lower, upper, sensitivity, uncertainty]).astype(np.float64)


def _evaluate_selection(
    selections: Mapping[str, str],
    outcome_by_group_plan: Mapping[tuple[str, str], Mapping[str, Any]],
    group_ids: Sequence[str],
) -> dict[str, Any]:
    values = [outcome_by_group_plan[(group_id, selections[group_id])] for group_id in group_ids]
    return {
        "groups": len(group_ids),
        "strict_success_rate": float(np.mean([row["strict_success_rate"] for row in values])),
        "terminal_normalized_error_mean": float(np.mean([row["terminal_normalized_error_mean"] for row in values])),
        "steps_mean": float(np.mean([row["steps_mean"] for row in values])),
        "boundary_risk_cost_mean": float(np.mean([row["boundary_risk_cost_mean"] for row in values])),
        "plan_distribution": dict(sorted(Counter(selections[group_id] for group_id in group_ids).items())),
    }


def _entropy(counter: Mapping[str, int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probabilities = np.asarray([value / total for value in counter.values() if value > 0])
    return float(-(probabilities * np.log2(probabilities)).sum())


def analyze(args: argparse.Namespace) -> None:
    artifact = args.artifact.resolve()
    rows = read_jsonl(artifact / "rollout_results.jsonl")
    groups = read_jsonl(artifact / "split_manifests/groups.jsonl")
    group_by_id = {str(row["group_id"]): row for row in groups}
    ranked_group_ids = {
        str(row["group_id"])
        for row in groups
        if args.include_confirmation or row["split"] != "candidate_confirmation"
    }
    expected = len(groups) * len(PLAN_NAMES) * 3
    if len(rows) != expected:
        raise RuntimeError(f"formal rollout incomplete: {len(rows)}/{expected}")
    if any(bool(row.get("formal_frozen_evaluation_enabled", True)) for row in rows):
        raise RuntimeError("formal rollout contains frozen/protected access")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_id"])].append(row)
    outcomes_rows = []
    rankings: dict[str, dict[str, Any]] = {}
    outcome_by_group_plan: dict[tuple[str, str], dict[str, Any]] = {}
    representative_by_group: dict[str, dict[str, Any]] = {}
    for group_id in sorted(ranked_group_ids):
        values = by_group[group_id]
        outcomes, ranking = _group_outcomes(values)
        group = group_by_id[group_id]
        rankings[group_id] = ranking
        representative_by_group[group_id] = next(
            row for row in values if row["plan_name"] == "direct_all_five" and int(row["seed_index"]) == 0
        )
        for rank, outcome in enumerate(outcomes, start=1):
            enriched = {
                "group_id": group_id,
                "base_state_id": group["base_state_id"],
                "split": group["split"],
                "visual_family": group["visual_family"],
                "intervention_type": group["intervention_type"],
                "rank": rank,
                "oracle_selected": rank == 1,
                "decisive_group": bool(ranking["decisive"]),
                **outcome,
            }
            outcomes_rows.append(enriched)
            outcome_by_group_plan[(group_id, str(outcome["plan_name"]))] = enriched
    csv_path = artifact / "plan_outcomes.csv"
    fieldnames = [
        "group_id", "base_state_id", "split", "visual_family", "intervention_type",
        "plan_name", "rank", "oracle_selected", "decisive_group", "strict_success_rate",
        "terminal_normalized_error_mean", "terminal_normalized_error_std", "steps_mean",
        "boundary_risk_cost_mean", "boundary_violation_count", "seed_outcomes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in outcomes_rows:
            serialized = dict(row)
            serialized["seed_outcomes"] = json.dumps(row["seed_outcomes"], sort_keys=True, separators=(",", ":"))
            writer.writerow(serialized)

    # Equality is checked per matched group, not merely globally.
    equality_errors = []
    plan_config_hashes: dict[str, set[str]] = defaultdict(set)
    group_action_bound_hashes: dict[str, set[str]] = defaultdict(set)
    group_seed_maps: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        plan_config_hashes[str(row["plan_name"])].add(str(row["fixed_controller_config_hash"]))
        group_action_bound_hashes[str(row["group_id"])].add(canonical_hash(row["action_bounds"]))
        group_seed_maps[str(row["group_id"])][int(row["seed_index"])].add(int(row["cem_seed"]))
        if row["fixed_controller_config"] != FIXED_CONTROLLER_CONFIG:
            equality_errors.append(f"{row['group_id']}::{row['plan_name']} config mismatch")
    for group_id, hashes in group_action_bound_hashes.items():
        if len(hashes) != 1:
            equality_errors.append(f"{group_id} action bounds differ across plans")
    for group_id, seed_map in group_seed_maps.items():
        for seed_index, seeds in seed_map.items():
            if len(seeds) != 1:
                equality_errors.append(f"{group_id} seed {seed_index} differs across plans")
    expected_config_hash = canonical_hash(FIXED_CONTROLLER_CONFIG)
    if any(values != {expected_config_hash} for values in plan_config_hashes.values()):
        equality_errors.append("per-plan fixed controller config hash mismatch")
    gain_audit = {
        "passed": not equality_errors,
        "errors": equality_errors,
        "episodes": len(rows),
        "fixed_controller_config": FIXED_CONTROLLER_CONFIG,
        "expected_hash": expected_config_hash,
        "hashes_by_plan": {name: sorted(plan_config_hashes[name]) for name in PLAN_NAMES},
        "per_group_action_bound_hash_count_max": max(map(len, group_action_bound_hashes.values())),
        "matched_seed_set_count_max": max(len(seeds) for values in group_seed_maps.values() for seeds in values.values()),
        "checkpoint_sha256": json.loads((artifact / "rollout_config.json").read_text())["source_files"]["learned_h1_checkpoint_sha256"],
        "same_gain_bound_population_iterations_horizon_steps_tolerance_checkpoint_simulator": not equality_errors,
    }
    atomic_json(artifact / "gain_equality_audit.json", gain_audit)
    if equality_errors:
        raise RuntimeError("gain/budget equality audit failed")

    nonconfirmation_ids = [
        row["group_id"] for row in groups if row["split"] in {"candidate_train", "candidate_dev"}
    ]
    decisive_ids = [group_id for group_id in nonconfirmation_ids if rankings[group_id]["decisive"]]
    ambiguous_ids = [group_id for group_id in nonconfirmation_ids if not rankings[group_id]["decisive"]]
    winner_counts = Counter(rankings[group_id]["selected_oracle_plan"] for group_id in decisive_ids)
    per_family = {}
    for family in sorted({str(group_by_id[group_id]["visual_family"]) for group_id in nonconfirmation_ids}):
        family_ids = [group_id for group_id in decisive_ids if group_by_id[group_id]["visual_family"] == family]
        counts = Counter(rankings[group_id]["selected_oracle_plan"] for group_id in family_ids)
        per_family[family] = {
            "decisive_groups": len(family_ids),
            "winner_distribution": dict(sorted(counts.items())),
            "winner_plan_count": len(counts),
        }

    oracle_selection = {group_id: rankings[group_id]["selected_oracle_plan"] for group_id in nonconfirmation_ids}
    fixed_results = {}
    for plan_name in PLAN_NAMES:
        selection = {group_id: plan_name for group_id in nonconfirmation_ids}
        fixed_results[plan_name] = _evaluate_selection(selection, outcome_by_group_plan, nonconfirmation_ids)
    best_fixed_name = sorted(PLAN_NAMES, key=lambda name: (
        -fixed_results[name]["strict_success_rate"],
        fixed_results[name]["terminal_normalized_error_mean"],
        fixed_results[name]["steps_mean"],
        fixed_results[name]["boundary_risk_cost_mean"],
        PLAN_NAMES.index(name),
    ))[0]
    oracle_metrics = _evaluate_selection(oracle_selection, outcome_by_group_plan, nonconfirmation_ids)

    # Leave-one-base-out selectors avoid intervention siblings leaking into training.
    features = {
        group_id: _feature_vector(group_by_id[group_id], representative_by_group[group_id])
        for group_id in nonconfirmation_ids
    }
    visual_selection = {}
    numeric_selection = {}
    frequency_selection = {}
    all_bases = sorted({str(group_by_id[group_id]["base_state_id"]) for group_id in nonconfirmation_ids})
    for held_base in all_bases:
        held_ids = [group_id for group_id in nonconfirmation_ids if group_by_id[group_id]["base_state_id"] == held_base]
        train_ids = [group_id for group_id in decisive_ids if group_by_id[group_id]["base_state_id"] != held_base]
        labels = [rankings[group_id]["selected_oracle_plan"] for group_id in train_ids]
        global_counts = Counter(labels)
        fallback = global_counts.most_common(1)[0][0] if global_counts else best_fixed_name
        for group_id in held_ids:
            family = str(group_by_id[group_id]["visual_family"])
            family_labels = [
                rankings[train_id]["selected_oracle_plan"]
                for train_id in train_ids
                if group_by_id[train_id]["visual_family"] == family
            ]
            visual_selection[group_id] = Counter(family_labels).most_common(1)[0][0] if family_labels else fallback
            if global_counts:
                ordered = sorted(global_counts)
                probabilities = np.asarray([global_counts[name] for name in ordered], dtype=np.float64)
                probabilities /= probabilities.sum()
                rng = np.random.default_rng(stable_seed(2026080413, group_id) % (2**32))
                frequency_selection[group_id] = str(rng.choice(ordered, p=probabilities))
            else:
                frequency_selection[group_id] = fallback
        unique_labels = sorted(set(labels))
        if len(unique_labels) >= 2 and len(train_ids) >= 8:
            model = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=min(16, len(train_ids)),
                    learning_rate_init=1e-3,
                    max_iter=500,
                    random_state=2026080412,
                    early_stopping=False,
                ),
            )
            model.fit(np.stack([features[group_id] for group_id in train_ids]), np.asarray(labels))
            predictions = model.predict(np.stack([features[group_id] for group_id in held_ids]))
            numeric_selection.update(zip(held_ids, map(str, predictions), strict=True))
        else:
            numeric_selection.update({group_id: fallback for group_id in held_ids})
    selector_metrics = {
        "oracle": oracle_metrics,
        "best_fixed": {"plan": best_fixed_name, **fixed_results[best_fixed_name]},
        "direct_all_five": fixed_results["direct_all_five"],
        "frequency_matched_lobo": _evaluate_selection(frequency_selection, outcome_by_group_plan, nonconfirmation_ids),
        "visual_diagnosis_lookup_lobo": _evaluate_selection(visual_selection, outcome_by_group_plan, nonconfirmation_ids),
        "numerical_only_mlp_lobo": _evaluate_selection(numeric_selection, outcome_by_group_plan, nonconfirmation_ids),
        "all_fixed_plans": fixed_results,
    }
    for baseline in ("best_fixed", "direct_all_five", "visual_diagnosis_lookup_lobo", "numerical_only_mlp_lobo"):
        base = selector_metrics[baseline]
        base["oracle_strict_success_uplift"] = oracle_metrics["strict_success_rate"] - base["strict_success_rate"]
        base["oracle_terminal_error_reduction"] = base["terminal_normalized_error_mean"] - oracle_metrics["terminal_normalized_error_mean"]

    intervention_rows = read_jsonl(artifact / "matched_interventions.jsonl")
    intervention_summary = {}
    for intervention_type in sorted({row["type"] for row in intervention_rows}):
        values = [row for row in intervention_rows if row["type"] == intervention_type and row["left_group_id"] in rankings and row["right_group_id"] in rankings]
        valid = [
            row for row in values
            if rankings[row["left_group_id"]]["decisive"] and rankings[row["right_group_id"]]["decisive"]
        ]
        flips = [
            rankings[row["left_group_id"]]["selected_oracle_plan"]
            != rankings[row["right_group_id"]]["selected_oracle_plan"]
            for row in valid
        ]
        intervention_summary[intervention_type] = {
            "pairs": len(values),
            "effective_decisive_pairs": len(valid),
            "oracle_plan_flips": int(sum(flips)),
            "oracle_plan_flip_rate": None if not flips else float(np.mean(flips)),
        }
    all_effective = sum(value["effective_decisive_pairs"] for value in intervention_summary.values())
    all_flips = sum(value["oracle_plan_flips"] for value in intervention_summary.values())

    # Empirical execution contracts.
    pair_differences = defaultdict(list)
    active_effect = []
    measurement_effect = []
    for group_id, group_rows in by_group.items():
        if group_id not in rankings:
            continue
        by_plan_seed = {(row["plan_name"], int(row["seed_index"])): row for row in group_rows}
        for seed_index in range(3):
            direct = by_plan_seed[("direct_all_five", seed_index)]
            for plan_name in PLAN_NAMES[1:]:
                other = by_plan_seed[(plan_name, seed_index)]
                pair_differences[plan_name].append(_actions_signature(direct) != _actions_signature(other))
            boundary = by_plan_seed[("boundary_safe_then_full", seed_index)]
            active_effect.append(
                bool(boundary["active_actuator_trajectory"])
                and boundary["active_actuator_trajectory"][0] != list(("lens_x", "lens_y", "camera_x", "camera_y"))
            )
            if group_by_id[group_id]["visual_family"] == "width_relative_reflection":
                primary = by_plan_seed[("primary_spot_then_full", seed_index)]
                measurement_effect.append(
                    direct["terminal_measured_metrics"] != primary["terminal_measured_metrics"]
                    or _actions_signature(direct) != _actions_signature(primary)
                )
    execution_contract = {
        "fixed_gain_config_passed": gain_audit["passed"],
        "trajectory_difference_rate_vs_direct": {
            name: float(np.mean(values)) for name, values in sorted(pair_differences.items())
        },
        "boundary_first_phase_active_subset_rate": float(np.mean(active_effect)),
        "reflection_primary_measurement_or_action_effect_rate": float(np.mean(measurement_effect)),
        "phase_transition_counts": {
            name: int(sum(len(row["phase_transitions"]) > 1 for row in rows if row["plan_name"] == name and row["group_id"] in rankings))
            for name in PLAN_NAMES
        },
        "all_primitives_executed": bool(
            all(np.mean(values) > 0.0 for values in pair_differences.values())
            and np.mean(active_effect) == 1.0
            and np.mean(measurement_effect) > 0.0
        ),
    }
    plan_contracts = json.loads((artifact / "plan_contracts.json").read_text())
    plan_contracts["formal_execution_evidence"] = execution_contract
    plan_contracts["formal_rollout_sha256"] = sha256_path(artifact / "rollout_results.jsonl")
    atomic_json(artifact / "plan_contracts.json", plan_contracts)

    decisive_total = len(decisive_ids)
    winner_frequencies = {
        name: (winner_counts.get(name, 0) / decisive_total if decisive_total else 0.0)
        for name in PLAN_NAMES
    }
    three_plan_gate = sum(value >= 0.10 for value in winner_frequencies.values()) >= 3
    within_family_gate = all(
        value["winner_plan_count"] >= 2 for value in per_family.values() if value["decisive_groups"] > 0
    )
    intervention_gate = all_effective > 0 and all_flips / all_effective >= 0.30
    fixed_advantage_gate = (
        oracle_metrics["strict_success_rate"] > selector_metrics["best_fixed"]["strict_success_rate"]
        or oracle_metrics["terminal_normalized_error_mean"] + 0.05 < selector_metrics["best_fixed"]["terminal_normalized_error_mean"]
    )
    lookup_residual_gate = (
        selector_metrics["visual_diagnosis_lookup_lobo"]["strict_success_rate"] < oracle_metrics["strict_success_rate"]
        or selector_metrics["visual_diagnosis_lookup_lobo"]["terminal_normalized_error_mean"] > oracle_metrics["terminal_normalized_error_mean"] + 0.05
    )
    numeric_residual_gate = (
        selector_metrics["numerical_only_mlp_lobo"]["strict_success_rate"] < oracle_metrics["strict_success_rate"]
        or selector_metrics["numerical_only_mlp_lobo"]["terminal_normalized_error_mean"] > oracle_metrics["terminal_normalized_error_mean"] + 0.05
    )
    gates = {
        "at_least_three_plans_win_ge_10pct_decisive": three_plan_gate,
        "each_main_visual_family_has_two_winner_plans": within_family_gate,
        "matched_intervention_flip_rate_ge_30pct": intervention_gate,
        "oracle_advantage_over_best_fixed": fixed_advantage_gate,
        "visual_lookup_leaves_residual_regret": lookup_residual_gate,
        "numerical_only_leaves_residual_regret": numeric_residual_gate,
        "gain_budget_equality": gain_audit["passed"],
        "all_plan_primitives_executed": execution_contract["all_primitives_executed"],
        "enough_decisive_samples": decisive_total >= 24,
    }
    gate_passed = all(gates.values())
    audit = {
        "candidate_only": True,
        "confirmation_excluded_from_gate_and_baseline_fitting": True,
        "confirmation_outcomes_opened": bool(args.include_confirmation),
        "groups_in_gate": len(nonconfirmation_ids),
        "decisive_groups": decisive_total,
        "ambiguous_groups": len(ambiguous_ids),
        "ambiguous_group_ids": ambiguous_ids,
        "winner_counts": {name: winner_counts.get(name, 0) for name in PLAN_NAMES},
        "winner_frequencies": winner_frequencies,
        "plan_label_entropy_bits": _entropy(winner_counts),
        "per_visual_family": per_family,
        "selector_closed_loop_value": selector_metrics,
        "matched_interventions": intervention_summary,
        "effective_matched_intervention_flip_rate": None if all_effective == 0 else all_flips / all_effective,
        "execution_contract": execution_contract,
        "gates": gates,
        "gate_passed": gate_passed,
        "gate_decision": "GO_TO_SFT" if gate_passed else "REVISE_PLAN_BANK_ONCE_BEFORE_NO_GO",
        "group_rankings": rankings,
        "hard_labels_come_only_from_real_rollout_outcomes": True,
    }
    audit_name = "final_outcome_audit.json" if args.include_confirmation else "anti_collapse_audit.json"
    atomic_json(artifact / audit_name, audit)
    print(json.dumps({
        "gate_passed": gate_passed,
        "decisive_groups": decisive_total,
        "ambiguous_groups": len(ambiguous_ids),
        "winner_counts": audit["winner_counts"],
        "best_fixed": best_fixed_name,
        "oracle": oracle_metrics,
    }, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--include-confirmation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
