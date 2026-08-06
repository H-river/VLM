#!/usr/bin/env python3
"""Evaluate learned and classical selectors on untouched candidate confirmation."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analyze_rollouts import _evaluate_selection, _feature_vector
from .export_sft import _visible_state
from .protocol import PLAN_NAMES, SYSTEM_PROMPT, USER_PREFIX


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/qwen_reasoning_plan_selector"
SEEDS = (2026080401, 2026080402, 2026080403)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def outcome_table(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    for row in rows:
        for key in ("strict_success_rate", "terminal_normalized_error_mean", "steps_mean", "boundary_risk_cost_mean"):
            row[key] = float(row[key])
        output[(row["group_id"], row["plan_name"])] = row
    return output


def selection_metrics(selection: Mapping[str, str], outcomes: Mapping[tuple[str, str], Mapping[str, Any]], ids: list[str]) -> dict[str, Any]:
    result = _evaluate_selection(selection, outcomes, ids)
    result["strict_success_count_equivalent"] = result["strict_success_rate"] * len(ids)
    counts = Counter(selection[group_id] for group_id in ids)
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(ids)
    result["plan_selection_entropy_bits"] = float(-np.sum(probabilities * np.log2(probabilities)))
    return result


def prompt_for(group: Mapping[str, Any], representative: Mapping[str, Any], *, visible_override: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    visible = dict(visible_override or _visible_state(group, representative))
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PREFIX + json.dumps(visible, sort_keys=True, separators=(",", ":"), allow_nan=False)}]},
    ]


def borda(predictions: list[dict[str, Any]], fallback: str) -> dict[str, str]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_group[row["group_id"]].append(row)
    output = {}
    for group_id, rows in by_group.items():
        scores = Counter()
        for row in rows:
            ranking = row["parsed"]["plan_ranking"] if row["valid_json"] else [fallback, *[name for name in PLAN_NAMES if name != fallback]]
            for rank, name in enumerate(ranking):
                scores[name] += len(PLAN_NAMES) - rank
        output[group_id] = sorted(PLAN_NAMES, key=lambda name: (-scores[name], PLAN_NAMES.index(name)))[0]
    return output


def evaluate(args: argparse.Namespace) -> None:
    groups = read_jsonl(ARTIFACT / "split_manifests/groups.jsonl")
    audit = json.loads((ARTIFACT / "final_outcome_audit.json").read_text())
    if not audit.get("gate_passed"):
        raise RuntimeError("SFT/evaluation is forbidden because the information gate did not pass")
    rollouts = read_jsonl(ARTIFACT / "rollout_results.jsonl")
    representative = {row["group_id"]: row for row in rollouts if row["plan_name"] == "direct_all_five" and int(row["seed_index"]) == 0}
    outcomes = outcome_table(ARTIFACT / "plan_outcomes.csv")
    group_by_id = {row["group_id"]: row for row in groups}
    train_dev_ids = [row["group_id"] for row in groups if row["split"] in {"candidate_train", "candidate_dev"}]
    decisive_ids = [group_id for group_id in train_dev_ids if audit["group_rankings"][group_id]["decisive"]]
    confirmation = [row for row in groups if row["split"] == "candidate_confirmation"]
    confirmation_ids = [row["group_id"] for row in confirmation]
    labels = {group_id: audit["group_rankings"][group_id]["selected_oracle_plan"] for group_id in decisive_ids}
    oracle = {group_id: audit["group_rankings"][group_id]["selected_oracle_plan"] for group_id in confirmation_ids}
    fixed_scores = {name: selection_metrics({gid: name for gid in train_dev_ids}, outcomes, train_dev_ids) for name in PLAN_NAMES}
    best_fixed = sorted(PLAN_NAMES, key=lambda name: (-fixed_scores[name]["strict_success_rate"], fixed_scores[name]["terminal_normalized_error_mean"], fixed_scores[name]["steps_mean"], PLAN_NAMES.index(name)))[0]
    frequency = Counter(labels.values()).most_common(1)[0][0]
    frequency_counts = Counter(labels.values())
    frequency_names = sorted(frequency_counts)
    frequency_probabilities = np.asarray([frequency_counts[name] for name in frequency_names], dtype=np.float64)
    frequency_probabilities /= frequency_probabilities.sum()
    rng = np.random.default_rng(2026080413)
    frequency_random = {group_id: str(rng.choice(frequency_names, p=frequency_probabilities)) for group_id in confirmation_ids}
    visual = {}
    for group in confirmation:
        family_labels = [labels[gid] for gid in decisive_ids if group_by_id[gid]["visual_family"] == group["visual_family"]]
        visual[group["group_id"]] = Counter(family_labels).most_common(1)[0][0] if family_labels else frequency
    features = {gid: _feature_vector(group_by_id[gid], representative[gid]) for gid in decisive_ids + confirmation_ids}
    numeric_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=2026080412, batch_size=min(16, len(decisive_ids))))
    numeric_model.fit(np.stack([features[gid] for gid in decisive_ids]), np.asarray([labels[gid] for gid in decisive_ids]))
    numeric_predictions = numeric_model.predict(np.stack([features[gid] for gid in confirmation_ids]))
    numeric = dict(zip(confirmation_ids, map(str, numeric_predictions), strict=True))
    baseline = {
        "oracle": selection_metrics(oracle, outcomes, confirmation_ids),
        "direct_all_five": selection_metrics({gid: "direct_all_five" for gid in confirmation_ids}, outcomes, confirmation_ids),
        "best_fixed": {"plan": best_fixed, **selection_metrics({gid: best_fixed for gid in confirmation_ids}, outcomes, confirmation_ids)},
        "frequency_only": {"plan": frequency, **selection_metrics({gid: frequency for gid in confirmation_ids}, outcomes, confirmation_ids)},
        "frequency_matched_random": selection_metrics(frequency_random, outcomes, confirmation_ids),
        "visual_diagnosis_lookup": selection_metrics(visual, outcomes, confirmation_ids),
        "numerical_only_mlp": selection_metrics(numeric, outcomes, confirmation_ids),
    }
    for name, metrics in baseline.items():
        if name == "oracle":
            continue
        metrics["oracle_strict_success_regret"] = baseline["oracle"]["strict_success_rate"] - metrics["strict_success_rate"]
        metrics["oracle_terminal_error_regret"] = metrics["terminal_normalized_error_mean"] - baseline["oracle"]["terminal_normalized_error_mean"]
    predictions = []
    for seed in SEEDS:
        path = ARTIFACT / f"predictions/confirmation_seed_{seed}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != len(confirmation_ids) or any(int(row["seed"]) != seed for row in rows):
            raise RuntimeError(f"prediction bundle for seed {seed} is incomplete")
        predictions.extend(rows)
    learned = borda(predictions, best_fixed)
    per_seed = {}
    for seed in SEEDS:
        rows = [row for row in predictions if row["seed"] == seed]
        selection = {row["group_id"]: row["parsed"]["selected_plan"] if row["valid_json"] else best_fixed for row in rows}
        per_seed[str(seed)] = {"valid_json_rate": float(np.mean([row["valid_json"] for row in rows])), **selection_metrics(selection, outcomes, confirmation_ids)}
    learned_metrics = selection_metrics(learned, outcomes, confirmation_ids)
    learned_metrics["per_training_seed"] = per_seed
    learned_metrics["beats_best_fixed_strict_success"] = learned_metrics["strict_success_rate"] > baseline["best_fixed"]["strict_success_rate"]
    learned_metrics["beats_visual_lookup_strict_success"] = learned_metrics["strict_success_rate"] > baseline["visual_diagnosis_lookup"]["strict_success_rate"]
    learned_metrics["beats_numerical_only_strict_success"] = learned_metrics["strict_success_rate"] > baseline["numerical_only_mlp"]["strict_success_rate"]
    learned_metrics["terminal_error_reduction_vs_best_fixed"] = baseline["best_fixed"]["terminal_normalized_error_mean"] - learned_metrics["terminal_normalized_error_mean"]
    learned_metrics["oracle_strict_success_regret"] = baseline["oracle"]["strict_success_rate"] - learned_metrics["strict_success_rate"]
    learned_metrics["oracle_terminal_error_regret"] = learned_metrics["terminal_normalized_error_mean"] - baseline["oracle"]["terminal_normalized_error_mean"]

    selector_selections = {"direct_all_five": {gid: "direct_all_five" for gid in confirmation_ids}, "best_fixed": {gid: best_fixed for gid in confirmation_ids}, "frequency_only": {gid: frequency for gid in confirmation_ids}, "frequency_matched_random": frequency_random, "visual_diagnosis_lookup": visual, "numerical_only_mlp": numeric, "qwen_borda_3seed": learned, "oracle": oracle}
    per_visual_family = {}
    for family in sorted({str(group["visual_family"]) for group in confirmation}):
        family_ids = [group["group_id"] for group in confirmation if group["visual_family"] == family]
        per_visual_family[family] = {name: selection_metrics(selection, outcomes, family_ids) for name, selection in selector_selections.items()}
    output = {"candidate_only": True, "confirmation_groups": len(confirmation_ids), "confirmation_episodes_already_physically_executed": len(confirmation_ids) * len(PLAN_NAMES) * 3, "selection_replay_uses_real_outcomes": True, "best_fixed_chosen_without_confirmation": best_fixed, "baselines": baseline, "selector_selections": selector_selections, "per_visual_family": per_visual_family, "qwen_borda_3seed": learned_metrics, "qwen_selection": learned, "oracle_selection": oracle, "raw_predictions": predictions, "frozen_or_protected_enabled": False}
    path = ARTIFACT / "confirmation_evaluation.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"qwen": learned_metrics, "baselines": baseline}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
