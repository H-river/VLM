#!/usr/bin/env python3
"""Train and evaluate the direct multi-positive v10 numerical inverse ranker."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import residual_components
from control_rebuild_v4.inverse_data import state_mapping
from physics_structured_rebuild_v10.contracts import (
    STATE_FIELDS,
    explicit_action_features,
    group_targets,
    physical_success,
    sha256_file,
    stable_seed,
    visible_context,
)
from physics_structured_rebuild_v10.evaluate import mean_ci, paired_difference_ci
from physics_structured_rebuild_v10.models import build_inverse_ranker
from physics_structured_rebuild_v9.forward_runtime import (
    load_residual_forward_runtime_v9,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_COMBINED_NATURAL_INVERSE_V9,
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
)
from specialist_rebuild_v2.common import raw_state_array

DEFAULT_CONFIG = Path(__file__).with_name("config_v10.json")
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v10_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def request_source_indices(row: dict[str, Any]) -> list[int]:
    indices = [int(row["natural_requested_action_index"])]
    indices.extend(
        int(request["source_action_index"])
        for request in row["visual_requests"]
    )
    return indices


def request_arrays(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    features = []
    desired = []
    labels = []
    hard_negative = []
    group_index = []
    source_index = []
    candidate_states_by_group = []
    for row_index, row in enumerate(rows):
        states, _ = group_targets(row)
        candidate_states_by_group.append(states)
        current = raw_state_array(row["current_beam_state"]).astype(np.float32)
        for target_index in request_source_indices(row):
            target = states[target_index]
            inverse_tolerance = np.asarray(
                [
                    0.5,
                    0.5,
                    1.0,
                    1.0,
                    max(0.02 * abs(float(target[4])), 1e-6),
                ],
                dtype=np.float32,
            )
            target_transformed = target.copy()
            target_transformed[4] = np.log1p(max(float(target_transformed[4]), 0.0))
            feature = np.concatenate(
                [
                    visible_context(row["setup"], row["current_beam_state"]),
                    target_transformed,
                    (target - current) / inverse_tolerance,
                ]
            )
            success = physical_success(states, target)
            components = np.abs(
                residual_components(states, target[None, :])
            )
            condition_failures = np.stack(
                [
                    np.hypot(components[:, 0], components[:, 1]) > 1.0,
                    components[:, 2] > 1.0,
                    components[:, 3] > 1.0,
                    components[:, 4] > 1.0,
                ],
                axis=1,
            )
            near = (
                (condition_failures.sum(axis=1) == 1)
                & (components.max(axis=1) <= 1.5)
                & ~success
            )
            features.append(feature)
            desired.append(target)
            labels.append(success)
            hard_negative.append(near)
            group_index.append(row_index)
            source_index.append(target_index)
    return {
        "features": np.asarray(features, dtype=np.float32),
        "desired": np.asarray(desired, dtype=np.float32),
        "labels": np.asarray(labels, dtype=bool),
        "hard_negative": np.asarray(hard_negative, dtype=bool),
        "group_index": np.asarray(group_index, dtype=np.int64),
        "source_index": np.asarray(source_index, dtype=np.int64),
        "candidate_states_by_group": np.asarray(
            candidate_states_by_group, dtype=np.float32
        ),
    }


def group_partition(
    rows: list[dict[str, Any]], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(
        [stable_seed(seed, row["group_id"], "inverse_split") for row in rows],
        dtype=np.uint64,
    )
    order = np.argsort(tokens)
    calibration_count = max(1, int(round(0.15 * len(rows))))
    return order[calibration_count:], order[:calibration_count]


def requests_for_groups(
    request_group_index: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    return np.flatnonzero(np.isin(request_group_index, groups))


def listwise_loss(
    torch: Any,
    scores: Any,
    positives: Any,
    hard_negatives: Any,
) -> Any:
    negative_infinity = torch.finfo(scores.dtype).min
    positive_logsum = torch.logsumexp(
        scores.masked_fill(~positives, negative_infinity),
        dim=1,
    )
    all_logsum = torch.logsumexp(scores, dim=1)
    positive_count = positives.sum(dim=1).float()
    request_weight = positive_count.rsqrt()
    listwise = ((all_logsum - positive_logsum) * request_weight).sum()
    listwise = listwise / request_weight.sum().clamp_min(1e-6)
    has_hard = hard_negatives.any(dim=1)
    if has_hard.any():
        hard_logsum = torch.logsumexp(
            scores.masked_fill(~hard_negatives, negative_infinity),
            dim=1,
        )
        hard = torch.nn.functional.softplus(
            hard_logsum[has_hard] - positive_logsum[has_hard] + 0.25
        ).mean()
    else:
        hard = scores.new_zeros(())
    return listwise + 0.30 * hard


def score_batches(
    torch: Any,
    model: Any,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    actions: Any,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            request = torch.as_tensor(
                (features[start : start + batch_size] - mean) / scale,
                dtype=torch.float32,
                device=device,
            )
            output.append(model(request, actions).float().cpu().numpy())
    return np.concatenate(output)


def ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    group_index: np.ndarray,
    rows: list[dict[str, Any]],
    seed: int,
    draws: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    order = np.argsort(-scores, axis=1)
    ordered_positive = np.take_along_axis(labels, order, axis=1)
    top1 = ordered_positive[:, 0]
    positive_count = labels.sum(axis=1)
    first_rank = ordered_positive.argmax(axis=1) + 1
    unique_groups = np.unique(group_index)
    group_top1 = np.asarray(
        [top1[group_index == group].mean() for group in unique_groups]
    )
    by_positive_count = {}
    for bucket_name, mask in {
        "1": positive_count == 1,
        "2_to_4": (positive_count >= 2) & (positive_count <= 4),
        "5_plus": positive_count >= 5,
    }.items():
        by_positive_count[bucket_name] = {
            "count": int(mask.sum()),
            "top1_success": (
                None if not mask.any() else float(top1[mask].mean())
            ),
            "mean_first_success_rank": (
                None if not mask.any() else float(first_rank[mask].mean())
            ),
        }
    metrics = {
        "top1_physical_success": mean_ci(group_top1, seed, draws),
        "request_top1_count": int(top1.sum()),
        "request_count": int(len(top1)),
        "top_k_success": {
            str(k): float(ordered_positive[:, :k].any(axis=1).mean())
            for k in (1, 3, 5, 10)
        },
        "mean_rank_of_first_success": float(first_rank.mean()),
        "median_rank_of_first_success": float(np.median(first_rank)),
        "by_available_successful_actions": by_positive_count,
        "available_successful_actions": {
            "minimum": int(positive_count.min()),
            "median": float(np.median(positive_count)),
            "mean": float(positive_count.mean()),
            "maximum": int(positive_count.max()),
        },
    }
    return metrics, {
        "request_top1": top1.astype(np.float64),
        "group_top1": group_top1,
        "positive_count": positive_count,
        "first_rank": first_rank,
        "group_ids": unique_groups,
    }


def frozen_v9_scores(
    torch: Any,
    device: Any,
    rows: list[dict[str, Any]],
    requests: dict[str, Any],
) -> np.ndarray:
    forward, _ = load_residual_forward_runtime_v9(
        DEFAULT_NATURAL_GRID_FORWARD_STATE,
        torch,
        device,
    )
    inverse, _ = load_inverse_runtime_v8(
        DEFAULT_COMBINED_NATURAL_INVERSE_V9,
        torch,
        device,
    )
    predicted_by_group = forward.predict_states(rows)
    group_index = requests["group_index"]
    current = np.asarray(
        [raw_state_array(rows[int(index)]["current_beam_state"]) for index in group_index],
        dtype=np.float32,
    )
    scored = inverse.score_requests(
        [rows[int(index)]["setup"] for index in group_index],
        current,
        requests["desired"],
        predicted_by_group[group_index],
        feature_desired=requests["desired"],
        batch_size=128,
    )
    return np.asarray(scored["scores"], dtype=np.float32)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / ("inverse_ranker_smoke_v10.pt" if args.smoke else "inverse_ranker_v10.pt")
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite inverse output: {output}")
    started = time.perf_counter()
    train_rows = read_jsonl(data_dir / "grids" / "train.jsonl")
    development_rows = read_jsonl(data_dir / "grids" / "development.jsonl")
    if args.smoke:
        train_rows = train_rows[: min(8, len(train_rows))]
        development_rows = development_rows[: min(4, len(development_rows))]
    train_requests = request_arrays(train_rows)
    development_requests = request_arrays(development_rows)
    seed = int(config["model_seed"]) + 100
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    training_groups, calibration_groups = group_partition(train_rows, seed)
    training = requests_for_groups(
        train_requests["group_index"], training_groups
    )
    calibration = requests_for_groups(
        train_requests["group_index"], calibration_groups
    )
    feature_mean = train_requests["features"][training].mean(axis=0)
    feature_scale = train_requests["features"][training].std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    training_config = config["inverse_training"]
    model_config = {
        "request_dim": int(train_requests["features"].shape[1]),
        "action_dim": int(explicit_action_features().shape[1]),
        "dimension": int(training_config["dimension"]),
        "dropout": float(training_config["dropout"]),
    }
    model = build_inverse_ranker(torch, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["smoke_epochs"] if args.smoke else training_config["epochs"])
    batch_size = min(int(training_config["batch_size"]), len(training))
    actions = torch.as_tensor(
        explicit_action_features(), dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(seed + 1)
    best_key = None
    best_state = None
    best_epoch = 0
    stale = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(training)
        running = 0.0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            features = torch.as_tensor(
                (train_requests["features"][indices] - feature_mean) / feature_scale,
                dtype=torch.float32,
                device=device,
            )
            positives = torch.as_tensor(
                train_requests["labels"][indices], dtype=torch.bool, device=device
            )
            hard = torch.as_tensor(
                train_requests["hard_negative"][indices],
                dtype=torch.bool,
                device=device,
            )
            scores = model(features, actions)
            loss = listwise_loss(torch, scores, positives, hard)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(indices)
        calibration_scores = score_batches(
            torch,
            model,
            train_requests["features"][calibration],
            feature_mean,
            feature_scale,
            actions,
            device,
            batch_size,
        )
        selected = calibration_scores.argmax(axis=1)
        success = train_requests["labels"][calibration, selected]
        ordered = np.argsort(-calibration_scores, axis=1)
        ordered_positive = np.take_along_axis(
            train_requests["labels"][calibration],
            ordered,
            axis=1,
        )
        key = (
            int(success.sum()),
            -float((ordered_positive.argmax(axis=1) + 1).mean()),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            record = {
                "epoch": epoch,
                "training_loss": running / len(training),
                "calibration_top1_success": float(success.mean()),
                "best_epoch": best_epoch,
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if not args.smoke and stale >= int(training_config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("inverse training did not select a checkpoint")
    model.load_state_dict(best_state)
    development_scores = score_batches(
        torch,
        model,
        development_requests["features"],
        feature_mean,
        feature_scale,
        actions,
        device,
        batch_size,
    )
    draws = 200 if args.smoke else int(config["bootstrap_draws"])
    candidate_metrics, candidate_vectors = ranking_metrics(
        development_scores,
        development_requests["labels"],
        development_requests["group_index"],
        development_rows,
        stable_seed(int(config["bootstrap_seed"]), "inverse_candidate"),
        draws,
    )
    frozen_scores = frozen_v9_scores(
        torch,
        device,
        development_rows,
        development_requests,
    )
    frozen_metrics, frozen_vectors = ranking_metrics(
        frozen_scores,
        development_requests["labels"],
        development_requests["group_index"],
        development_rows,
        stable_seed(int(config["bootstrap_seed"]), "inverse_v9"),
        draws,
    )
    difference = paired_difference_ci(
        candidate_vectors["group_top1"],
        frozen_vectors["group_top1"],
        stable_seed(int(config["bootstrap_seed"]), "inverse_difference"),
        draws,
    )
    one_positive = candidate_vectors["positive_count"] == 1
    candidate_one = (
        None
        if not one_positive.any()
        else float(candidate_vectors["request_top1"][one_positive].mean())
    )
    frozen_one = (
        None
        if not one_positive.any()
        else float(frozen_vectors["request_top1"][one_positive].mean())
    )
    one_regression = (
        0.0
        if candidate_one is None or frozen_one is None
        else frozen_one - candidate_one
    )
    accepted = bool(
        difference["mean"] >= 0.03
        and difference["ci95_low"] > 0.0
        and one_regression <= 0.03
    )
    artifact = {
        "version": "physics_structured_rebuild_v10_direct_inverse_ranker",
        "model_config": model_config,
        "state_dict": best_state,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "seed": seed,
        "best_epoch": best_epoch,
        "training_group_ids": [train_rows[int(index)]["group_id"] for index in training_groups],
        "calibration_group_ids": [train_rows[int(index)]["group_id"] for index in calibration_groups],
        "locked_test_files_opened": [],
        "old_system_evaluation_files_opened": [],
    }
    torch.save(artifact, output)
    result = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "development": {
            "candidate": candidate_metrics,
            "frozen_v9": frozen_metrics,
            "paired_candidate_minus_v9": difference,
            "one_of_81_positive": {
                "request_count": int(one_positive.sum()),
                "candidate": candidate_one,
                "frozen_v9": frozen_one,
                "regression": one_regression,
            },
        },
        "acceptance": {
            "accepted": accepted,
            "minimum_mean_gain": 0.03,
            "paired_lower_bound_above": 0.0,
            "maximum_one_of_81_regression": 0.03,
        },
        "trace": trace,
        "smoke": bool(args.smoke),
        "locked_test_files_opened": [],
        "old_system_evaluation_files_opened": [],
        "seconds": time.perf_counter() - started,
        "complete": True,
    }
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
