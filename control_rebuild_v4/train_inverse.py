#!/usr/bin/env python3
"""Train numerical inverse control with v4 physics and measurement errors."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import (
    ACTION_GRID,
    MOVEMENT,
    inverse_candidate_features,
    read_jsonl,
)
from control_rebuild_v3.forward_runtime import load_forward_runtime
from control_rebuild_v3.inverse_runtime import load_inverse_runtime
from control_rebuild_v3.models import inverse_ranker_model
from control_rebuild_v3.train_forward import configure, iter_batches
from control_rebuild_v3.train_inverse import (
    feature_statistics,
    inverse_metrics,
    predict_corrections,
)
from control_rebuild_v4.assess_validation import THRESHOLDS
from control_rebuild_v4.compare_v3_v4_selection_validation import (
    evaluate_rows,
)
from control_rebuild_v4.forward_runtime import load_forward_runtime_v4
from control_rebuild_v4.inverse_data import (
    derived_inverse_pairs,
    request_arrays,
    state_mapping,
)
from control_rebuild_v4.selection_gates import (
    gate_aware_selection_key,
    gate_margin_summary,
)

DEFAULT_V2_DATA = REPO_ROOT.parent / "VLM_data/specialist_rebuild_v2"
DEFAULT_V4_DATA = REPO_ROOT.parent / "VLM_data/control_rebuild_v4_numerical"
DEFAULT_OUTPUT = REPO_ROOT.parent / "VLM_runs/control_rebuild_v4_one_seed"
DEFAULT_ERROR_BANK = DEFAULT_OUTPUT / "measurement_error_bank_v4.npz"
DEFAULT_V3_INVERSE = (
    REPO_ROOT.parent / "VLM_runs/control_rebuild_v3_one_seed/inverse_control_v3.pt"
)
DEFAULT_V3_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/control_rebuild_v3_one_seed/forward_control_v3_calibrated.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-data", type=Path, default=DEFAULT_V2_DATA)
    parser.add_argument("--v4-data", type=Path, default=DEFAULT_V4_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--forward-artifact",
        type=Path,
        default=DEFAULT_OUTPUT / "forward_physics_residual_v4.pt",
    )
    parser.add_argument("--error-bank", type=Path, default=DEFAULT_ERROR_BANK)
    parser.add_argument("--initialization", type=Path, default=DEFAULT_V3_INVERSE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-pairs", type=int, default=256)
    parser.add_argument("--max-v2-train-pairs", type=int)
    parser.add_argument("--max-v4-train-groups", type=int)
    parser.add_argument("--v4-train-repeat", type=int, default=1)
    parser.add_argument(
        "--v3-forward-artifact",
        type=Path,
        default=DEFAULT_V3_FORWARD,
    )
    parser.add_argument(
        "--v3-inverse-artifact",
        type=Path,
        default=DEFAULT_V3_INVERSE,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def noisy_rows(
    rows: Sequence[Mapping[str, Any]],
    measured_current: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "group_id": f"{row['group_id']}:measurement_augmented",
            "setup": row["setup"],
            "current_beam_state": state_mapping(measured_current[index]),
        }
        for index, row in enumerate(rows)
    ]


def prepare_requests(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    forward: Any,
    error_bank: np.ndarray,
    error_conditions: np.ndarray,
    seed: int,
    noisy_fraction: float,
) -> dict[str, Any]:
    clean_states = forward.predict_states(rows)
    # request_arrays samples one current-state error per group.  Its first pass
    # exposes those perturbed currents; use them to recompute forward states.
    placeholder = request_arrays(
        pairs,
        rows,
        clean_states,
        clean_states,
        error_bank,
        error_conditions,
        seed,
        noisy_fraction,
    )
    perturbed = noisy_rows(rows, placeholder["measured_current_by_group"])
    noisy_states = forward.predict_states(perturbed)
    return request_arrays(
        pairs,
        rows,
        clean_states,
        noisy_states,
        error_bank,
        error_conditions,
        seed,
        noisy_fraction,
    )


def evaluation_block(
    torch: Any,
    model: Any,
    arrays: Mapping[str, Any],
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_scale: np.ndarray,
    status_mean: np.ndarray,
    status_scale: np.ndarray,
    alpha_candidates: Sequence[float],
    device: Any,
) -> dict[str, Any]:
    contexts = ((arrays["contexts"] - context_mean) / context_scale).astype(np.float32)
    correction, logits, base = predict_corrections(
        torch,
        model,
        contexts,
        arrays["candidate_bank"],
        arrays["candidate_indices"],
        arrays["desired_observed"],
        candidate_mean,
        candidate_scale,
        status_mean,
        status_scale,
        device,
    )
    candidates = []
    for alpha in alpha_candidates:
        candidates.append(
            {
                "alpha": float(alpha),
                **inverse_metrics(
                    base + float(alpha) * correction,
                    logits,
                    arrays["positives"],
                    arrays["statuses"],
                    arrays["selected"],
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["target_success_feasible"],
            row["minimum_movement_exact_feasible"],
            row["status_macro_f1"],
            -row["alpha"],
        ),
    )
    return {"selected": selected, "alpha_candidates": candidates}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device = configure(20260726, args.device)

    v2_train_rows = read_jsonl(args.v2_data.resolve() / "grids/train.jsonl")
    v2_val_rows = read_jsonl(args.v2_data.resolve() / "grids/val.jsonl")
    v4_train_rows = read_jsonl(args.v4_data.resolve() / "grids/train.jsonl")
    v4_val_rows = read_jsonl(args.v4_data.resolve() / "grids/val.jsonl")
    if args.max_v4_train_groups is not None:
        v4_train_rows = v4_train_rows[: args.max_v4_train_groups]
    v2_train_pairs = read_jsonl(args.v2_data.resolve() / "inverse/train.jsonl")
    if args.max_v2_train_pairs is not None:
        v2_train_pairs = v2_train_pairs[: args.max_v2_train_pairs]
    v2_val_pairs = read_jsonl(args.v2_data.resolve() / "inverse/val.jsonl")
    v4_train_pairs = derived_inverse_pairs(v4_train_rows)
    v4_val_pairs = derived_inverse_pairs(v4_val_rows)
    train_rows = [*v2_train_rows, *v4_train_rows]
    train_pairs = [*v2_train_pairs, *v4_train_pairs]
    v4_train_repeat = int(args.v4_train_repeat)
    if v4_train_repeat < 1:
        raise ValueError("--v4-train-repeat must be positive")

    loaded_bank = np.load(args.error_bank.resolve(), allow_pickle=False)
    error_bank = np.asarray(loaded_bank["normalized_errors"], dtype=np.float32)
    error_conditions = np.asarray(loaded_bank["condition_indices"], dtype=np.int64)
    forward, _ = load_forward_runtime_v4(args.forward_artifact.resolve(), torch, device)
    train = prepare_requests(
        train_rows,
        train_pairs,
        forward,
        error_bank,
        error_conditions,
        seed=20260726 + 701,
        noisy_fraction=0.5,
    )
    iid_clean = prepare_requests(
        v2_val_rows,
        v2_val_pairs,
        forward,
        error_bank,
        error_conditions,
        seed=20260726 + 702,
        noisy_fraction=0.0,
    )
    iid_noisy = prepare_requests(
        v2_val_rows,
        v2_val_pairs,
        forward,
        error_bank,
        error_conditions,
        seed=20260726 + 702,
        noisy_fraction=1.0,
    )
    expanded_clean = prepare_requests(
        v4_val_rows,
        v4_val_pairs,
        forward,
        error_bank,
        error_conditions,
        seed=20260726 + 703,
        noisy_fraction=0.0,
    )
    expanded_noisy = prepare_requests(
        v4_val_rows,
        v4_val_pairs,
        forward,
        error_bank,
        error_conditions,
        seed=20260726 + 703,
        noisy_fraction=1.0,
    )
    category_rows = {
        category: [
            row for row in v4_val_rows if str(row["source_category"]) == category
        ]
        for category in sorted({str(row["source_category"]) for row in v4_val_rows})
    }
    required_hard_categories = {"ood_boundary", "high_nonlinearity"}
    if not required_hard_categories.issubset(category_rows):
        raise ValueError("v4 validation lacks a boundary or high-nonlinearity category")
    expanded_clean_by_category = {
        category: prepare_requests(
            rows,
            derived_inverse_pairs(rows),
            forward,
            error_bank,
            error_conditions,
            seed=20260726 + 710 + index,
            noisy_fraction=0.0,
        )
        for index, (category, rows) in enumerate(category_rows.items())
    }
    v3_forward, _ = load_forward_runtime(
        args.v3_forward_artifact.resolve(),
        torch,
        device,
    )
    v3_inverse, _ = load_inverse_runtime(
        args.v3_inverse_artifact.resolve(),
        torch,
        device,
    )
    v3_reference = {
        "all": evaluate_rows(
            v4_val_rows,
            v3_forward,
            v3_inverse,
        ),
        "by_category": {
            category: evaluate_rows(
                rows,
                v3_forward,
                v3_inverse,
            )
            for category, rows in category_rows.items()
        },
    }

    base_training_indices = np.arange(len(train_pairs), dtype=np.int64)
    v4_training_indices = np.arange(
        len(v2_train_pairs),
        len(train_pairs),
        dtype=np.int64,
    )
    training_indices = np.concatenate(
        [
            base_training_indices,
            *[v4_training_indices for _ in range(v4_train_repeat - 1)],
        ]
    )
    context_mean = train["contexts"][training_indices].mean(axis=0)
    context_scale = train["contexts"][training_indices].std(axis=0)
    context_scale[context_scale < 1e-7] = 1.0
    contexts_train = ((train["contexts"] - context_mean) / context_scale).astype(
        np.float32
    )
    candidate_mean, candidate_scale, status_mean, status_scale = feature_statistics(
        train["candidate_bank"],
        train["candidate_indices"][training_indices],
        train["desired_observed"][training_indices],
    )
    sample_candidate, _, sample_status = inverse_candidate_features(
        train["candidate_bank"][train["candidate_indices"][:1]],
        train["desired_observed"][:1],
    )
    model = inverse_ranker_model(
        torch,
        contexts_train.shape[1],
        sample_candidate.shape[-1],
        sample_status.shape[-1],
    ).to(device)
    initialization = torch.load(
        args.initialization.resolve(),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(initialization["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    status_counts = Counter(train["statuses"][training_indices].tolist())
    status_weights = torch.as_tensor(
        [
            math.sqrt(len(training_indices) / max(3 * status_counts.get(index, 0), 1))
            for index in range(3)
        ],
        dtype=torch.float32,
        device=device,
    )
    alpha_candidates = [0.0, 0.1, 0.25, 0.5, 1.0]
    rng = np.random.default_rng(20260726 + 704)
    best_score = float("-inf")
    best_selection_key = None
    best_epoch = 0
    best_alpha = 0.0
    best_state = None
    trace = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        for position in iter_batches(
            len(training_indices),
            int(args.batch_pairs),
            rng,
        ):
            index = training_indices[position]
            candidate, cost, status_feature = inverse_candidate_features(
                train["candidate_bank"][train["candidate_indices"][index]],
                train["desired_observed"][index],
            )
            candidate = (candidate - candidate_mean) / candidate_scale
            status_feature = (status_feature - status_mean) / status_scale
            correction, logits = model(
                torch.as_tensor(
                    contexts_train[index],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(candidate, dtype=torch.float32, device=device),
                torch.as_tensor(
                    status_feature,
                    dtype=torch.float32,
                    device=device,
                ),
            )
            positive = torch.as_tensor(
                train["positives"][index],
                dtype=torch.bool,
                device=device,
            )
            status_target = torch.as_tensor(
                train["statuses"][index],
                dtype=torch.long,
                device=device,
            )
            scores = (
                torch.as_tensor(-cost, dtype=torch.float32, device=device) + correction
            )
            feasible = positive.any(dim=1)
            if feasible.any():
                positive_scores = scores[feasible].masked_fill(
                    ~positive[feasible], -1e9
                )
                rank_loss = (
                    torch.logsumexp(scores[feasible], dim=1)
                    - torch.logsumexp(positive_scores, dim=1)
                ).mean()
            else:
                rank_loss = correction.sum() * 0.0
            status_loss = torch.nn.functional.cross_entropy(
                logits, status_target, weight=status_weights
            )
            loss = rank_loss + 0.45 * status_loss + 0.002 * correction.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(index)

        blocks = {
            "iid_clean": evaluation_block(
                torch,
                model,
                iid_clean,
                context_mean,
                context_scale,
                candidate_mean,
                candidate_scale,
                status_mean,
                status_scale,
                alpha_candidates,
                device,
            ),
            "iid_measurement_augmented": evaluation_block(
                torch,
                model,
                iid_noisy,
                context_mean,
                context_scale,
                candidate_mean,
                candidate_scale,
                status_mean,
                status_scale,
                alpha_candidates,
                device,
            ),
            "expanded_clean": evaluation_block(
                torch,
                model,
                expanded_clean,
                context_mean,
                context_scale,
                candidate_mean,
                candidate_scale,
                status_mean,
                status_scale,
                alpha_candidates,
                device,
            ),
            "expanded_measurement_augmented": evaluation_block(
                torch,
                model,
                expanded_noisy,
                context_mean,
                context_scale,
                candidate_mean,
                candidate_scale,
                status_mean,
                status_scale,
                alpha_candidates,
                device,
            ),
        }
        category_blocks = {
            category: evaluation_block(
                torch,
                model,
                arrays,
                context_mean,
                context_scale,
                candidate_mean,
                candidate_scale,
                status_mean,
                status_scale,
                alpha_candidates,
                device,
            )
            for category, arrays in expanded_clean_by_category.items()
        }
        selected_rows = [block["selected"] for block in blocks.values()]
        alpha_scores = {}
        alpha_gate_summaries = {}
        for alpha in alpha_candidates:
            rows_for_alpha = [
                next(row for row in block["alpha_candidates"] if row["alpha"] == alpha)
                for block in blocks.values()
            ]
            alpha_scores[alpha] = float(
                np.mean(
                    [
                        row["target_success_feasible"]
                        + 0.05 * row["minimum_movement_exact_feasible"]
                        + 0.10 * row["status_macro_f1"]
                        for row in rows_for_alpha
                    ]
                )
            )
            metrics_for_alpha = {
                name: next(
                    row for row in block["alpha_candidates"] if row["alpha"] == alpha
                )
                for name, block in blocks.items()
            }
            category_metrics_for_alpha = {
                category: next(
                    row for row in block["alpha_candidates"] if row["alpha"] == alpha
                )
                for category, block in category_blocks.items()
            }
            expanded_target = metrics_for_alpha["expanded_clean"][
                "target_success_feasible"
            ]
            expanded_noisy_target = metrics_for_alpha["expanded_measurement_augmented"][
                "target_success_feasible"
            ]
            retention = (
                expanded_noisy_target / expanded_target
                if expanded_target > 0.0
                else 0.0
            )
            alpha_gate_summaries[alpha] = gate_margin_summary(
                {
                    "same_distribution_overall_inverse_delta": (
                        expanded_target
                        - v3_reference["all"]["inverse"]["target_success_feasible"]
                        - THRESHOLDS["same_distribution_overall_inverse_delta"]
                    ),
                    "same_distribution_inverse_status_delta": (
                        metrics_for_alpha["expanded_clean"]["status_accuracy"]
                        - v3_reference["all"]["inverse"]["status_accuracy"]
                        - THRESHOLDS["same_distribution_inverse_status_delta_floor"]
                    ),
                    "ood_boundary_inverse_delta": (
                        category_metrics_for_alpha["ood_boundary"][
                            "target_success_feasible"
                        ]
                        - v3_reference["by_category"]["ood_boundary"]["inverse"][
                            "target_success_feasible"
                        ]
                        - THRESHOLDS["hard_category_delta_floor"]
                    ),
                    "high_nonlinearity_inverse_delta": (
                        category_metrics_for_alpha["high_nonlinearity"][
                            "target_success_feasible"
                        ]
                        - v3_reference["by_category"]["high_nonlinearity"]["inverse"][
                            "target_success_feasible"
                        ]
                        - THRESHOLDS["hard_category_delta_floor"]
                    ),
                    "old_iid_inverse_floor": (
                        metrics_for_alpha["iid_clean"]["target_success_feasible"]
                        - THRESHOLDS["iid_inverse_floor"]
                    ),
                    "paired_error_retention_floor": (
                        retention - THRESHOLDS["paired_error_retention_floor"]
                    ),
                }
            )
        epoch_alpha = max(
            alpha_scores,
            key=lambda alpha: gate_aware_selection_key(
                alpha_gate_summaries[alpha],
                alpha_scores[alpha],
                -alpha,
            ),
        )
        score = alpha_scores[epoch_alpha]
        selection_key = gate_aware_selection_key(
            alpha_gate_summaries[epoch_alpha],
            score,
            -epoch_alpha,
        )
        metrics_at_selected_alpha = {
            name: next(
                candidate
                for candidate in block["alpha_candidates"]
                if candidate["alpha"] == epoch_alpha
            )
            for name, block in blocks.items()
        }
        row = {
            "epoch": epoch,
            "train_loss": running / len(training_indices),
            "score": score,
            "selected_alpha": float(epoch_alpha),
            "metrics_at_selected_alpha": metrics_at_selected_alpha,
            "validation": blocks,
            "validation_expanded_by_category": category_blocks,
            "component_selected_rows": selected_rows,
            "checkpoint_selection_gates": alpha_gate_summaries[epoch_alpha],
            "checkpoint_selection_key": list(selection_key),
            "alpha_gate_summaries": alpha_gate_summaries,
        }
        trace.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best_selection_key is None or selection_key > best_selection_key:
            best_score = score
            best_selection_key = selection_key
            best_epoch = epoch
            best_alpha = float(epoch_alpha)
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("inverse v4 produced no checkpoint")
    artifact_path = output_dir / "inverse_control_v4.pt"
    torch.save(
        {
            "version": "control_rebuild_v4_one_seed",
            "seed": 20260726,
            "model": "measurement_augmented_inverse_control_v4",
            "state_dict": best_state,
            "context_dim": int(contexts_train.shape[1]),
            "candidate_dim": int(sample_candidate.shape[-1]),
            "status_dim": int(sample_status.shape[-1]),
            "context_mean": context_mean,
            "context_scale": context_scale,
            "candidate_mean": candidate_mean,
            "candidate_scale": candidate_scale,
            "status_mean": status_mean,
            "status_scale": status_scale,
            "correction_alpha": best_alpha,
            "statuses": [
                "unique",
                "ambiguous",
                "infeasible_within_limits",
            ],
            "action_grid": ACTION_GRID,
            "forward_artifact": str(args.forward_artifact.resolve()),
            "measurement_error_bank": str(args.error_bank.resolve()),
            "measurement_augmented_fraction": 0.5,
            "measurement_error_condition_pairing": (
                "current_and_desired_same_condition"
            ),
            "v4_train_repeat": v4_train_repeat,
            "checkpoint_selection": (
                "validation_only_gate_count_then_worst_margin_then_composite"
            ),
        },
        artifact_path,
    )
    summary = {
        "version": "control_rebuild_v4_one_seed",
        "artifact": str(artifact_path),
        "forward_artifact": str(args.forward_artifact.resolve()),
        "error_bank": str(args.error_bank.resolve()),
        "measurement_error_condition_pairing": ("current_and_desired_same_condition"),
        "initialization": str(args.initialization.resolve()),
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training": {
            "v2_pairs": len(v2_train_pairs),
            "v4_unique_derived_pairs": len(v4_train_pairs),
            "v4_train_repeat": v4_train_repeat,
            "v4_derived_pairs": len(v4_train_pairs) * v4_train_repeat,
            "total_unique_pairs": len(train_pairs),
            "total_pairs": len(training_indices),
            "measurement_augmented_pairs": int(train["noisy"][training_indices].sum()),
        },
        "validation_pair_counts": {
            "iid": len(v2_val_pairs),
            "expanded": len(v4_val_pairs),
        },
        "epochs": int(args.epochs),
        "best_epoch": best_epoch,
        "best_alpha": best_alpha,
        "best_composite_score": best_score,
        "best_selection_key": list(best_selection_key),
        "v3_selection_validation_reference": v3_reference,
        "validation": trace[best_epoch - 1],
        "trace": trace,
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "inverse_control_v4_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
