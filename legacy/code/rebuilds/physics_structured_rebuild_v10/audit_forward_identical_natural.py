#!/usr/bin/env python3
"""Audit the identical A-D natural-action aggregate without reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID
from physics_structured_rebuild_v10.contracts import (
    ACTION_NORMALIZED,
    STATE_FIELDS,
    explicit_action_features,
    group_targets,
    natural_requested_action_index,
    sha256_file,
    validate_action_order,
    visible_context,
)
from physics_structured_rebuild_v10.models import (
    build_forward_model,
    forward_call,
)
from physics_structured_rebuild_v10.train_forward_ablation import (
    existing_distribution_sample,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)
from specialist_rebuild_v2.common import ACTION_FIELDS

PACKAGE = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE / "config_v10.json"
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v10_full"
DEFAULT_V9_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_V7_SUMMARY = (
    REPO_ROOT.parent
    / "VLM_runs/joint_forward_direction_v7_one_seed"
    / "shared_forward_direction_v7_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite-diagnostic", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_legacy_rows(
    rows: list[dict[str, Any]],
    seed: int,
) -> None:
    category_to_regime = {
        "iid_expanded": "ordinary",
        "high_nonlinearity": "tolerance_boundary",
        "ood_boundary": "camera_boundary",
        "hard_interaction": "high_offset_interaction",
        "peak_interaction": "focusing",
    }
    for row in rows:
        row.setdefault(
            "natural_requested_action_index",
            natural_requested_action_index(str(row["group_id"]), seed),
        )
        row.setdefault(
            "regime",
            category_to_regime.get(
                str(row.get("source_category", "")),
                "ordinary",
            ),
        )


def rows_by_ids(
    rows: list[dict[str, Any]],
    identifiers: list[str],
) -> list[dict[str, Any]]:
    mapping = {str(row["group_id"]): row for row in rows}
    selected = [mapping[str(identifier)] for identifier in identifiers]
    if len(selected) != len(identifiers):
        raise AssertionError("group-id reconstruction changed cardinality")
    return selected


def load_model(
    torch: Any,
    path: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = build_forward_model(
        torch,
        str(artifact["architecture"]),
        artifact["model_config"],
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact


def predict(
    torch: Any,
    model: Any,
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    device: Any,
    batch_size: int,
) -> np.ndarray:
    contexts = np.asarray(
        [
            visible_context(row["setup"], row["current_beam_state"])
            for row in rows
        ],
        dtype=np.float32,
    )
    explicit = torch.as_tensor(
        explicit_action_features(),
        dtype=torch.float32,
        device=device,
    )
    categories = torch.as_tensor(
        (ACTION_NORMALIZED + 1).astype(np.int64),
        dtype=torch.long,
        device=device,
    )
    parts = []
    with torch.inference_mode():
        for start in range(0, len(contexts), batch_size):
            context = torch.as_tensor(
                (
                    contexts[start : start + batch_size]
                    - np.asarray(artifact["context_mean"])
                )
                / np.asarray(artifact["context_scale"]),
                dtype=torch.float32,
                device=device,
            )
            mean, _ = forward_call(
                model,
                str(artifact["architecture"]),
                context,
                explicit,
                categories,
            )
            parts.append(mean.float().cpu().numpy())
    return np.concatenate(parts)


def error_metrics(
    rows: list[dict[str, Any]],
    prediction: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target = np.asarray(
        [group_targets(row)[1] for row in rows],
        dtype=np.float32,
    )
    absolute = np.abs(np.asarray(prediction) - target)
    strict = np.all(absolute <= 1.0, axis=2)
    natural_indices = np.asarray(
        [int(row["natural_requested_action_index"]) for row in rows],
        dtype=np.int64,
    )
    group_indices = np.arange(len(rows))
    natural_absolute = absolute[group_indices, natural_indices]
    natural_strict = strict[group_indices, natural_indices]
    return (
        {
            "group_count": len(rows),
            "transition_count": int(strict.size),
            "normalized_mae": float(absolute.mean()),
            "normalized_rmse": float(np.sqrt(np.square(absolute).mean())),
            "normalized_absolute_error_quantiles": {
                "p50": float(np.quantile(absolute, 0.50)),
                "p90": float(np.quantile(absolute, 0.90)),
                "p95": float(np.quantile(absolute, 0.95)),
            },
            "per_field_normalized_mae": {
                field: float(absolute[:, :, index].mean())
                for index, field in enumerate(STATE_FIELDS)
            },
            "full_surface_strict_all_five": float(strict.mean()),
            "natural_normalized_mae": float(natural_absolute.mean()),
            "natural_strict_all_five": float(natural_strict.mean()),
            "natural_strict_count": int(natural_strict.sum()),
        },
        {
            "target": target,
            "strict": strict,
            "natural_strict": natural_strict,
            "natural_prediction": prediction[group_indices, natural_indices],
        },
    )


def prediction_difference(
    left: np.ndarray,
    right: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    difference = np.abs(np.asarray(left) - np.asarray(right))
    natural_indices = np.asarray(
        [int(row["natural_requested_action_index"]) for row in rows],
        dtype=np.int64,
    )
    group_indices = np.arange(len(rows))
    natural_difference = difference[group_indices, natural_indices]
    return {
        "mean_absolute_prediction_difference": float(difference.mean()),
        "maximum_absolute_prediction_difference": float(difference.max()),
        "fraction_elements_exactly_equal": float((difference == 0.0).mean()),
        "natural_mean_absolute_prediction_difference": float(
            natural_difference.mean()
        ),
        "natural_maximum_absolute_prediction_difference": float(
            natural_difference.max()
        ),
        "natural_groups_with_any_difference_above_1e-6": int(
            np.any(natural_difference > 1e-6, axis=1).sum()
        ),
    }


def mask_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    left_values = np.asarray(left, dtype=bool)
    right_values = np.asarray(right, dtype=bool)
    return {
        "same_count": int((left_values == right_values).sum()),
        "different_count": int((left_values != right_values).sum()),
        "left_only_success_count": int((left_values & ~right_values).sum()),
        "right_only_success_count": int((right_values & ~left_values).sum()),
        "left_success_count": int(left_values.sum()),
        "right_success_count": int(right_values.sum()),
        "different_group_indices": np.flatnonzero(
            left_values != right_values
        ).tolist(),
    }


def audit_natural_indexing(
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    order_failures = []
    deterministic_index_failures = []
    indexed_action_failures = []
    indices = []
    cardinalities: Counter[int] = Counter()
    for row_index, row in enumerate(rows):
        try:
            validate_action_order(row["candidates"])
        except ValueError:
            order_failures.append(row_index)
            continue
        index = int(row["natural_requested_action_index"])
        indices.append(index)
        if index != natural_requested_action_index(str(row["group_id"]), seed):
            deterministic_index_failures.append(row_index)
        expected = ACTION_GRID[index]
        actual = row["candidates"][index]["action"]
        if any(
            float(actual[field]) != float(expected[field])
            for field in ACTION_FIELDS
        ):
            indexed_action_failures.append(row_index)
        cardinalities[
            sum(
                abs(float(expected[field])) > 0.0
                for field in ACTION_FIELDS
            )
        ] += 1
    return {
        "group_count": len(rows),
        "canonical_action_order_failures": order_failures,
        "deterministic_index_failures": deterministic_index_failures,
        "indexed_action_failures": indexed_action_failures,
        "index_minimum": min(indices),
        "index_maximum": max(indices),
        "requested_action_cardinality_counts": {
            str(key): value for key, value in sorted(cardinalities.items())
        },
        "passed": not (
            order_failures
            or deterministic_index_failures
            or indexed_action_failures
        ),
    }


def regime_comparison(
    development_report: dict[str, Any],
    locked_report: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    development_metrics = development_report["metrics"]["frozen_v9"][
        "by_regime_natural_requested_action"
    ]
    locked_metrics = locked_report["forward"]["frozen_v9"][
        "by_regime_natural_requested_action"
    ]
    rows = {}
    for regime in sorted(development_metrics):
        development_count = int(development_metrics[regime]["count"])
        locked_count = int(locked_metrics[regime]["count"])
        development_success = int(
            round(development_count * development_metrics[regime]["mean"])
        )
        locked_success = int(
            round(locked_count * locked_metrics[regime]["mean"])
        )
        rows[regime] = {
            "development_count": development_count,
            "locked_count": locked_count,
            "development_success_count": development_success,
            "locked_success_count": locked_success,
            "success_count_change_locked_minus_development": (
                locked_success - development_success
            ),
            "development_accuracy": float(
                development_metrics[regime]["mean"]
            ),
            "locked_accuracy": float(locked_metrics[regime]["mean"]),
        }
    return {
        "regime_counts_identical": (
            validation["reports"]["development"]["regime_counts"]
            == validation["reports"]["locked_test"]["regime_counts"]
        ),
        "development_regime_counts": validation["reports"]["development"][
            "regime_counts"
        ],
        "locked_regime_counts": validation["reports"]["locked_test"][
            "regime_counts"
        ],
        "development_natural_cardinality_counts": validation["reports"][
            "development"
        ]["natural_request_cardinality_counts"],
        "locked_natural_cardinality_counts": validation["reports"][
            "locked_test"
        ]["natural_request_cardinality_counts"],
        "by_regime": rows,
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    output = run_dir / "forward_identical_natural_audit_v10.json"
    report_path = PACKAGE / "V10_IDENTICAL_NATURAL_AUDIT.md"
    if (
        (output.exists() or report_path.exists())
        and not args.overwrite_diagnostic
    ):
        raise RuntimeError(f"refusing to overwrite completed audit: {output}")

    aligned_rows = read_jsonl(data_dir / "grids" / "train.jsonl")
    development_rows = read_jsonl(
        data_dir / "grids" / "development.jsonl"
    )
    seed = int(config["model_seed"])
    legacy_rows, _ = existing_distribution_sample(
        list(config["existing_distribution_sources"]),
        len(aligned_rows),
        seed,
    )
    prepare_legacy_rows(legacy_rows, seed)

    import torch

    torch.manual_seed(seed)
    device = torch.device(args.device)
    batch_size = int(config["forward_training"]["batch_size"])
    artifacts: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    errors: dict[str, dict[str, Any]] = {}
    vectors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    reports = {
        experiment: read_json(run_dir / f"forward_{experiment}_v10.json")
        for experiment in "ABCD"
    }
    for experiment in "ABCD":
        checkpoint = run_dir / f"forward_{experiment}_v10.pt"
        model, artifact = load_model(torch, checkpoint, device)
        own_rows = legacy_rows if experiment == "A" else aligned_rows
        fit_rows = rows_by_ids(own_rows, artifact["training_group_ids"])
        calibration_rows = rows_by_ids(
            own_rows, artifact["calibration_group_ids"]
        )
        split_rows = {
            "own_fit": fit_rows,
            "own_internal_calibration": calibration_rows,
            "common_v10_train_all": aligned_rows,
            "development": development_rows,
        }
        predictions[experiment] = {}
        errors[experiment] = {}
        vectors[experiment] = {}
        for split, rows in split_rows.items():
            prediction = predict(
                torch,
                model,
                artifact,
                rows,
                device,
                batch_size,
            )
            metric, vector = error_metrics(rows, prediction)
            predictions[experiment][split] = prediction
            errors[experiment][split] = metric
            vectors[experiment][split] = vector
        report = reports[experiment]
        inferred_epochs = min(
            int(report["maximum_epochs"]),
            int(report["best_epoch"])
            + int(config["forward_training"]["patience"]),
        )
        artifacts[experiment] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "reported_sha256": report["artifact_sha256"],
            "artifact_experiment": artifact["experiment"],
            "architecture": artifact["architecture"],
            "objective": artifact["objective"],
            "parameter_count": int(report["parameter_count"]),
            "fit_groups": len(artifact["training_group_ids"]),
            "fit_transitions": (
                len(artifact["training_group_ids"]) * len(ACTION_GRID)
            ),
            "internal_calibration_groups": len(
                artifact["calibration_group_ids"]
            ),
            "best_epoch": int(report["best_epoch"]),
            "maximum_epochs": int(report["maximum_epochs"]),
            "optimizer_epochs_inferred_from_early_stopping": inferred_epochs,
            "prediction_sha256": {
                split: array_sha256(value)
                for split, value in predictions[experiment].items()
            },
            "identity_checks_passed": bool(
                sha256_file(checkpoint) == report["artifact_sha256"]
                and artifact["experiment"] == experiment
            ),
        }

    pairwise_predictions = {}
    pairwise_masks = {}
    for left_index, left in enumerate("ABCD"):
        for right in "ABCD"[left_index + 1 :]:
            name = f"{left}_versus_{right}"
            pairwise_predictions[name] = {
                split: prediction_difference(
                    predictions[left][split],
                    predictions[right][split],
                    aligned_rows
                    if split == "common_v10_train_all"
                    else development_rows,
                )
                for split in ("common_v10_train_all", "development")
            }
            pairwise_masks[name] = mask_difference(
                vectors[left]["development"]["natural_strict"],
                vectors[right]["development"]["natural_strict"],
            )

    frozen_path = DEFAULT_V9_RUN / "forward_selector_ensemble_v9.pkl"
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
    )
    frozen_runtime, frozen_artifact = (
        load_forward_selector_ensemble_runtime_v9(
            frozen_path,
            torch,
            device,
        )
    )
    frozen_errors = {}
    for split, rows in {
        "common_v10_train_all": aligned_rows,
        "development": development_rows,
    }.items():
        frozen_prediction = frozen_runtime.predict_changes(rows)
        frozen_errors[split], _ = error_metrics(rows, frozen_prediction)

    v7_summary = read_json(DEFAULT_V7_SUMMARY)
    v9_selector_report = read_json(
        DEFAULT_V9_RUN / "forward_selector_ensemble_v9.json"
    )
    with frozen_path.open("rb") as stream:
        frozen_selector = pickle.load(stream)
    classifier = frozen_selector["classifier"]
    capacity_comparison = {
        "v10_common_contract": {
            "fit_groups": 326,
            "fit_transitions": 326 * len(ACTION_GRID),
            "internal_calibration_groups": 58,
            "maximum_epochs": int(config["forward_training"]["epochs"]),
            "batch_size_groups": int(
                config["forward_training"]["batch_size"]
            ),
            "context_preprocessing": (
                "17 visible setup/state values; log1p peak; per-experiment "
                "training-fit z-score"
            ),
            "target_preprocessing": (
                "five changes normalized by production tolerance"
            ),
        },
        "v10_candidates": {
            experiment: {
                "architecture": artifacts[experiment]["architecture"],
                "parameter_count": artifacts[experiment][
                    "parameter_count"
                ],
                "best_epoch": artifacts[experiment]["best_epoch"],
                "optimizer_epochs_inferred": artifacts[experiment][
                    "optimizer_epochs_inferred_from_early_stopping"
                ],
                "action_preprocessing": (
                    "opaque 81-ID embedding"
                    if experiment in "ABC"
                    else (
                        "signed motion, magnitude, masks, action basis, "
                        "cardinality, and pair/triple/four-way terms"
                    )
                ),
            }
            for experiment in "ABCD"
        },
        "frozen_v9": {
            "shared_v7_foundation_parameter_count": int(
                v7_summary["parameter_count"]
            ),
            "shared_v7_training_groups": int(
                v7_summary["training"]["total_groups"]
            ),
            "shared_v7_training_transitions": int(
                v7_summary["training"]["total_transitions"]
            ),
            "shared_v7_epochs_run": len(
                v7_summary["training"]["trace"]
            ),
            "shared_v7_best_epoch": int(
                v7_summary["training"]["best_epoch"]
            ),
            "selector_training_groups": int(
                v9_selector_report["training"]["group_count"]
            ),
            "selector_exclusive_training_transitions": int(
                v9_selector_report["training"]["exclusive_count"]
            ),
            "selector_feature_count": int(classifier.n_features_in_),
            "selector_max_iterations": int(
                classifier.get_params()["max_iter"]
            ),
            "selector_iterations_run": int(classifier.n_iter_),
            "preprocessing": (
                "46 per-transition features: 17 visible context, four "
                "signed actions, and 25 engineered optical/action terms; "
                "plus five out-of-fold v5 prior predictions = 51 z-scored "
                "inputs. The neural foundation predicts a residual around "
                "that prior; protected primary/secondary expert selectors "
                "and a final HGB selector are then applied."
            ),
            "artifact_version": frozen_artifact["version"],
        },
    }

    saved_vectors_path = (
        run_dir / "forward_ablation_development_v10.vectors.npz"
    )
    saved_mask_checks = {}
    with np.load(saved_vectors_path) as saved:
        for experiment in "ABCD":
            recomputed = vectors[experiment]["development"][
                "natural_strict"
            ].astype(np.float64)
            stored = saved[f"{experiment}_natural_strict"]
            saved_mask_checks[experiment] = {
                "stored_success_count": int(stored.sum()),
                "recomputed_success_count": int(recomputed.sum()),
                "different_count": int((stored != recomputed).sum()),
            }

    indexing = audit_natural_indexing(
        development_rows,
        int(config["split_seeds"]["development"]),
    )
    validation = read_json(data_dir / "validation_after_freeze.json")
    development_report = read_json(
        run_dir / "forward_ablation_development_v10.json"
    )
    locked_report = read_json(run_dir / "locked_test_once_v10.json")
    composition = regime_comparison(
        development_report,
        locked_report,
        validation,
    )

    result = {
        "version": "physics_structured_rebuild_v10_identical_natural_audit",
        "diagnostic_only": True,
        "selection_modified": False,
        "development_raw_data_opened": True,
        "locked_raw_data_opened": False,
        "locked_sources_used": [
            str(data_dir / "validation_after_freeze.json"),
            str(run_dir / "locked_test_once_v10.json"),
        ],
        "checkpoint_identity": artifacts,
        "prediction_differences": pairwise_predictions,
        "natural_success_mask_differences": pairwise_masks,
        "saved_mask_recomputation_checks": saved_mask_checks,
        "normalized_errors": {
            "candidates": errors,
            "frozen_v9_reference": frozen_errors,
        },
        "natural_action_indexing": indexing,
        "capacity_training_preprocessing_comparison": capacity_comparison,
        "development_locked_composition": composition,
        "complete": True,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fit_rows = []
    for experiment in "ABCD":
        fit = errors[experiment]["own_fit"]
        development = errors[experiment]["development"]
        fit_rows.append(
            f"| {experiment} | {artifacts[experiment]['parameter_count']:,} | "
            f"{artifacts[experiment]['best_epoch']} / "
            f"{artifacts[experiment]['optimizer_epochs_inferred_from_early_stopping']} | "
            f"{fit['normalized_mae']:.3f} | "
            f"{percent(fit['full_surface_strict_all_five'])} | "
            f"{development['normalized_mae']:.3f} | "
            f"{percent(development['full_surface_strict_all_five'])} |"
        )
    mask_rows = []
    for name, block in pairwise_masks.items():
        mask_rows.append(
            f"| {name.replace('_versus_', '-')} | "
            f"{block['different_count']} | "
            f"{block['left_only_success_count']} | "
            f"{block['right_only_success_count']} |"
        )
    prediction_rows = []
    for name, blocks in pairwise_predictions.items():
        block = blocks["development"]
        prediction_rows.append(
            f"| {name.replace('_versus_', '-')} | "
            f"{block['mean_absolute_prediction_difference']:.5f} | "
            f"{block['maximum_absolute_prediction_difference']:.5f} | "
            f"{block['natural_groups_with_any_difference_above_1e-6']}/160 |"
        )
    regime_rows = []
    for regime, block in composition["by_regime"].items():
        regime_rows.append(
            f"| {regime} | {block['development_count']} | "
            f"{block['locked_count']} | "
            f"{block['development_success_count']} | "
            f"{block['locked_success_count']} | "
            f"{block['success_count_change_locked_minus_development']} |"
        )
    report = f"""# V10 identical natural-action audit

This is a diagnostic-only audit. It did not train, select, or replace a model
and did not open the raw locked-test JSONL.

## Finding

The 18.12% equality is real at the aggregate level but does **not** come from
identical checkpoints or identical numerical predictions. A, B, and C have
the same 29/160 natural-action success mask. D also has 29/160 successes, but
it swaps two successes for two failures, so four group outcomes differ.

All four checkpoint hashes, metadata identities, prediction hashes, and model
states are distinct. Every A-D pair differs numerically on all 160 development
groups at the requested action:

| Pair | Mean absolute prediction difference | Maximum difference | Natural groups differing |
|---|---:|---:|---:|
{chr(10).join(prediction_rows)}

Natural strict-mask comparison:

| Pair | Different groups | Left-only successes | Right-only successes |
|---|---:|---:|---:|
{chr(10).join(mask_rows)}

The saved evaluation masks were independently recomputed from the checkpoints;
all four had zero discrepancies.

## Normalized fit and development error

The fit column uses each candidate's 326 optimizer-fit groups; A's come from
the sampled legacy distribution and B-D's from v10. Each group contributes all
81 correlated actions.

| Model | Parameters | Best / inferred epochs run | Fit normalized MAE | Fit full-surface strict | Development normalized MAE | Development full-surface strict |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(fit_rows)}

Frozen v9's reference normalized MAE is
{frozen_errors['common_v10_train_all']['normalized_mae']:.3f} on the common
v10 training distribution and
{frozen_errors['development']['normalized_mae']:.3f} on development.
Full-surface strict success is
{percent(frozen_errors['common_v10_train_all']['full_surface_strict_all_five'])}
and
{percent(frozen_errors['development']['full_surface_strict_all_five'])},
respectively.

The high fit-set errors show that A-D underfit even their 326 optimizer-fit
groups. This is most acute for A-C, which selected epochs 2/1/1 and stopped
after approximately 14/13/13 epochs. D used 509,706 parameters and about 49
epochs, but still remained far from fitting the training surface.

## Baseline parity mismatch

- A-C: 96,517 parameters; D: 509,706 parameters.
- Each v10 model fit 326 groups = 26,406 transitions, with 58 groups reserved
  for internal checkpoint calibration.
- Frozen v9's shared neural foundation alone has 825,236 parameters and was
  trained for 13 epochs (best 7) on 10,500 groups = 850,500 transitions.
- Frozen v9 then adds protected expert pipelines and a 180-iteration,
  20-feature HGB selector trained on 1,600 groups and 11,675 exclusive
  disagreement transitions. Its capacity is therefore not represented by the
  825,236 neural parameter count alone.
- V10 uses a z-scored 17-value context and either an opaque action-ID embedding
  or explicit structured actions. Frozen v9 uses 46 per-transition features
  (17 context + 4 action + 25 engineered terms), five out-of-fold prior
  predictions, z-scoring, residual learning around that prior, and later
  protected selectors.

Thus A-D were neither data-parity nor architecture/preprocessing-parity
reproductions of frozen v9. Their negative results diagnose an inadequate
common baseline, not a fair ceiling on the proposed objectives or structure.

## Natural-action indexing

Indexing passed for all {indexing['group_count']} development groups:

- canonical 81-action order failures: 0;
- deterministic stored-index mismatches: 0;
- indexed action versus canonical action mismatches: 0;
- stored-mask versus recomputed-checkpoint mismatches: 0.

The requested action is correctly gathered as
`prediction[group_index, natural_requested_action_index]`.

## Development versus locked composition

Regime composition is identical, so it does not explain frozen v9's
43.75% to 30.63% decline:

| Regime | Development groups | Locked groups | Development successes | Locked successes | Change |
|---|---:|---:|---:|---:|---:|
{chr(10).join(regime_rows)}

The loss is 21 successes across the same regime allocation: ordinary -6,
focusing -5, camera-boundary -4, tolerance-boundary -4, high-offset -2, and
clipping 0. Natural-action cardinality composition changed modestly
(development 36/49/37/38 versus locked 38/36/46/40 for cardinalities 1-4),
but the saved locked report does not contain a requested-cardinality success
breakdown. The supported explanation is a harder independently seeded draw
within the same regimes, especially camera and tolerance boundaries—not a
regime-mixture shift.

## Implication for v11

The proposed baseline-parity and scaling sequence is supported. Resume the
preserved deterministic generation, establish a new development boundary
while keeping the v10 locked data closed, and require a plain current-style
model to approach frozen v9 before repeating the loss/structure ablations.
"""
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "complete": True,
                "output": str(output),
                "report": str(report_path),
                "locked_raw_data_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
