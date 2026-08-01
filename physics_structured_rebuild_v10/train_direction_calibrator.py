#!/usr/bin/env python3
"""Calibrate direction margins from the frozen forward selection."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v10.contracts import (
    ACTION_NORMALIZED,
    STATE_FIELDS,
    direction_labels,
    explicit_action_features,
    group_targets,
    sha256_file,
    stable_seed,
    visible_context,
)
from physics_structured_rebuild_v10.evaluate import mean_ci, paired_difference_ci
from physics_structured_rebuild_v10.models import (
    build_forward_model,
    forward_call,
)
from physics_structured_rebuild_v9.boundary_direction_runtime import (
    BoundaryDirectionCorrectionRuntimeV9,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)

DEFAULT_CONFIG = Path(__file__).with_name("config_v10.json")
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v10_full"
FROZEN_FORWARD = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_selector_ensemble_v9.pkl"
)
FROZEN_DIRECTION_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "boundary_direction_protected_calibrated_state_v9.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--forward-report", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def selected_forward_predictions(
    torch: Any,
    device: Any,
    selected: str,
    run_dir: Path,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if selected == "frozen_v9":
        runtime, artifact = load_forward_selector_ensemble_runtime_v9(
            FROZEN_FORWARD, torch, device
        )
        return runtime.predict_changes(rows), {
            "path": str(FROZEN_FORWARD),
            "sha256": sha256_file(FROZEN_FORWARD),
            "version": artifact["version"],
        }
    path = run_dir / f"forward_{selected}_v10.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = build_forward_model(
        torch,
        artifact["architecture"],
        artifact["model_config"],
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    explicit = torch.as_tensor(
        explicit_action_features(), dtype=torch.float32, device=device
    )
    categories = torch.as_tensor(
        (ACTION_NORMALIZED + 1).astype(np.int64),
        dtype=torch.long,
        device=device,
    )
    contexts = np.asarray(
        [visible_context(row["setup"], row["current_beam_state"]) for row in rows],
        dtype=np.float32,
    )
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
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
                artifact["architecture"],
                context,
                explicit,
                categories,
            )
            output.append(mean.float().cpu().numpy())
    return np.concatenate(output), {
        "path": str(path),
        "sha256": sha256_file(path),
        "version": artifact["version"],
    }


def build_calibrator(torch: Any, input_dim: int) -> Any:
    class DirectionCalibrator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.LayerNorm(128),
                torch.nn.GELU(),
                torch.nn.Dropout(0.05),
                torch.nn.Linear(128, 96),
                torch.nn.GELU(),
                torch.nn.Linear(96, 15),
            )

        def forward(self, values: Any) -> Any:
            return self.network(values).reshape(*values.shape[:-1], 5, 3)

    return DirectionCalibrator()


def features(predicted_change: np.ndarray) -> np.ndarray:
    action = explicit_action_features()
    broadcast = np.broadcast_to(
        action[None, :, :],
        (len(predicted_change), 81, action.shape[1]),
    )
    return np.concatenate(
        [
            predicted_change,
            np.abs(predicted_change),
            np.abs(np.abs(predicted_change) - 1.0),
            broadcast,
        ],
        axis=2,
    ).astype(np.float32)


def split_groups(rows: list[dict[str, Any]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(
        np.asarray(
            [stable_seed(seed, row["group_id"], "direction_split") for row in rows],
            dtype=np.uint64,
        )
    )
    calibration_count = max(1, int(round(0.15 * len(rows))))
    return order[calibration_count:], order[:calibration_count]


def label_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    seed: int,
    draws: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    exact = prediction == truth
    strict = exact.all(axis=2)
    group_success = strict.mean(axis=1)
    per_field = {}
    for field_index, field in enumerate(STATE_FIELDS):
        values = []
        for label in range(3):
            true_positive = ((prediction[:, :, field_index] == label) & (truth[:, :, field_index] == label)).sum()
            false_positive = ((prediction[:, :, field_index] == label) & (truth[:, :, field_index] != label)).sum()
            false_negative = ((prediction[:, :, field_index] != label) & (truth[:, :, field_index] == label)).sum()
            denominator = 2 * true_positive + false_positive + false_negative
            values.append(0.0 if denominator == 0 else float(2 * true_positive / denominator))
        per_field[field] = {
            "accuracy": float(exact[:, :, field_index].mean()),
            "macro_f1": float(np.mean(values)),
        }
    return {
        "strict_all_five": mean_ci(group_success, seed, draws),
        "strict_count": int(strict.sum()),
        "transition_count": int(strict.size),
        "per_field": per_field,
    }, {"group_success": group_success, "strict": strict.astype(np.float64)}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / (
        "direction_calibrator_smoke_v10.pt"
        if args.smoke
        else "direction_calibrator_v10.pt"
    )
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite direction output: {output}")
    forward_report_path = (
        args.forward_report.resolve()
        if args.forward_report is not None
        else run_dir / (
            "forward_ablation_smoke_v10.json"
            if args.smoke
            else "forward_ablation_development_v10.json"
        )
    )
    forward_report = json.loads(forward_report_path.read_text(encoding="utf-8"))
    selected = str(forward_report["selected_forward"])
    started = time.perf_counter()
    train_rows = read_jsonl(data_dir / "grids" / "train.jsonl")
    development_rows = read_jsonl(data_dir / "grids" / "development.jsonl")
    if args.smoke:
        train_rows = train_rows[: min(8, len(train_rows))]
        development_rows = development_rows[: min(4, len(development_rows))]
    seed = int(config["model_seed"]) + 300
    from control_rebuild_v3.train_forward import configure

    torch, device = configure(seed, args.device)
    train_forward, forward_artifact = selected_forward_predictions(
        torch, device, selected, run_dir, train_rows, 64
    )
    development_forward, _ = selected_forward_predictions(
        torch, device, selected, run_dir, development_rows, 64
    )
    train_truth = np.asarray(
        [direction_labels(group_targets(row)[1]) for row in train_rows],
        dtype=np.int64,
    )
    development_truth = np.asarray(
        [direction_labels(group_targets(row)[1]) for row in development_rows],
        dtype=np.int64,
    )
    train_features = features(train_forward)
    development_features = features(development_forward)
    training_groups, calibration_groups = split_groups(train_rows, seed)
    feature_mean = train_features[training_groups].reshape(-1, train_features.shape[-1]).mean(axis=0)
    feature_scale = train_features[training_groups].reshape(-1, train_features.shape[-1]).std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    model = build_calibrator(torch, train_features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    epochs = 2 if args.smoke else 50
    batch_size = min(24, len(training_groups))
    rng = np.random.default_rng(seed + 1)
    best_state = None
    best_key = None
    best_epoch = 0
    stale = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(training_groups)
        running = 0.0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            values = torch.as_tensor(
                (train_features[indices] - feature_mean) / feature_scale,
                dtype=torch.float32,
                device=device,
            )
            truth = torch.as_tensor(
                train_truth[indices], dtype=torch.long, device=device
            )
            logits = model(values)
            field_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3),
                truth.reshape(-1),
                reduction="none",
            ).reshape(len(indices), 81, 5)
            boundary_weight = 1.0 + 0.8 * (
                truth == 1
            ).float()
            joint = torch.logsumexp(field_loss * 3.0, dim=2) / 3.0
            loss = (field_loss * boundary_weight).mean() + 0.25 * joint.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(indices)
        model.eval()
        with torch.inference_mode():
            logits = model(
                torch.as_tensor(
                    (train_features[calibration_groups] - feature_mean) / feature_scale,
                    dtype=torch.float32,
                    device=device,
                )
            )
            prediction = logits.argmax(dim=-1).cpu().numpy()
        strict = (prediction == train_truth[calibration_groups]).all(axis=2)
        key = (int(strict.sum()),)
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
                "training_loss": running / len(training_groups),
                "calibration_strict_all_five": float(strict.mean()),
                "best_epoch": best_epoch,
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if not args.smoke and stale >= 8:
            break
    if best_state is None:
        raise RuntimeError("direction training did not select a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        calibrated = model(
            torch.as_tensor(
                (development_features - feature_mean) / feature_scale,
                dtype=torch.float32,
                device=device,
            )
        ).argmax(dim=-1).cpu().numpy()
    direct = direction_labels(development_forward)
    frozen_runtime = BoundaryDirectionCorrectionRuntimeV9(
        torch, FROZEN_DIRECTION_OVERLAY, device
    )
    _, _, _, frozen_flat = frozen_runtime.base_grid(development_rows)
    frozen = frozen_flat.reshape(len(development_rows), 81, 5)
    draws = 200 if args.smoke else int(config["bootstrap_draws"])
    blocks = {}
    vectors = {}
    for name, prediction in {
        "frozen_separate_direction": frozen,
        "direct_forward_thresholding": direct,
        "forward_plus_calibration_head": calibrated,
    }.items():
        blocks[name], vectors[name] = label_metrics(
            prediction,
            development_truth,
            stable_seed(int(config["bootstrap_seed"]), name),
            draws,
        )
    difference = paired_difference_ci(
        vectors["forward_plus_calibration_head"]["group_success"],
        vectors["direct_forward_thresholding"]["group_success"],
        stable_seed(int(config["bootstrap_seed"]), "direction_calibration"),
        draws,
    )
    calibrated_fields = np.asarray(
        [
            blocks["forward_plus_calibration_head"]["per_field"][field]["accuracy"]
            for field in STATE_FIELDS
        ]
    )
    direct_fields = np.asarray(
        [
            blocks["direct_forward_thresholding"]["per_field"][field]["accuracy"]
            for field in STATE_FIELDS
        ]
    )
    max_regression = float(np.max(direct_fields - calibrated_fields))
    accepted = bool(
        difference["mean"] >= 0.02
        and difference["ci95_low"] > 0.0
        and max_regression <= 0.02
    )
    selected_direction = (
        "forward_plus_calibration_head"
        if accepted
        else max(
            ("frozen_separate_direction", "direct_forward_thresholding"),
            key=lambda name: blocks[name]["strict_all_five"]["mean"],
        )
    )
    artifact = {
        "version": "physics_structured_rebuild_v10_direction_calibrator",
        "state_dict": best_state,
        "input_dim": int(train_features.shape[-1]),
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "selected_forward": selected,
        "selected_forward_artifact": forward_artifact,
        "seed": seed,
        "best_epoch": best_epoch,
        "locked_test_files_opened": [],
    }
    torch.save(artifact, output)
    result = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "selected_forward": selected,
        "development": blocks,
        "calibration_head_versus_direct": difference,
        "maximum_field_regression": max_regression,
        "acceptance": {
            "accepted": accepted,
            "minimum_mean_gain": 0.02,
            "paired_lower_bound_above": 0.0,
            "maximum_field_regression": 0.02,
        },
        "selected_direction": selected_direction,
        "separate_joint_model_run": False,
        "separate_joint_model_reason": (
            "the preregistered comparison among the frozen specialist, direct "
            "forward thresholding, and one calibration head is sufficient for "
            "this controlled cycle; no broad fallback search was launched"
        ),
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

