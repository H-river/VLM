#!/usr/bin/env python3
"""Train direct image-primary physical-success scoring over all 81 actions."""

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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.train_forward import configure
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from physics_structured_rebuild_v10.contracts import (
    explicit_action_features,
    group_targets,
    physical_success,
    sha256_file,
    stable_seed,
    visible_context,
)
from physics_structured_rebuild_v10.evaluate import mean_ci, paired_difference_ci
from physics_structured_rebuild_v10.models import build_visual_inverse_ranker
from physics_structured_rebuild_v10.train_inverse_ranker import listwise_loss
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

DEFAULT_CONFIG = Path(__file__).with_name("config_v10.json")
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v10_full"
DEFAULT_OVERLAY = (
    REPO_ROOT.parent
    / "VLM_runs/direction_rebuild_v4_tree_one_seed"
    / "candidate_overlay_direction_v4_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--v4-overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return array[None, :, :]


def visual_arrays(
    rows: list[dict[str, Any]],
    data_dir: Path,
) -> dict[str, Any]:
    current_images = []
    target_images = []
    contexts = []
    labels = []
    group_index = []
    condition = []
    requests = []
    for row_index, row in enumerate(rows):
        states, _ = group_targets(row)
        context = visible_context(row["setup"], row["current_beam_state"])
        for request in row["visual_requests"]:
            target = np.asarray(
                [
                    float(request["desired_beam_state"][field])
                    for field in (
                        "centroid_x_px",
                        "centroid_y_px",
                        "sigma_x_px",
                        "sigma_y_px",
                        "peak_intensity",
                    )
                ],
                dtype=np.float32,
            )
            current_images.append(load_image(data_dir / request["current_image"]))
            target_images.append(load_image(data_dir / request["target_image"]))
            contexts.append(context)
            labels.append(physical_success(states, target))
            group_index.append(row_index)
            condition.append(str(request["condition"]))
            requests.append(request)
    return {
        "current_images": np.asarray(current_images, dtype=np.float32),
        "target_images": np.asarray(target_images, dtype=np.float32),
        "contexts": np.asarray(contexts, dtype=np.float32),
        "labels": np.asarray(labels, dtype=bool),
        "group_index": np.asarray(group_index, dtype=np.int64),
        "condition": np.asarray(condition, dtype=np.str_),
        "requests": requests,
    }


def group_partition(rows: list[dict[str, Any]], seed: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(
        [stable_seed(seed, row["group_id"], "visual_split") for row in rows],
        dtype=np.uint64,
    )
    order = np.argsort(tokens)
    calibration_count = max(1, int(round(0.15 * len(rows))))
    return order[calibration_count:], order[:calibration_count]


def score_batches(
    torch: Any,
    model: Any,
    arrays: dict[str, Any],
    indices: np.ndarray,
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    actions: Any,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            output.append(
                model(
                    torch.as_tensor(
                        arrays["current_images"][chosen],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        arrays["target_images"][chosen],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        (arrays["contexts"][chosen] - context_mean) / context_scale,
                        dtype=torch.float32,
                        device=device,
                    ),
                    actions,
                )
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(output)


def group_success(
    request_success: np.ndarray,
    group_index: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            request_success[group_index == group].mean()
            for group in np.unique(group_index)
        ],
        dtype=np.float64,
    )


def metric_block(
    scores: np.ndarray,
    labels: np.ndarray,
    group_index: np.ndarray,
    condition: np.ndarray,
    seed: int,
    draws: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    order = np.argsort(-scores, axis=1)
    ordered = np.take_along_axis(labels, order, axis=1)
    top1 = ordered[:, 0]
    by_condition = {}
    for name in sorted(set(condition.tolist())):
        mask = condition == name
        by_condition[name] = {
            "count": int(mask.sum()),
            "top1_physical_success": float(top1[mask].mean()),
        }
    group_top1 = group_success(top1, group_index)
    return {
        "top1_physical_success": mean_ci(group_top1, seed, draws),
        "request_success_count": int(top1.sum()),
        "request_count": int(len(top1)),
        "top_k_success": {
            str(k): float(ordered[:, :k].any(axis=1).mean())
            for k in (1, 3, 5, 10)
        },
        "mean_first_success_rank": float((ordered.argmax(axis=1) + 1).mean()),
        "by_image_condition": by_condition,
    }, {"request_top1": top1.astype(np.float64), "group_top1": group_top1}


def frozen_visual_scores(
    torch: Any,
    device: Any,
    overlay: Path,
    rows: list[dict[str, Any]],
    arrays: dict[str, Any],
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    backend = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        overlay.resolve(),
        device,
    )
    natural_forward, _ = load_residual_forward_runtime_v9(
        DEFAULT_NATURAL_GRID_FORWARD_STATE,
        torch,
        device,
    )
    adapted_inverse, _ = load_inverse_runtime_v8(
        DEFAULT_COMBINED_NATURAL_INVERSE_V9,
        torch,
        device,
    )
    backend.visual.forward = natural_forward
    backend.visual.inverse = adapted_inverse
    direct_selected = []
    measured_state_selected = []
    request_offset = 0
    for row in rows:
        for request in row["visual_requests"]:
            result = backend.visual.predict_from_images(
                row["setup"],
                data_dir / request["current_image"],
                data_dir / request["target_image"],
                request["image_calibration"],
                request_id=str(request["request_id"]),
            )
            direct_selected.append(int(result["selected_indices"][0]))
            measured_state_selected.append(
                int(result["numerical_selected_indices"][0])
            )
            request_offset += 1
            if request_offset % 50 == 0:
                print(
                    json.dumps(
                        {
                            "frozen_visual_requests": request_offset,
                            "total": len(arrays["labels"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    direct = np.full((len(direct_selected), 81), -1e6, dtype=np.float32)
    numerical = np.full_like(direct, -1e6)
    direct[np.arange(len(direct_selected)), direct_selected] = 1e6
    numerical[
        np.arange(len(measured_state_selected)), measured_state_selected
    ] = 1e6
    return direct, numerical


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / (
        "visual_inverse_smoke_v10.pt" if args.smoke else "visual_inverse_v10.pt"
    )
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite visual output: {output}")
    started = time.perf_counter()
    train_rows = read_jsonl(data_dir / "grids" / "train.jsonl")
    development_rows = read_jsonl(data_dir / "grids" / "development.jsonl")
    if args.smoke:
        train_rows = train_rows[: min(8, len(train_rows))]
        development_rows = development_rows[: min(4, len(development_rows))]
    train = visual_arrays(train_rows, data_dir)
    development = visual_arrays(development_rows, data_dir)
    seed = int(config["model_seed"]) + 200
    torch, device = configure(seed, args.device)
    training_groups, calibration_groups = group_partition(train_rows, seed)
    training = np.flatnonzero(np.isin(train["group_index"], training_groups))
    calibration = np.flatnonzero(np.isin(train["group_index"], calibration_groups))
    context_mean = train["contexts"][training].mean(axis=0)
    context_scale = train["contexts"][training].std(axis=0)
    context_scale[context_scale < 1e-6] = 1.0
    training_config = config["visual_training"]
    model_config = {
        "context_dim": int(train["contexts"].shape[1]),
        "action_dim": int(explicit_action_features().shape[1]),
        "dimension": int(training_config["dimension"]),
        "dropout": float(training_config["dropout"]),
    }
    model = build_visual_inverse_ranker(torch, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    actions = torch.as_tensor(
        explicit_action_features(), dtype=torch.float32, device=device
    )
    epochs = int(training_config["smoke_epochs"] if args.smoke else training_config["epochs"])
    batch_size = min(int(training_config["batch_size"]), len(training))
    rng = np.random.default_rng(seed + 1)
    best_state = None
    best_key = None
    best_epoch = 0
    stale = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(training)
        running = 0.0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            scores = model(
                torch.as_tensor(train["current_images"][indices], dtype=torch.float32, device=device),
                torch.as_tensor(train["target_images"][indices], dtype=torch.float32, device=device),
                torch.as_tensor(
                    (train["contexts"][indices] - context_mean) / context_scale,
                    dtype=torch.float32,
                    device=device,
                ),
                actions,
            )
            positives = torch.as_tensor(
                train["labels"][indices], dtype=torch.bool, device=device
            )
            loss = listwise_loss(
                torch,
                scores,
                positives,
                torch.zeros_like(positives),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(indices)
        calibration_scores = score_batches(
            torch,
            model,
            train,
            calibration,
            context_mean,
            context_scale,
            actions,
            device,
            batch_size,
        )
        selected = calibration_scores.argmax(axis=1)
        success = train["labels"][calibration, selected]
        key = (int(success.sum()),)
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
        raise RuntimeError("visual training did not select a checkpoint")
    model.load_state_dict(best_state)
    development_indices = np.arange(len(development["labels"]))
    candidate_scores = score_batches(
        torch,
        model,
        development,
        development_indices,
        context_mean,
        context_scale,
        actions,
        device,
        batch_size,
    )
    frozen_direct_scores, frozen_measured_scores = frozen_visual_scores(
        torch,
        device,
        args.v4_overlay,
        development_rows,
        development,
        data_dir,
    )
    draws = 200 if args.smoke else int(config["bootstrap_draws"])
    blocks = {}
    vectors = {}
    for name, scores in {
        "candidate_direct_image": candidate_scores,
        "frozen_v9_direct_visual_scorer": frozen_direct_scores,
        "frozen_v9_measured_state_numerical_route": frozen_measured_scores,
    }.items():
        blocks[name], vectors[name] = metric_block(
            scores,
            development["labels"],
            development["group_index"],
            development["condition"],
            stable_seed(int(config["bootstrap_seed"]), name),
            draws,
        )
    difference = paired_difference_ci(
        vectors["candidate_direct_image"]["group_top1"],
        vectors["frozen_v9_direct_visual_scorer"]["group_top1"],
        stable_seed(int(config["bootstrap_seed"]), "visual_difference"),
        draws,
    )
    clean = development["condition"] == "clean"
    candidate_clean = float(
        vectors["candidate_direct_image"]["request_top1"][clean].mean()
    ) if clean.any() else None
    frozen_clean = float(
        vectors["frozen_v9_direct_visual_scorer"]["request_top1"][clean].mean()
    ) if clean.any() else None
    clean_regression = (
        0.0
        if candidate_clean is None or frozen_clean is None
        else frozen_clean - candidate_clean
    )
    accepted = bool(
        difference["mean"] >= 0.03
        and difference["ci95_low"] > 0.0
        and clean_regression <= 0.02
    )
    artifact = {
        "version": "physics_structured_rebuild_v10_direct_visual_inverse",
        "model_config": model_config,
        "state_dict": best_state,
        "context_mean": context_mean,
        "context_scale": context_scale,
        "seed": seed,
        "best_epoch": best_epoch,
        "image_primary": True,
        "measured_values_are_auxiliary_only": True,
        "locked_test_files_opened": [],
    }
    torch.save(artifact, output)
    result = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "development": {
            **blocks,
            "paired_candidate_minus_frozen_direct": difference,
            "clean_condition": {
                "candidate": candidate_clean,
                "frozen_direct": frozen_clean,
                "regression": clean_regression,
            },
        },
        "acceptance": {
            "accepted": accepted,
            "minimum_mean_gain": 0.03,
            "paired_lower_bound_above": 0.0,
            "maximum_clean_regression": 0.02,
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

