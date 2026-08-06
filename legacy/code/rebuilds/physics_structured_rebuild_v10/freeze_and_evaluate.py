#!/usr/bin/env python3
"""Freeze v10 decisions, then perform the separately invoked one-time locked run."""

from __future__ import annotations

import argparse
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
    STATE_FIELDS,
    direction_labels,
    explicit_action_features,
    group_targets,
    sha256_file,
    stable_seed,
    visible_context,
)
from physics_structured_rebuild_v10.evaluate import forward_metrics
from physics_structured_rebuild_v10.evaluate_forward_ablation import (
    FROZEN_V9,
    predict_artifact,
)
from physics_structured_rebuild_v10.models import (
    build_inverse_ranker,
    build_visual_inverse_ranker,
)
from physics_structured_rebuild_v10.train_direction_calibrator import (
    FROZEN_DIRECTION_OVERLAY,
    build_calibrator,
    features as direction_features,
    label_metrics,
    selected_forward_predictions,
)
from physics_structured_rebuild_v10.train_inverse_ranker import (
    frozen_v9_scores,
    ranking_metrics,
    request_arrays,
    score_batches as score_inverse_batches,
)
from physics_structured_rebuild_v10.train_visual_inverse import (
    frozen_visual_scores,
    metric_block as visual_metric_block,
    score_batches as score_visual_batches,
    visual_arrays,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "evaluate-locked"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def artifact_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size": resolved.stat().st_size,
    }


def freeze(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    output = run_dir / "freeze_manifest_v10.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite freeze manifest: {output}")
    forward = read_json(run_dir / "forward_ablation_development_v10.json")
    inverse = read_json(run_dir / "inverse_ranker_v10.json")
    visual = read_json(run_dir / "visual_inverse_v10.json")
    direction = read_json(run_dir / "direction_calibrator_v10.json")
    selected_forward = str(forward["selected_forward"])
    selected_inverse = (
        "direct_v10"
        if inverse["acceptance"]["accepted"]
        else "frozen_v9"
    )
    selected_visual = (
        "direct_v10"
        if visual["acceptance"]["accepted"]
        else "frozen_v9_direct_visual_scorer"
    )
    selected_direction = str(direction["selected_direction"])
    component_artifacts = {
        "forward": (
            artifact_entry(FROZEN_V9)
            if selected_forward == "frozen_v9"
            else artifact_entry(run_dir / f"forward_{selected_forward}_v10.pt")
        ),
        "inverse": (
            artifact_entry(run_dir / "inverse_ranker_v10.pt")
            if selected_inverse == "direct_v10"
            else artifact_entry(
                REPO_ROOT.parent
                / "VLM_runs/physics_structured_rebuild_v9_one_seed"
                / "combined_natural_inverse_ranker_adaptation/inverse.pt"
            )
        ),
        "visual": (
            artifact_entry(run_dir / "visual_inverse_v10.pt")
            if selected_visual == "direct_v10"
            else artifact_entry(
                REPO_ROOT.parent
                / "VLM_runs/control_rebuild_v4_one_seed"
                / "visual_sensor_scorer_v4.pt"
            )
        ),
        "direction": (
            artifact_entry(run_dir / "direction_calibrator_v10.pt")
            if selected_direction == "forward_plus_calibration_head"
            else artifact_entry(FROZEN_DIRECTION_OVERLAY)
            if selected_direction == "frozen_separate_direction"
            else (
                artifact_entry(FROZEN_V9)
                if selected_forward == "frozen_v9"
                else artifact_entry(
                    run_dir / f"forward_{selected_forward}_v10.pt"
                )
            )
        ),
    }
    reports = {
        name: artifact_entry(path)
        for name, path in {
            "forward": run_dir / "forward_ablation_development_v10.json",
            "inverse": run_dir / "inverse_ranker_v10.json",
            "visual": run_dir / "visual_inverse_v10.json",
            "direction": run_dir / "direction_calibrator_v10.json",
        }.items()
    }
    rejected_forward = [
        experiment
        for experiment, gate in forward["acceptance_gates"].items()
        if not gate["accepted"]
    ]
    result = {
        "version": "physics_structured_rebuild_v10_freeze_manifest",
        "frozen_at_unix_seconds": time.time(),
        "selected_components": {
            "forward": selected_forward,
            "numerical_inverse": selected_inverse,
            "visual_inverse": selected_visual,
            "direction": selected_direction,
        },
        "component_artifacts": component_artifacts,
        "selection_reports": reports,
        "rejected_forward_candidates": rejected_forward,
        "candidate_acceptance": {
            "inverse": bool(inverse["acceptance"]["accepted"]),
            "visual": bool(visual["acceptance"]["accepted"]),
            "direction_calibrator": bool(direction["acceptance"]["accepted"]),
        },
        "configuration": artifact_entry(args.config.resolve()),
        "dataset_manifest": artifact_entry(data_dir / "manifest.json"),
        "split_hashes": read_json(data_dir / "manifest.json")["split_summary"],
        "locked_test_evaluation_count_before_freeze": 0,
        "locked_test_files_opened_for_selection": [],
        "old_system_evaluation_files_opened_for_selection": [],
        "model_choices_frozen": True,
        "complete": True,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_locked(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    freeze_path = run_dir / "freeze_manifest_v10.json"
    output = run_dir / "locked_test_once_v10.json"
    if output.exists():
        raise RuntimeError(
            "locked-test output already exists; v10 permits exactly one run"
        )
    freeze_manifest = read_json(freeze_path)
    if not freeze_manifest.get("model_choices_frozen"):
        raise RuntimeError("model choices are not frozen")
    started = time.perf_counter()
    config = read_json(args.config.resolve())
    rows = read_jsonl(data_dir / "grids" / "locked_test.jsonl")
    import torch

    device = torch.device(args.device)
    contexts = np.asarray(
        [visible_context(row["setup"], row["current_beam_state"]) for row in rows],
        dtype=np.float32,
    )
    targets = np.asarray([group_targets(row)[1] for row in rows], dtype=np.float32)
    selected_forward = freeze_manifest["selected_components"]["forward"]
    if selected_forward == "frozen_v9":
        selected_runtime, _ = load_forward_selector_ensemble_runtime_v9(
            FROZEN_V9, torch, device
        )
        selected_forward_prediction = selected_runtime.predict_changes(rows)
        selected_forward_uncertainty = None
    else:
        (
            selected_forward_prediction,
            selected_forward_uncertainty,
            _,
        ) = predict_artifact(
            torch,
            run_dir / f"forward_{selected_forward}_v10.pt",
            contexts,
            device,
            int(config["forward_training"]["batch_size"]),
        )
    frozen_runtime, _ = load_forward_selector_ensemble_runtime_v9(
        FROZEN_V9, torch, device
    )
    frozen_forward_prediction = frozen_runtime.predict_changes(rows)
    draws = int(config["bootstrap_draws"])
    forward_selected_metrics, _ = forward_metrics(
        rows,
        selected_forward_prediction,
        targets,
        selected_forward_uncertainty,
        stable_seed(int(config["bootstrap_seed"]), "locked_forward_selected"),
        draws,
    )
    forward_v9_metrics, _ = forward_metrics(
        rows,
        frozen_forward_prediction,
        targets,
        None,
        stable_seed(int(config["bootstrap_seed"]), "locked_forward_v9"),
        draws,
    )

    inverse_requests = request_arrays(rows)
    selected_inverse = freeze_manifest["selected_components"]["numerical_inverse"]
    if selected_inverse == "direct_v10":
        inverse_artifact = torch.load(
            run_dir / "inverse_ranker_v10.pt",
            map_location="cpu",
            weights_only=False,
        )
        inverse_model = build_inverse_ranker(
            torch, inverse_artifact["model_config"]
        ).to(device)
        inverse_model.load_state_dict(inverse_artifact["state_dict"])
        inverse_actions = torch.as_tensor(
            explicit_action_features(), dtype=torch.float32, device=device
        )
        selected_inverse_scores = score_inverse_batches(
            torch,
            inverse_model,
            inverse_requests["features"],
            np.asarray(inverse_artifact["feature_mean"]),
            np.asarray(inverse_artifact["feature_scale"]),
            inverse_actions,
            device,
            int(config["inverse_training"]["batch_size"]),
        )
    else:
        selected_inverse_scores = frozen_v9_scores(
            torch, device, rows, inverse_requests
        )
    frozen_inverse_scores = frozen_v9_scores(
        torch, device, rows, inverse_requests
    )
    inverse_selected_metrics, _ = ranking_metrics(
        selected_inverse_scores,
        inverse_requests["labels"],
        inverse_requests["group_index"],
        rows,
        stable_seed(int(config["bootstrap_seed"]), "locked_inverse_selected"),
        draws,
    )
    inverse_v9_metrics, _ = ranking_metrics(
        frozen_inverse_scores,
        inverse_requests["labels"],
        inverse_requests["group_index"],
        rows,
        stable_seed(int(config["bootstrap_seed"]), "locked_inverse_v9"),
        draws,
    )

    visual = visual_arrays(rows, data_dir)
    selected_visual = freeze_manifest["selected_components"]["visual_inverse"]
    frozen_visual, frozen_measured = frozen_visual_scores(
        torch,
        device,
        (
            REPO_ROOT.parent
            / "VLM_runs/direction_rebuild_v4_tree_one_seed"
            / "candidate_overlay_direction_v4_manifest.json"
        ),
        rows,
        visual,
        data_dir,
    )
    if selected_visual == "direct_v10":
        visual_artifact = torch.load(
            run_dir / "visual_inverse_v10.pt",
            map_location="cpu",
            weights_only=False,
        )
        visual_model = build_visual_inverse_ranker(
            torch, visual_artifact["model_config"]
        ).to(device)
        visual_model.load_state_dict(visual_artifact["state_dict"])
        visual_actions = torch.as_tensor(
            explicit_action_features(), dtype=torch.float32, device=device
        )
        selected_visual_scores = score_visual_batches(
            torch,
            visual_model,
            visual,
            np.arange(len(visual["labels"])),
            np.asarray(visual_artifact["context_mean"]),
            np.asarray(visual_artifact["context_scale"]),
            visual_actions,
            device,
            int(config["visual_training"]["batch_size"]),
        )
    else:
        selected_visual_scores = frozen_visual
    visual_blocks = {}
    for name, scores in {
        "selected": selected_visual_scores,
        "frozen_v9_direct": frozen_visual,
        "frozen_v9_measured_state": frozen_measured,
    }.items():
        visual_blocks[name], _ = visual_metric_block(
            scores,
            visual["labels"],
            visual["group_index"],
            visual["condition"],
            stable_seed(int(config["bootstrap_seed"]), "locked_visual", name),
            draws,
        )

    direction_truth = np.asarray(
        [direction_labels(group_targets(row)[1]) for row in rows],
        dtype=np.int64,
    )
    selected_direction = freeze_manifest["selected_components"]["direction"]
    if selected_direction == "forward_plus_calibration_head":
        artifact = torch.load(
            run_dir / "direction_calibrator_v10.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = build_calibrator(torch, int(artifact["input_dim"])).to(device)
        model.load_state_dict(artifact["state_dict"])
        change, _ = selected_forward_predictions(
            torch,
            device,
            str(artifact["selected_forward"]),
            run_dir,
            rows,
            64,
        )
        values = direction_features(change)
        with torch.inference_mode():
            selected_direction_prediction = model(
                torch.as_tensor(
                    (values - np.asarray(artifact["feature_mean"]))
                    / np.asarray(artifact["feature_scale"]),
                    dtype=torch.float32,
                    device=device,
                )
            ).argmax(dim=-1).cpu().numpy()
    elif selected_direction == "direct_forward_thresholding":
        selected_direction_prediction = direction_labels(
            selected_forward_prediction
        )
    else:
        frozen_direction_runtime = BoundaryDirectionCorrectionRuntimeV9(
            torch, FROZEN_DIRECTION_OVERLAY, device
        )
        _, _, _, values = frozen_direction_runtime.base_grid(rows)
        selected_direction_prediction = values.reshape(len(rows), 81, 5)
    frozen_direction_runtime = BoundaryDirectionCorrectionRuntimeV9(
        torch, FROZEN_DIRECTION_OVERLAY, device
    )
    _, _, _, frozen_values = frozen_direction_runtime.base_grid(rows)
    frozen_direction_prediction = frozen_values.reshape(len(rows), 81, 5)
    direction_selected_metrics, _ = label_metrics(
        selected_direction_prediction,
        direction_truth,
        stable_seed(int(config["bootstrap_seed"]), "locked_direction_selected"),
        draws,
    )
    direction_v9_metrics, _ = label_metrics(
        frozen_direction_prediction,
        direction_truth,
        stable_seed(int(config["bootstrap_seed"]), "locked_direction_v9"),
        draws,
    )
    result = {
        "version": "physics_structured_rebuild_v10_locked_test_once",
        "freeze_manifest": artifact_entry(freeze_path),
        "locked_dataset": artifact_entry(data_dir / "grids/locked_test.jsonl"),
        "group_count": len(rows),
        "evaluation_count": 1,
        "selected_components": freeze_manifest["selected_components"],
        "forward": {
            "selected": forward_selected_metrics,
            "frozen_v9": forward_v9_metrics,
        },
        "numerical_inverse": {
            "selected": inverse_selected_metrics,
            "frozen_v9": inverse_v9_metrics,
        },
        "visual_inverse": visual_blocks,
        "direction": {
            "selected": direction_selected_metrics,
            "frozen_v9": direction_v9_metrics,
        },
        "selection_modified_after_locked_test": False,
        "old_150_case_system_evaluation_opened": False,
        "seconds": time.perf_counter() - started,
        "complete": True,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.mode == "freeze":
        freeze(args)
    else:
        evaluate_locked(args)


if __name__ == "__main__":
    main()

