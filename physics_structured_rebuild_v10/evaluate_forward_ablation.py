#!/usr/bin/env python3
"""Evaluate A-D and frozen v9 on one untouched system-like development set."""

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
    ACTION_NORMALIZED,
    explicit_action_features,
    group_targets,
    sha256_file,
    stable_seed,
    visible_context,
)
from physics_structured_rebuild_v10.evaluate import (
    forward_metrics,
    paired_difference_ci,
    request_identity,
)
from physics_structured_rebuild_v10.models import (
    build_forward_model,
    forward_call,
)
from physics_structured_rebuild_v9.forward_runtime import (
    load_forward_selector_ensemble_runtime_v9,
)

DEFAULT_CONFIG = Path(__file__).with_name("config_v10.json")
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/physics_structured_rebuild_v10"
DEFAULT_RUN = REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v10_pilot"
FROZEN_V9 = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_one_seed"
    / "forward_selector_ensemble_v9.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def predict_artifact(
    torch: Any,
    artifact_path: Path,
    contexts: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    architecture = str(artifact["architecture"])
    model = build_forward_model(
        torch,
        architecture,
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
    predictions = []
    uncertainties = []
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
            prediction, log_variance = forward_call(
                model,
                architecture,
                context,
                explicit,
                categories,
            )
            predictions.append(prediction.float().cpu().numpy())
            if log_variance is not None:
                uncertainties.append(log_variance.float().cpu().numpy())
    return (
        np.concatenate(predictions),
        None if not uncertainties else np.concatenate(uncertainties),
        artifact,
    )


def gate(
    candidate: dict[str, Any],
    candidate_vectors: dict[str, np.ndarray],
    a_vectors: dict[str, np.ndarray],
    v9_vectors: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    versus_a = paired_difference_ci(
        candidate_vectors["natural_strict"],
        a_vectors["natural_strict"],
        stable_seed(seed, candidate["experiment"], "A"),
        draws,
    )
    versus_v9 = paired_difference_ci(
        candidate_vectors["natural_strict"],
        v9_vectors["natural_strict"],
        stable_seed(seed, candidate["experiment"], "v9"),
        draws,
    )
    ordinary = np.asarray(
        [row["regime"] == "ordinary" for row in rows], dtype=bool
    )
    ordinary_difference = paired_difference_ci(
        candidate_vectors["natural_strict"][ordinary],
        v9_vectors["natural_strict"][ordinary],
        stable_seed(seed, candidate["experiment"], "ordinary"),
        draws,
    )
    candidate_fields = np.asarray(candidate_vectors["natural_fields"]).mean(axis=0)
    v9_fields = np.asarray(v9_vectors["natural_fields"]).mean(axis=0)
    maximum_field_regression = float(np.max(v9_fields - candidate_fields))
    passed = bool(
        versus_a["mean"] >= 0.03
        and versus_a["ci95_low"] > 0.0
        and versus_v9["ci95_low"] > 0.0
        and ordinary_difference["ci95_low"] >= -0.02
        and maximum_field_regression <= 0.03
    )
    return {
        "accepted": passed,
        "versus_A": versus_a,
        "versus_frozen_v9": versus_v9,
        "ordinary_versus_frozen_v9": ordinary_difference,
        "maximum_per_field_regression_versus_v9": maximum_field_regression,
        "criteria": {
            "minimum_mean_gain_versus_A": 0.03,
            "paired_lower_bound_versus_A_above": 0.0,
            "paired_lower_bound_versus_v9_above": 0.0,
            "ordinary_lower_bound_minimum": -0.02,
            "maximum_per_field_regression": 0.03,
        },
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    output = run_dir / (
        "forward_ablation_smoke_v10.json"
        if args.smoke
        else "forward_ablation_development_v10.json"
    )
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evaluation: {output}")
    rows = load_rows(data_dir / "grids" / "development.jsonl")
    if args.smoke:
        rows = rows[: min(4, len(rows))]
    contexts = np.asarray(
        [visible_context(row["setup"], row["current_beam_state"]) for row in rows],
        dtype=np.float32,
    )
    targets = np.asarray([group_targets(row)[1] for row in rows], dtype=np.float32)
    import torch

    device = torch.device(args.device)
    batch_size = int(config["forward_training"]["batch_size"])
    draws = 200 if args.smoke else int(config["bootstrap_draws"])
    metrics: dict[str, Any] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    artifacts: dict[str, Any] = {}
    for experiment in ("A", "B", "C", "D"):
        path = run_dir / f"forward_{experiment}_v10.pt"
        prediction, uncertainty, artifact = predict_artifact(
            torch,
            path,
            contexts,
            device,
            batch_size,
        )
        experiment_metrics, experiment_vectors = forward_metrics(
            rows,
            prediction,
            targets,
            uncertainty,
            stable_seed(int(config["bootstrap_seed"]), experiment),
            draws,
        )
        metrics[experiment] = experiment_metrics
        vectors[experiment] = experiment_vectors
        artifacts[experiment] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "contract": artifact["experiment_contract"],
        }
    frozen_runtime, frozen_artifact = load_forward_selector_ensemble_runtime_v9(
        FROZEN_V9,
        torch,
        device,
    )
    frozen_prediction = frozen_runtime.predict_changes(rows)
    frozen_metrics, frozen_vectors = forward_metrics(
        rows,
        frozen_prediction,
        targets,
        None,
        stable_seed(int(config["bootstrap_seed"]), "frozen_v9"),
        draws,
    )
    metrics["frozen_v9"] = frozen_metrics
    vectors["frozen_v9"] = frozen_vectors
    artifacts["frozen_v9"] = {
        "path": str(FROZEN_V9),
        "sha256": sha256_file(FROZEN_V9),
        "version": frozen_artifact["version"],
    }
    gates = {
        experiment: gate(
            {"experiment": experiment},
            vectors[experiment],
            vectors["A"],
            vectors["frozen_v9"],
            rows,
            int(config["bootstrap_seed"]),
            draws,
        )
        for experiment in ("B", "C", "D")
    }
    accepted = [
        experiment for experiment in ("B", "C", "D") if gates[experiment]["accepted"]
    ]
    selected = (
        max(
            accepted,
            key=lambda experiment: metrics[experiment]["natural_requested_action"][
                "strict_all_five"
            ]["mean"],
        )
        if accepted
        else "frozen_v9"
    )
    comparisons = {
        "distribution_alignment_B_minus_A": paired_difference_ci(
            vectors["B"]["natural_strict"],
            vectors["A"]["natural_strict"],
            stable_seed(int(config["bootstrap_seed"]), "B-A"),
            draws,
        ),
        "metric_alignment_C_minus_B": paired_difference_ci(
            vectors["C"]["natural_strict"],
            vectors["B"]["natural_strict"],
            stable_seed(int(config["bootstrap_seed"]), "C-B"),
            draws,
        ),
        "structured_model_D_minus_C": paired_difference_ci(
            vectors["D"]["natural_strict"],
            vectors["C"]["natural_strict"],
            stable_seed(int(config["bootstrap_seed"]), "D-C"),
            draws,
        ),
    }
    result = {
        "version": "physics_structured_rebuild_v10_forward_ablation",
        "scope": "system-like development only",
        "development_group_count": len(rows),
        "development_request_identity_sha256": request_identity(rows),
        "artifacts": artifacts,
        "metrics": metrics,
        "controlled_comparisons": comparisons,
        "acceptance_gates": gates,
        "selected_forward": selected,
        "locked_test_files_opened": [],
        "old_system_evaluation_files_opened": [],
        "smoke": bool(args.smoke),
        "seconds": time.perf_counter() - started,
        "complete": True,
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output.with_suffix(".vectors.npz"),
        **{
            f"{experiment}_{key}": value
            for experiment, experiment_vectors in vectors.items()
            for key, value in experiment_vectors.items()
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

