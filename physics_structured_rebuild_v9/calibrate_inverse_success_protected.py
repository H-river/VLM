#!/usr/bin/env python3
"""Calibrate inverse-success gates on one half and confirm on the other."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID, MOVEMENT
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.inverse_lightgbm_runtime import (
    normalized_per_request,
    ranker_features,
)
from physics_structured_rebuild_v9.inverse_runtime_v8 import (
    load_inverse_runtime_v8,
)
from physics_structured_rebuild_v9.inverse_success_runtime import (
    apply_gate,
    sha256,
    sigmoid,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_DATA = REPO_ROOT.parent / "VLM_data/tabm_transformer_inverse_v8"
DEFAULT_INPUT = DEFAULT_RUN / "inverse_success_ranker_v9.pkl"
DEFAULT_OUTPUT = (
    DEFAULT_RUN / "inverse_success_ranker_protected_calibrated_v9.pkl"
)
BLOCKS = ("iid_clean", "difficult_clean")
QUALITY_WEIGHTS = (0.0, 0.25, 0.5, 1.0)
MODEL_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
CANDIDATE_PROBABILITIES = (0.0, 0.2, 0.4, 0.6, 0.8)
PROBABILITY_ADVANTAGES = (-0.2, -0.1, 0.0, 0.1, 0.2)
BASE_PROBABILITIES = (1.0, 0.8, 0.6, 0.4)
MODEL_SCORE_ADVANTAGES = (-0.5, -0.2, 0.0, 0.2, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def success_metrics(
    positives: np.ndarray,
    base_indices: np.ndarray,
    selected: np.ndarray,
    changed: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    positions = np.arange(len(positives))
    base_success = positives[positions, base_indices]
    selected_success = positives[positions, selected]
    chosen = np.asarray(indices, dtype=np.int64)
    return {
        "count": int(len(chosen)),
        "base_success_count": int(base_success[chosen].sum()),
        "candidate_success_count": int(selected_success[chosen].sum()),
        "gain": int(
            selected_success[chosen].sum() - base_success[chosen].sum()
        ),
        "changed_count": int(changed[chosen].sum()),
        "candidate_only_success_count": int(
            (
                selected_success[chosen]
                & ~base_success[chosen]
            ).sum()
        ),
        "base_only_success_count": int(
            (
                base_success[chosen]
                & ~selected_success[chosen]
            ).sum()
        ),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    input_path = args.input.resolve()
    with input_path.open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("model") != "group_balanced_inverse_success_v9":
        raise ValueError("unexpected inverse-success artifact")
    torch, device = configure(int(args.seed), args.device)
    base_path = Path(str(artifact["base_inverse_artifact"])).resolve()
    base, _ = load_inverse_runtime_v8(base_path, torch, device)

    blocks: dict[str, dict[str, Any]] = {}
    for block_name in BLOCKS:
        path = args.data_dir.resolve() / f"{block_name}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            contexts = np.asarray(arrays["contexts"], dtype=np.float32)
            desired = np.asarray(arrays["desired"], dtype=np.float32)
            states = np.asarray(
                arrays["candidate_states"],
                dtype=np.float32,
            )
            positives = np.asarray(arrays["positives"], dtype=np.bool_)
        base_result = base.score_feature_arrays(
            contexts,
            desired,
            states,
        )
        base_scores = np.asarray(base_result["scores"], dtype=np.float32)
        base_indices = np.asarray(
            base_result["selected_indices"],
            dtype=np.int64,
        )
        features = ranker_features(
            contexts,
            desired,
            states,
            base_scores,
            include_action_basis=True,
        )
        flat = features.reshape(-1, features.shape[-1])
        raw = np.asarray(
            artifact["classifier"].predict(flat, raw_score=True),
            dtype=np.float32,
        ).reshape(len(features), len(ACTION_GRID))
        predicted_quality = np.asarray(
            artifact["regressor"].predict(flat),
            dtype=np.float32,
        ).reshape(len(features), len(ACTION_GRID))
        calibration = np.arange(0, len(features), 2, dtype=np.int64)
        confirmation = np.arange(1, len(features), 2, dtype=np.int64)
        blocks[block_name] = {
            "positives": positives,
            "base_scores": base_scores,
            "base_indices": base_indices,
            "raw": raw,
            "probabilities": sigmoid(raw),
            "quality_score": normalized_per_request(-predicted_quality),
            "calibration": calibration,
            "confirmation": confirmation,
            "full": np.arange(len(features), dtype=np.int64),
        }

    best = None
    search_count = 0
    for quality_weight in QUALITY_WEIGHTS:
        candidate_cache: dict[tuple[str, float], dict[str, np.ndarray]] = {}
        for block_name, values in blocks.items():
            model_score = (
                normalized_per_request(values["raw"])
                + float(quality_weight) * values["quality_score"]
            )
            for model_weight in MODEL_WEIGHTS:
                blended = (
                    normalized_per_request(values["base_scores"])
                    + float(model_weight)
                    * normalized_per_request(model_score)
                )
                candidate_cache[(block_name, model_weight)] = {
                    "model_score": model_score,
                    "candidate_indices": (
                        blended - 1e-7 * MOVEMENT[None, :]
                    ).argmax(axis=1),
                }
        for model_weight in MODEL_WEIGHTS:
            for candidate_probability in CANDIDATE_PROBABILITIES:
                for probability_advantage in PROBABILITY_ADVANTAGES:
                    for base_probability in BASE_PROBABILITIES:
                        for model_advantage in MODEL_SCORE_ADVANTAGES:
                            rule = {
                                "candidate_probability_min": float(
                                    candidate_probability
                                ),
                                "probability_advantage_min": float(
                                    probability_advantage
                                ),
                                "base_probability_max": float(
                                    base_probability
                                ),
                                "model_score_advantage_min": float(
                                    model_advantage
                                ),
                            }
                            block_results = {}
                            for block_name, values in blocks.items():
                                cached = candidate_cache[
                                    (block_name, model_weight)
                                ]
                                selected, changed = apply_gate(
                                    values["base_indices"],
                                    cached["candidate_indices"],
                                    values["probabilities"],
                                    cached["model_score"],
                                    rule,
                                )
                                block_results[block_name] = success_metrics(
                                    values["positives"],
                                    values["base_indices"],
                                    selected,
                                    changed,
                                    values["calibration"],
                                )
                            gains = [
                                value["gain"]
                                for value in block_results.values()
                            ]
                            if min(gains) < 0:
                                search_count += 1
                                continue
                            changed_total = sum(
                                value["changed_count"]
                                for value in block_results.values()
                            )
                            key = (
                                min(gains),
                                sum(gains),
                                -changed_total,
                                -float(model_weight),
                                -float(quality_weight),
                            )
                            candidate = {
                                "key": key,
                                "quality_weight": float(quality_weight),
                                "model_weight": float(model_weight),
                                "gate_rule": rule,
                                "calibration": block_results,
                            }
                            if best is None or key > best["key"]:
                                best = candidate
                            search_count += 1
    if best is None:
        raise RuntimeError("no non-regressing inverse gate found")

    evaluations = {}
    confirmation_passed = True
    full_passed = True
    for block_name, values in blocks.items():
        model_score = (
            normalized_per_request(values["raw"])
            + float(best["quality_weight"]) * values["quality_score"]
        )
        blended = (
            normalized_per_request(values["base_scores"])
            + float(best["model_weight"])
            * normalized_per_request(model_score)
        )
        candidate_indices = (
            blended - 1e-7 * MOVEMENT[None, :]
        ).argmax(axis=1)
        selected, changed = apply_gate(
            values["base_indices"],
            candidate_indices,
            values["probabilities"],
            model_score,
            best["gate_rule"],
        )
        evaluations[block_name] = {
            split: success_metrics(
                values["positives"],
                values["base_indices"],
                selected,
                changed,
                values[split],
            )
            for split in ("calibration", "confirmation", "full")
        }
        confirmation_passed = (
            confirmation_passed
            and evaluations[block_name]["confirmation"]["gain"] >= 0
        )
        full_passed = (
            full_passed and evaluations[block_name]["full"]["gain"] >= 0
        )

    calibrated = dict(artifact)
    calibrated.update(
        {
            "version": "inverse_success_ranker_protected_calibrated_v9_one_seed",
            "quality_weight": best["quality_weight"],
            "model_weight": best["model_weight"],
            "gate_rule": best["gate_rule"],
            "uncalibrated_artifact": str(input_path),
            "uncalibrated_artifact_sha256": sha256(input_path),
            "protected_calibration_used": True,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(calibrated, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": calibrated["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "selection": {
            **best,
            "key": list(best["key"]),
        },
        "evaluation": evaluations,
        "confirmation_passed": bool(confirmation_passed),
        "full_protected_passed": bool(full_passed),
        "promotion_passed": bool(confirmation_passed and full_passed),
        "searched_rule_count": int(search_count),
        "source_contract": {
            "protected_data_dir": str(args.data_dir.resolve()),
            "calibration_indices": "even",
            "confirmation_indices": "odd",
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
        "seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
