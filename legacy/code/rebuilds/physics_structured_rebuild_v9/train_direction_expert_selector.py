#!/usr/bin/env python3
"""Train a protected selector between complementary direction specialists."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import ACTION_GRID
from control_rebuild_v3.train_forward import configure
from control_rebuild_v3.visual_inverse import sensor_to_base_legacy
from control_rebuild_v4.inverse_data import state_mapping
from control_rebuild_v4.orchestrated_runtime import (
    OrchestratedSpecialistRuntimeV4,
)
from direction_rebuild_v4.data import (
    labels_from_normalized_change,
    load_grid_arrays,
)
from joint_forward_direction_v7.runtime import (
    load_forward_direction_runtime_v7,
)
from physics_structured_rebuild_v9.calibrate_fieldwise_direction import (
    action_complexity_mask,
)
from physics_structured_rebuild_v9.calibrate_system_direction import (
    DEFAULT_OVERLAY,
    DEFAULT_QWEN_DATA,
    read_jsonl,
    tree_probabilities,
)
from physics_structured_rebuild_v9.train_direction_tree_targeted import (
    stream_forward_changes,
)
from physics_structured_rebuild_v9.train_forward_expert_selector import (
    DEFAULT_DIFFICULT,
    DEFAULT_FORWARD,
    DEFAULT_OLD,
    DEFAULT_TRAINING,
    calibrated_prediction,
    row_indices,
    sha256,
    split_groups,
)
from specialist_rebuild_v2.common import (
    ACTION_FIELDS,
    DIRECTION_FIELDS,
    forward_feature,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CURRENT_DIR = (
    DEFAULT_RUN / "hybrid_direction_extra_forward_dual_tree"
)
DEFAULT_ALTERNATIVE = (
    DEFAULT_RUN / "hybrid_direction/hybrid_direction.pkl"
)
DEFAULT_SYSTEM_CACHE = (
    DEFAULT_RUN
    / "hybrid_direction_targeted_tree/direction_validation_cache.npz"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "direction_expert_selector_v9.pkl"
ROUTES = {
    "state": "predict_direction_from_state_v1",
    "image": "predict_direction_from_image_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    parser.add_argument(
        "--alternative-artifact",
        type=Path,
        default=DEFAULT_ALTERNATIVE,
    )
    parser.add_argument("--old-validation", type=Path, default=DEFAULT_OLD)
    parser.add_argument(
        "--difficult-validation",
        type=Path,
        default=DEFAULT_DIFFICULT,
    )
    parser.add_argument("--forward-artifact", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--qwen-data", type=Path, default=DEFAULT_QWEN_DATA)
    parser.add_argument("--v4-overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--system-cache",
        type=Path,
        default=DEFAULT_SYSTEM_CACHE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


class BatchHybridDirection:
    """Vectorized equivalent of HybridDirectionRuntimeV9 for fixed rows."""

    def __init__(self, artifact_path: Path) -> None:
        self.path = artifact_path.resolve()
        with self.path.open("rb") as stream:
            self.artifact = pickle.load(stream)
        if self.artifact.get("model") != (
            "frozen_v7_forward_plus_boundary_tree_v4"
        ):
            raise ValueError("unexpected direction hybrid artifact")
        with Path(str(self.artifact["tree_artifact"])).open("rb") as stream:
            tree = pickle.load(stream)
        self.feature_mode = str(tree.get("feature_mode", "engineered_46"))
        self.models = list(tree["models"])
        secondary_path = self.artifact.get("secondary_tree_artifact")
        if secondary_path is not None:
            with Path(str(secondary_path)).open("rb") as stream:
                secondary = pickle.load(stream)
            sources = self.artifact["field_tree_source"]
            for index, field in enumerate(DIRECTION_FIELDS):
                if str(sources[field]) == "secondary":
                    self.models[index] = secondary["models"][index]
        self.threshold_forward = self.artifact.get(
            "threshold_forward_artifact"
        )
        declared = self.artifact.get("field_calibration")
        self.calibrations = [
            dict(
                self.artifact["calibration"]
                if declared is None
                else declared[field]
            )
            for field in DIRECTION_FIELDS
        ]

    def predict(
        self,
        features51: np.ndarray,
        prior: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.threshold_forward is None:
            changes = prior
        else:
            changes = calibrated_prediction(
                Path(str(self.threshold_forward)),
                features51,
                prior,
            )
        if self.feature_mode == "engineered_46":
            tree_features = features51[:, :46]
        elif self.feature_mode == "engineered_46_plus_forward_v7_change_5":
            tree_features = features51
        else:
            raise ValueError(
                f"unsupported selector tree features: {self.feature_mode}"
            )
        probabilities = tree_probabilities(self.models, tree_features)
        threshold = labels_from_normalized_change(changes)
        predicted = threshold.copy()
        for field_index, calibration in enumerate(self.calibrations):
            field_probability = probabilities[:, field_index]
            ordered = np.sort(field_probability, axis=1)
            tree_class = field_probability.argmax(axis=1)
            distance = np.abs(np.abs(changes[:, field_index]) - 1.0)
            apply = (
                (distance <= float(calibration["boundary_limit"]))
                & (
                    ordered[:, -1]
                    >= float(calibration["confidence_limit"])
                )
                & (
                    ordered[:, -1] - ordered[:, -2]
                    >= float(calibration["margin_limit"])
                )
                & (tree_class != threshold[:, field_index])
            )
            predicted[:, field_index] = np.where(
                apply,
                tree_class,
                threshold[:, field_index],
            )
        return predicted.astype(np.int64), probabilities, changes


def direction_selector_features(
    engineered51: np.ndarray,
    prior: np.ndarray,
    current_prediction: np.ndarray,
    alternative_prediction: np.ndarray,
    current_probability: np.ndarray,
    alternative_probability: np.ndarray,
    current_changes: np.ndarray,
    alternative_changes: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            engineered51,
            prior,
            current_changes,
            alternative_changes,
            current_probability.reshape(len(prior), -1),
            alternative_probability.reshape(len(prior), -1),
            (current_prediction != alternative_prediction).astype(np.float32),
            np.abs(np.abs(current_changes) - 1.0),
            np.abs(np.abs(alternative_changes) - 1.0),
        ],
        axis=1,
    ).astype(np.float32)


def joint_success(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.all(prediction == truth, axis=1)


def choose_threshold(
    probability: np.ndarray,
    current_success: np.ndarray,
    alternative_success: np.ndarray,
    protected: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> dict[str, Any]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 1.0, np.inf]),
                np.quantile(probability, np.linspace(0.0, 1.0, 1001)),
            ]
        )
    )
    best = None
    valid_count = 0
    for threshold in thresholds:
        valid = True
        for block_probability, block_current, block_alternative, high in (
            protected
        ):
            selected = np.where(
                block_probability >= threshold,
                block_alternative,
                block_current,
            )
            valid &= int(selected.sum()) >= int(block_current.sum())
            valid &= int(selected[high].sum()) >= int(
                block_current[high].sum()
            )
        if not valid:
            continue
        valid_count += 1
        selected = np.where(
            probability >= threshold,
            alternative_success,
            current_success,
        )
        alternative_count = int((probability >= threshold).sum())
        candidate = {
            "threshold": float(threshold),
            "success_count": int(selected.sum()),
            "alternative_count": alternative_count,
        }
        key = (candidate["success_count"], -alternative_count)
        if best is None or key > best[0]:
            best = (key, candidate)
    if best is None:
        raise RuntimeError("no protected direction selector threshold")
    best[1]["valid_threshold_count"] = valid_count
    return best[1]


def build_system_features(
    args: argparse.Namespace,
    base: Any,
    torch: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    canonical = [
        row
        for row in read_jsonl(
            args.qwen_data.resolve() / "canonical/val.jsonl"
        )
        if str(row["category"]) in set(ROUTES.values())
    ]
    measurement = OrchestratedSpecialistRuntimeV4.from_manifest(
        torch,
        args.v4_overlay.resolve(),
        device,
    )
    rows = []
    features = []
    routes = []
    example_ids = []
    for row in canonical:
        target = row["target_decision"]
        arguments = target["arguments"]
        route = str(target["route_name"])
        if route == ROUTES["state"]:
            current = arguments["current_beam_state"]
        else:
            measured = measurement.visual.measure_image(
                args.qwen_data.resolve() / str(row["images"][0]),
                arguments["image_calibration"],
            )
            current = state_mapping(
                sensor_to_base_legacy(
                    measured["beam_state"],
                    [arguments["setup"]],
                )[0]
            )
        rows.append(
            {
                "group_id": str(row["example_id"]),
                "setup": arguments["setup"],
                "current_beam_state": current,
            }
        )
        features.append(
            forward_feature(
                arguments["setup"],
                current,
                arguments["action"],
            )
        )
        routes.append(route)
        example_ids.append(str(row["example_id"]))
    grid_prior = base.predict_changes(rows)
    action_lookup = {
        tuple(float(action[field]) for field in ACTION_FIELDS): index
        for index, action in enumerate(ACTION_GRID)
    }
    selected_prior = []
    for index, row in enumerate(canonical):
        action = row["target_decision"]["arguments"]["action"]
        key = tuple(
            float(action[field])
            for field in ACTION_FIELDS
        )
        selected_prior.append(grid_prior[index, action_lookup[key]])
    prior = np.asarray(selected_prior, dtype=np.float32)
    engineered51 = np.concatenate(
        [np.asarray(features, dtype=np.float32), prior],
        axis=1,
    )
    with np.load(args.system_cache.resolve(), allow_pickle=False) as cache:
        if not np.array_equal(
            np.asarray(example_ids, dtype=np.str_),
            cache["example_ids"],
        ):
            raise ValueError("rebuilt direction system example order differs")
        if not np.array_equal(np.asarray(routes, dtype=np.str_), cache["routes"]):
            raise ValueError("rebuilt direction system routes differ")
        truth = np.asarray(cache["truth"], dtype=np.int64)
        if not np.allclose(prior, cache["changes"], atol=1e-5, rtol=1e-5):
            raise ValueError("rebuilt direction system priors differ")
    return engineered51, prior, truth, np.asarray(routes, dtype=np.str_)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    from sklearn.ensemble import HistGradientBoostingClassifier

    with np.load(args.training_cache.resolve(), allow_pickle=False) as cache:
        features51 = np.asarray(cache["grid_features"], dtype=np.float32)
        prior = np.asarray(cache["grid_base_prediction"], dtype=np.float32)
        target = np.asarray(
            cache["grid_target_normalized"],
            dtype=np.float32,
        )
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
    truth = labels_from_normalized_change(target)
    training_groups, calibration_groups = split_groups(
        group_ids,
        int(args.seed),
    )
    train_indices = row_indices(training_groups)
    calibration_indices = row_indices(calibration_groups)

    current_dir = args.current_dir.resolve()
    alternative_path = args.alternative_artifact.resolve()
    alternative = BatchHybridDirection(alternative_path)
    current_models = {
        route_key: BatchHybridDirection(
            current_dir / f"hybrid_direction_{route_key}.pkl"
        )
        for route_key in ROUTES
    }
    route_arrays = {}
    for route_key, current in current_models.items():
        cp, cprob, cchange = current.predict(features51, prior)
        ap, aprob, achange = alternative.predict(features51, prior)
        selector_input = direction_selector_features(
            features51,
            prior,
            cp,
            ap,
            cprob,
            aprob,
            cchange,
            achange,
        )
        route_arrays[route_key] = {
            "current_prediction": cp,
            "alternative_prediction": ap,
            "current_probability": cprob,
            "alternative_probability": aprob,
            "current_changes": cchange,
            "alternative_changes": achange,
            "features": selector_input,
            "current_success": joint_success(cp, truth),
            "alternative_success": joint_success(ap, truth),
        }

    torch, device = configure(int(args.seed), args.device)
    base, _ = load_forward_direction_runtime_v7(
        args.forward_artifact.resolve(),
        torch,
        device,
    )
    protected_blocks = {}
    for block, path in (
        ("old_iid", args.old_validation.resolve()),
        ("difficult", args.difficult_validation.resolve()),
    ):
        arrays = load_grid_arrays(path, include_legacy_features=False)
        block_prior = stream_forward_changes(path, base)
        block_features51 = np.concatenate(
            [arrays.features, block_prior],
            axis=1,
        ).astype(np.float32)
        protected_blocks[block] = {
            "features51": block_features51,
            "prior": block_prior,
            "truth": arrays.labels,
            "high": action_complexity_mask(arrays.group_count),
        }

    routes = {}
    route_reports = {}
    for route_index, route_key in enumerate(ROUTES):
        values = route_arrays[route_key]
        exclusive = (
            values["current_success"] ^ values["alternative_success"]
        )
        selected_train = train_indices[exclusive[train_indices]]
        labels = values["alternative_success"][selected_train].astype(
            np.int64
        )
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError("direction selector lacks both exclusive classes")
        weights = (len(labels) / (2.0 * counts))[labels]
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=int(args.seed) + route_index,
        )
        classifier.fit(
            values["features"][selected_train],
            labels,
            sample_weight=weights,
        )
        protected_constraints = []
        protected_report = {}
        for block, block_values in protected_blocks.items():
            current = current_models[route_key]
            cp, cprob, cchange = current.predict(
                block_values["features51"],
                block_values["prior"],
            )
            ap, aprob, achange = alternative.predict(
                block_values["features51"],
                block_values["prior"],
            )
            probability = classifier.predict_proba(
                direction_selector_features(
                    block_values["features51"],
                    block_values["prior"],
                    cp,
                    ap,
                    cprob,
                    aprob,
                    cchange,
                    achange,
                )
            )[:, 1]
            current_success = joint_success(cp, block_values["truth"])
            alternative_success = joint_success(ap, block_values["truth"])
            protected_constraints.append(
                (
                    probability,
                    current_success,
                    alternative_success,
                    block_values["high"],
                )
            )
            protected_report[block] = {
                "probability": probability,
                "current_success": current_success,
                "alternative_success": alternative_success,
                "high": block_values["high"],
            }
        calibration_probability = classifier.predict_proba(
            values["features"][calibration_indices]
        )[:, 1]
        threshold = choose_threshold(
            calibration_probability,
            values["current_success"][calibration_indices],
            values["alternative_success"][calibration_indices],
            protected_constraints,
        )
        for block_values in protected_report.values():
            selected = np.where(
                block_values["probability"] >= threshold["threshold"],
                block_values["alternative_success"],
                block_values["current_success"],
            )
            block_values["current"] = [
                int(block_values["current_success"].sum()),
                int(
                    block_values["current_success"][
                        block_values["high"]
                    ].sum()
                ),
            ]
            block_values["selected"] = [
                int(selected.sum()),
                int(selected[block_values["high"]].sum()),
            ]
            for key in (
                "probability",
                "current_success",
                "alternative_success",
                "high",
            ):
                del block_values[key]
        routes[route_key] = {
            "classifier": classifier,
            "threshold": float(threshold["threshold"]),
        }
        route_reports[route_key] = {
            "exclusive_training_count": int(len(selected_train)),
            "class_count": {
                "current_only": int(counts[0]),
                "alternative_only": int(counts[1]),
            },
            "calibration": threshold,
            "protected": protected_report,
        }
        print(
            json.dumps(
                {"route": route_key, **route_reports[route_key]},
                sort_keys=True,
            ),
            flush=True,
        )

    system_features51, system_prior, system_truth, system_routes = (
        build_system_features(args, base, torch, device)
    )
    system_report = {}
    total = 0
    oracle_total = 0
    for route_key, route_name in ROUTES.items():
        current = current_models[route_key]
        cp, cprob, cchange = current.predict(
            system_features51,
            system_prior,
        )
        ap, aprob, achange = alternative.predict(
            system_features51,
            system_prior,
        )
        probability = routes[route_key]["classifier"].predict_proba(
            direction_selector_features(
                system_features51,
                system_prior,
                cp,
                ap,
                cprob,
                aprob,
                cchange,
                achange,
            )
        )[:, 1]
        mask = system_routes == route_name
        current_success = joint_success(cp[mask], system_truth[mask])
        alternative_success = joint_success(ap[mask], system_truth[mask])
        choose_alternative = (
            probability[mask] >= routes[route_key]["threshold"]
        )
        selected = np.where(
            choose_alternative,
            alternative_success,
            current_success,
        )
        total += int(selected.sum())
        oracle_total += int((current_success | alternative_success).sum())
        system_report[route_key] = {
            "count": int(mask.sum()),
            "current_success_count": int(current_success.sum()),
            "alternative_success_count": int(alternative_success.sum()),
            "selected_success_count": int(selected.sum()),
            "oracle_union_success_count": int(
                (current_success | alternative_success).sum()
            ),
            "alternative_count": int(choose_alternative.sum()),
        }

    artifact = {
        "version": "direction_expert_selector_v9_one_seed",
        "model": "route_specific_hgb_direction_expert_selector_v9",
        "seed": int(args.seed),
        "feature_mode": (
            "engineered51_prior_current_alt_changes_probabilities_"
            "disagreement_boundary_distances"
        ),
        "routes": routes,
        "current_dir": str(current_dir),
        "alternative_artifact": str(alternative_path),
        "current_state_sha256": sha256(
            current_dir / "hybrid_direction_state.pkl"
        ),
        "current_image_sha256": sha256(
            current_dir / "hybrid_direction_image.pkl"
        ),
        "alternative_sha256": sha256(alternative_path),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "training_group_count": int(len(training_groups)),
        "calibration_group_count": int(len(calibration_groups)),
        "routes": route_reports,
        "protected_non_regression": all(
            selected >= current
            for route in route_reports.values()
            for block in route["protected"].values()
            for selected, current in zip(
                block["selected"],
                block["current"],
                strict=True,
            )
        ),
        "system_validation": {
            "routes": system_report,
            "selected_success_count": total,
            "selected_success_rate": total / 300,
            "oracle_union_success_count": oracle_total,
        },
        "source_contract": {
            "training_cache": str(args.training_cache.resolve()),
            "system_validation_used_for_training": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
