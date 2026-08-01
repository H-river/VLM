#!/usr/bin/env python3
"""Build and evaluate a native direction-plus-magnitude forward predictor.

The model-facing records expose the complete optical setup, current measured
beam state, and action.  The simulator is used only while constructing labels
and is never available to either the LLM direction stage or the small MLP.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from optical_sim.src.experiment_generator import load_yaml as load_sim_yaml

from .build_dataset import simulator_result
from .core import load_yaml, read_jsonl, sample_setup_config, stable_json_hash, write_jsonl


VERSION = "native_hybrid_v1"
FIELDS = (
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
)
DIRECTION_FIELDS = (
    "centroid_x",
    "centroid_y",
    "sigma_x",
    "sigma_y",
    "peak_intensity",
)
ACTION_FIELDS = (
    "lens_x_delta_mm",
    "lens_y_delta_mm",
    "camera_x_delta_mm",
    "camera_y_delta_mm",
)
FEATURE_FIELDS = (
    "wavelength_nm",
    "beam_waist_mm",
    "power_w",
    "lens_focal_length_mm",
    "lens_aperture_mm",
    "source_to_lens_mm",
    "lens_to_camera_mm",
    "lens_x_offset_mm",
    "lens_y_offset_mm",
    "camera_x_offset_mm",
    "camera_y_offset_mm",
    "pixel_size_um",
    "centroid_x_px",
    "centroid_y_px",
    "sigma_x_px",
    "sigma_y_px",
    "peak_intensity",
    *ACTION_FIELDS,
)
THRESHOLDS = {
    "centroid_x_px": 1.0,
    "centroid_y_px": 1.0,
    "sigma_x_px": 2.0,
    "sigma_y_px": 2.0,
    "peak_intensity_relative": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--pilot-root", type=Path, required=True)
    build.add_argument("--dev-root", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--train-actions", type=int, default=12)
    build.add_argument("--val-actions", type=int, default=8)
    build.add_argument("--ood-cases", type=int, default=60)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--seed", type=int, default=20260717)

    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seeds", default="17,42,91")
    train.add_argument("--max-epochs", type=int, default=300)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--llm-predictions", type=Path)
    return parser.parse_args()


def full_setup(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return every physical value needed by the predictor, in readable units."""
    return {
        "wavelength_nm": float(config["source"]["wavelength"]) * 1e9,
        "beam_waist_mm": float(config["source"]["beam_waist"]) * 1e3,
        "power_w": float(config["source"]["power"]),
        "lens_focal_length_mm": float(config["lens"]["focal_length"]) * 1e3,
        "lens_aperture_mm": float(config["lens"]["clear_aperture"]) * 1e3,
        "source_to_lens_mm": float(config["geometry"]["laser_to_lens"]) * 1e3,
        "lens_to_camera_mm": float(config["geometry"]["lens_to_camera"]) * 1e3,
        "lens_x_offset_mm": float(config["lens"]["x_offset"]) * 1e3,
        "lens_y_offset_mm": float(config["lens"]["y_offset"]) * 1e3,
        "camera_x_offset_mm": float(config["camera"]["x_offset"]) * 1e3,
        "camera_y_offset_mm": float(config["camera"]["y_offset"]) * 1e3,
        "pixel_size_um": float(config["sensor"]["pixel_pitch"]) * 1e6,
        "sensor_resolution_px": [int(v) for v in config["sensor"]["resolution"]],
        "coordinate_convention": "sensor x increases right; sensor y increases down",
    }


def rounded(values: Mapping[str, Any], digits: int = 6) -> dict[str, float]:
    return {key: round(float(values[key]), digits) for key in FIELDS}


def state_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    return {key: round(float(after[key]) - float(before[key]), 6) for key in FIELDS}


def direction(value: float, threshold: float) -> str:
    if value > threshold:
        return "increase"
    if value < -threshold:
        return "decrease"
    return "no_change"


def directions(change: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, str]:
    peak_threshold = THRESHOLDS["peak_intensity_relative"] * max(
        abs(float(before["peak_intensity"])), 1e-12
    )
    return {
        "centroid_x": direction(float(change["centroid_x_px"]), 1.0),
        "centroid_y": direction(float(change["centroid_y_px"]), 1.0),
        "sigma_x": direction(float(change["sigma_x_px"]), 2.0),
        "sigma_y": direction(float(change["sigma_y_px"]), 2.0),
        "peak_intensity": direction(float(change["peak_intensity"]), peak_threshold),
    }


def action_for(group_id: str, index: int, seed: int) -> dict[str, float]:
    token = hashlib.sha256(f"{seed}:{group_id}:{index}".encode()).digest()
    rng = random.Random(int.from_bytes(token[:8], "big"))
    action = {key: 0.0 for key in ACTION_FIELDS}
    # One third of the curriculum isolates an actuator for identifiability;
    # two thirds matches deployment, where several actuators move together.
    mode = index % 12
    if mode < 4:
        key = ACTION_FIELDS[mode]
        bound = 0.05 if key.startswith("lens") else 0.02
        sign = 1.0 if token[8 + mode] % 2 == 0 else -1.0
        action[key] = sign * rng.uniform(0.35 * bound, bound)
    else:
        action = {
            "lens_x_delta_mm": rng.uniform(-0.05, 0.05),
            "lens_y_delta_mm": rng.uniform(-0.05, 0.05),
            "camera_x_delta_mm": rng.uniform(-0.02, 0.02),
            "camera_y_delta_mm": rng.uniform(-0.02, 0.02),
        }
    return {key: round(float(value), 6) for key, value in action.items()}


def _simulate_job(job: tuple[str, str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str | None]) -> dict[str, Any]:
    example_id, group_id, config, before, action, ood_parameter = job
    after = simulator_result(config, action)["state"]
    change = state_delta(before, after)
    return {
        "example_id": example_id,
        "group_id": group_id,
        "distribution": "ood" if ood_parameter else "iid",
        "ood_parameter": ood_parameter,
        "inputs": {
            "setup": full_setup(config),
            "current_beam_state": rounded(before),
            "action": dict(action),
        },
        "target": {
            "change": change,
            "after_state": rounded(after),
            "directions": directions(change, before),
        },
    }


def canonical_forward_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(path) if row.get("task_type") == "forward_prediction"]


def build_jobs_from_cases(
    cases: Iterable[Mapping[str, Any]], split: str, count: int, seed: int
) -> list[tuple[str, str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str | None]]:
    jobs = []
    for case in cases:
        if case.get("split") != split:
            continue
        for index in range(count):
            jobs.append(
                (
                    f"{VERSION}_{case['group_id']}_{index:02d}",
                    str(case["group_id"]),
                    case["setup_config"],
                    case["baseline_state"],
                    action_for(str(case["group_id"]), index, seed),
                    None,
                )
            )
    return jobs


def build_iid_eval(dev_root: Path) -> list[dict[str, Any]]:
    cases = {row["group_id"]: row for row in read_jsonl(dev_root / "master/cases.jsonl")}
    records = canonical_forward_rows(dev_root / "canonical/val.jsonl")
    rows = []
    for record in records:
        # This first round is deliberately tabular.  Visual-only initial states
        # are reserved for the CNN round rather than secretly using simulator state.
        inputs = record["prompt_inputs"]
        if "current_observation" not in inputs:
            continue
        case = cases[record["group_id"]]
        change = record["target"]["answer"]["change"]
        rows.append(
            {
                "example_id": f"{VERSION}_{record['example_id']}",
                "group_id": record["group_id"],
                "distribution": "iid",
                "ood_parameter": None,
                "inputs": {
                    "setup": full_setup(case["setup_config"]),
                    "current_beam_state": rounded(inputs["current_observation"]),
                    "action": dict(inputs["action"]),
                },
                "target": {
                    "change": {key: float(change[key]) for key in FIELDS},
                    "after_state": {
                        key: float(record["target"]["answer"]["after_state"][key]) for key in FIELDS
                    },
                    "directions": directions(change, inputs["current_observation"]),
                },
            }
        )
    return rows


def build_fresh_ood(
    count: int, seed: int, dataset_cfg: Mapping[str, Any], base_cfg: Mapping[str, Any]
) -> list[tuple[str, str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str | None]]:
    parameters = tuple(dataset_cfg["simulation"]["ood_factors"])
    jobs = []
    for index in range(count):
        group_id = f"{VERSION}_ood_{index:04d}"
        parameter = parameters[index % len(parameters)]
        rng = random.Random(seed + 100_000 + index)
        config, _ = sample_setup_config(
            base_cfg,
            dataset_cfg["simulation"],
            rng,
            ood_parameter=parameter,
            ood_band=(index // len(parameters)) % 2,
        )
        before = simulator_result(config)["state"]
        action = action_for(group_id, 6 + index, seed)
        jobs.append((f"{group_id}_forward_00", group_id, config, before, action, parameter))
    return jobs


def direction_prompt(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row["inputs"]
    contract = {
        "status": "answerable",
        "answer": {
            "directions": {
                "centroid_x": "increase | decrease | no_change",
                "centroid_y": "increase | decrease | no_change",
                "sigma_x": "increase | decrease | no_change",
                "sigma_y": "increase | decrease | no_change",
                "peak_intensity": "increase | decrease | no_change",
            }
        },
    }
    prompt = (
        "Predict only the qualitative change caused by the proposed actuator action. "
        "The after-state is not provided. Use the complete initial optical setup and measured "
        "beam state below. Treat absolute changes of at most 1 px in each centroid, 2 px in "
        "each width, and 5% of the initial peak intensity as no_change.\n\n"
        f"Input data:\n{json.dumps(payload, indent=2, sort_keys=True)}\n\n"
        "Return only strict JSON matching this contract:\n"
        f"{json.dumps(contract, indent=2)}"
    )
    return {
        "example_id": row["example_id"],
        "group_id": row["group_id"],
        "task_type": "forward_direction_prediction",
        "modality": "text",
        "prompt_inputs": {**copy.deepcopy(payload), "images": []},
        "prompt": prompt,
    }


def simulate_all(jobs: list[tuple[Any, ...]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_simulate_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_simulate_job, jobs, chunksize=2))


def command_build(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    cases = read_jsonl(args.pilot_root / "master/cases.jsonl")
    train_jobs = build_jobs_from_cases(cases, "train", args.train_actions, args.seed)
    val_jobs = build_jobs_from_cases(cases, "val", args.val_actions, args.seed)
    base_cfg = load_sim_yaml(Path(cfg["simulation"]["base_config"]))
    ood_jobs = build_fresh_ood(args.ood_cases, args.seed, cfg, base_cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = simulate_all(train_jobs, args.workers)
    val_rows = simulate_all(val_jobs, args.workers)
    iid_rows = build_iid_eval(args.dev_root)
    ood_rows = simulate_all(ood_jobs, args.workers)
    for name, rows in (
        ("train", train_rows),
        ("val", val_rows),
        ("eval_iid", iid_rows),
        ("eval_ood", ood_rows),
    ):
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    prompts = [direction_prompt(row) for row in iid_rows + ood_rows]
    write_jsonl(args.output_dir / "direction_prompts.jsonl", prompts)

    group_sets = {
        name: {row["group_id"] for row in rows}
        for name, rows in (
            ("train", train_rows),
            ("val", val_rows),
            ("eval_iid", iid_rows),
            ("eval_ood", ood_rows),
        )
    }
    overlaps = {
        f"{a}:{b}": len(group_sets[a] & group_sets[b])
        for i, a in enumerate(group_sets)
        for b in list(group_sets)[i + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError(f"scenario overlap detected: {overlaps}")
    manifest = {
        "version": VERSION,
        "seed": args.seed,
        "simulator_role": "offline label generation only",
        "model_visible_inputs": ["complete setup", "current measured beam state", "action"],
        "counts": {name: len(rows) for name, rows in (
            ("train", train_rows), ("val", val_rows), ("eval_iid", iid_rows), ("eval_ood", ood_rows)
        )},
        "group_counts": {key: len(value) for key, value in group_sets.items()},
        "group_overlaps": overlaps,
        "thresholds": THRESHOLDS,
        "feature_fields": FEATURE_FIELDS,
        "record_hashes": {
            name: stable_json_hash(rows) for name, rows in (
                ("train", train_rows), ("val", val_rows), ("eval_iid", iid_rows), ("eval_ood", ood_rows)
            )
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def feature_vector(row: Mapping[str, Any]) -> list[float]:
    inputs = row["inputs"]
    merged = {**inputs["setup"], **inputs["current_beam_state"], **inputs["action"]}
    return [
        math.log1p(max(float(merged[key]), 0.0)) if key == "peak_intensity" else float(merged[key])
        for key in FEATURE_FIELDS
    ]


def arrays(rows: list[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    y = np.asarray(
        [[float(row["target"]["change"][key]) for key in FIELDS] for row in rows],
        dtype=np.float64,
    )
    return x, y


def target_scales(rows: list[Mapping[str, Any]]) -> np.ndarray:
    """Scale every output by its sensor-relevant scoring tolerance."""
    peak = [
        0.05 * max(abs(float(row["inputs"]["current_beam_state"]["peak_intensity"])), 1e-12)
        for row in rows
    ]
    return np.column_stack(
        [
            np.ones(len(rows)),
            np.ones(len(rows)),
            np.full(len(rows), 2.0),
            np.full(len(rows), 2.0),
            np.asarray(peak),
        ]
    )


def fit_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    max_epochs: int,
) -> dict[str, Any]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler().fit(x_train)
    # y arrays have already been converted to tolerance units by command_train.
    y_scaler = StandardScaler().fit(y_train)
    xt, xv = x_scaler.transform(x_train), x_scaler.transform(x_val)
    yt, yv = y_scaler.transform(y_train), y_scaler.transform(y_val)
    models = {}
    traces = {}
    for kind, target_train, target_val in (
        ("direct", yt, yv),
        ("magnitude", np.abs(y_train) / y_scaler.scale_, np.abs(y_val) / y_scaler.scale_),
    ):
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=1,
            warm_start=True,
            random_state=seed,
        )
        best_blob = None
        best_loss = float("inf")
        best_epoch = 0
        patience = 40
        history = []
        for epoch in range(1, max_epochs + 1):
            model.partial_fit(xt, target_train)
            if epoch == 1 or epoch % 5 == 0:
                pred = model.predict(xv)
                loss = float(np.mean((pred - target_val) ** 2))
                history.append({"epoch": epoch, "val_scaled_mse": loss})
                if loss < best_loss - 1e-7:
                    best_loss, best_epoch = loss, epoch
                    best_blob = pickle.dumps(model)
                elif epoch - best_epoch >= patience:
                    break
        models[kind] = pickle.loads(best_blob) if best_blob is not None else model
        traces[kind] = {
            "best_epoch": best_epoch,
            "best_val_scaled_mse": best_loss,
            "history": history,
        }
    return {
        "seed": seed,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "models": models,
        "traces": traces,
    }


def command_train(args: argparse.Namespace) -> None:
    train_rows = read_jsonl(args.data_dir / "train.jsonl")
    val_rows = read_jsonl(args.data_dir / "val.jsonl")
    x_train, y_train = arrays(train_rows)
    x_val, y_val = arrays(val_rows)
    y_train = y_train / target_scales(train_rows)
    y_val = y_val / target_scales(val_rows)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    members = [fit_one(x_train, y_train, x_val, y_val, seed, args.max_epochs) for seed in seeds]
    with (args.output_dir / "ensemble.pkl").open("wb") as stream:
        pickle.dump({"version": VERSION, "feature_fields": FEATURE_FIELDS, "fields": FIELDS, "members": members}, stream)
    training_summary = {
        "version": VERSION,
        "train_records": len(train_rows),
        "train_groups": len({row["group_id"] for row in train_rows}),
        "val_records": len(val_rows),
        "val_groups": len({row["group_id"] for row in val_rows}),
        "seeds": seeds,
        "members": [{"seed": item["seed"], "traces": item["traces"]} for item in members],
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(training_summary, indent=2, sort_keys=True))


def ensemble_predict(
    bundle: Mapping[str, Any], x: np.ndarray, rows: list[Mapping[str, Any]], kind: str
) -> np.ndarray:
    predictions = []
    for member in bundle["members"]:
        xs = member["x_scaler"].transform(x)
        pred = member["models"][kind].predict(xs)
        if kind == "direct":
            pred = member["y_scaler"].inverse_transform(pred)
        else:
            pred = np.maximum(pred, 0.0) * member["y_scaler"].scale_
        predictions.append(pred)
    # Convert tolerance-normalized changes back to physical output units.
    return np.mean(np.stack(predictions), axis=0) * target_scales(rows)


def sign_matrix(rows: list[Mapping[str, Any]], labels: list[Mapping[str, str]]) -> np.ndarray:
    signs = []
    for row, item in zip(rows, labels):
        signs.append([
            {"decrease": -1.0, "no_change": 0.0, "increase": 1.0}.get(str(item.get(field)), 0.0)
            for field in DIRECTION_FIELDS
        ])
    return np.asarray(signs, dtype=np.float64)


def labels_from_values(values: np.ndarray, rows: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    result = []
    for vector, row in zip(values, rows):
        change = dict(zip(FIELDS, vector.tolist()))
        result.append(directions(change, row["inputs"]["current_beam_state"]))
    return result


def macro_f1(predicted: list[Mapping[str, str]], targets: list[Mapping[str, str]]) -> dict[str, Any]:
    from collections import Counter

    from sklearn.metrics import f1_score

    by_field = {}
    classes = ["decrease", "no_change", "increase"]
    for field in DIRECTION_FIELDS:
        by_field[field] = float(
            f1_score(
                [item[field] for item in targets],
                [item.get(field, "no_change") for item in predicted],
                labels=classes,
                average="macro",
                zero_division=0,
            )
        )
    joint = statistics.fmean(
        float(all(p.get(field) == t[field] for field in DIRECTION_FIELDS))
        for p, t in zip(predicted, targets)
    )
    return {
        "field_macro_f1": by_field,
        "equal_field_macro_f1": statistics.fmean(by_field.values()),
        "joint_exact": joint,
        "target_distribution": {
            field: dict(Counter(item[field] for item in targets)) for field in DIRECTION_FIELDS
        },
        "predicted_distribution": {
            field: dict(Counter(item.get(field, "no_change") for item in predicted))
            for field in DIRECTION_FIELDS
        },
    }


def parse_llm_labels(path: Path | None, rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, str]] | None, float]:
    if path is None or not path.exists():
        return None, 0.0
    by_id = {item["example_id"]: item for item in read_jsonl(path)}
    labels = []
    valid = 0
    allowed = {"increase", "decrease", "no_change"}
    for row in rows:
        parsed = by_id.get(row["example_id"], {}).get("parsed_json")
        item = parsed.get("answer", {}).get("directions", {}) if isinstance(parsed, Mapping) else {}
        if all(item.get(field) in allowed for field in DIRECTION_FIELDS):
            valid += 1
            labels.append({field: str(item[field]) for field in DIRECTION_FIELDS})
        else:
            labels.append({field: "no_change" for field in DIRECTION_FIELDS})
    return labels, valid / len(rows) if rows else 0.0


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [center - radius, center + radius]


def numeric_metrics(pred: np.ndarray, target: np.ndarray, rows: list[Mapping[str, Any]], zero_mse: float) -> dict[str, Any]:
    error = pred - target
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    normalized_sq = []
    passes = []
    for index, row in enumerate(rows):
        peak_tol = 0.05 * max(abs(float(row["inputs"]["current_beam_state"]["peak_intensity"])), 1e-12)
        scales = np.asarray([1.0, 1.0, 2.0, 2.0, peak_tol])
        normalized_sq.extend(((error[index] / scales) ** 2).tolist())
        centroid_ok = math.hypot(float(error[index, 0]), float(error[index, 1])) <= 1.0
        passes.append(
            centroid_ok
            and abs(float(error[index, 2])) <= 2.0
            and abs(float(error[index, 3])) <= 2.0
            and abs(float(error[index, 4])) <= peak_tol
        )
    mse = statistics.fmean(normalized_sq)
    successes = sum(passes)
    return {
        "mae": dict(zip(FIELDS, mae.tolist())),
        "rmse": dict(zip(FIELDS, rmse.tolist())),
        "normalized_mse": mse,
        "skill_vs_zero": 1.0 - mse / zero_mse if zero_mse > 0 else 0.0,
        "strict_joint_pass_rate": successes / len(rows),
        "strict_joint_pass_wilson_95": wilson(successes, len(rows)),
    }


def evaluate_split(
    rows: list[dict[str, Any]], bundle: Mapping[str, Any], llm_path: Path | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x, target = arrays(rows)
    direct = ensemble_predict(bundle, x, rows, "direct")
    magnitude = ensemble_predict(bundle, x, rows, "magnitude")
    zero = np.zeros_like(target)
    target_labels = [row["target"]["directions"] for row in rows]
    oracle = magnitude * sign_matrix(rows, target_labels)
    direct_labels = labels_from_values(direct, rows)
    llm_labels, llm_valid = parse_llm_labels(llm_path, rows)

    peak_scales = np.asarray([
        0.05 * max(abs(float(row["inputs"]["current_beam_state"]["peak_intensity"])), 1e-12)
        for row in rows
    ])
    scales = np.column_stack([
        np.ones(len(rows)), np.ones(len(rows)), np.full(len(rows), 2.0), np.full(len(rows), 2.0), peak_scales
    ])
    zero_mse = float(np.mean((target / scales) ** 2))
    predictions = {"zero_change": zero, "direct_mlp": direct, "oracle_direction_plus_magnitude": oracle}
    direction_metrics = {"direct_mlp": macro_f1(direct_labels, target_labels)}
    if llm_labels is not None:
        predictions["llm_direction_plus_magnitude"] = magnitude * sign_matrix(rows, llm_labels)
        direction_metrics["llm"] = {**macro_f1(llm_labels, target_labels), "schema_valid_rate": llm_valid}

    systems = {name: numeric_metrics(pred, target, rows, zero_mse) for name, pred in predictions.items()}
    details = []
    for index, row in enumerate(rows):
        details.append({
            "example_id": row["example_id"],
            "group_id": row["group_id"],
            "distribution": row["distribution"],
            "ood_parameter": row.get("ood_parameter"),
            "target_change": dict(zip(FIELDS, target[index].tolist())),
            "target_directions": target_labels[index],
            "llm_directions": llm_labels[index] if llm_labels is not None else None,
            "predictions": {name: dict(zip(FIELDS, pred[index].tolist())) for name, pred in predictions.items()},
        })
    return {
        "count": len(rows),
        "zero_normalized_mse": zero_mse,
        "direction": direction_metrics,
        "systems": systems,
    }, details


def report_markdown(summary: Mapping[str, Any]) -> str:
    promoted = bool(summary["promotion"]["promoted"])
    lines = [
        "# Native hybrid direction + magnitude: first round",
        "",
        f"**Promotion decision: {'PASS' if promoted else 'REJECT'}**",
        "",
        "This experiment gives both stages the complete visible setup, current measured beam state, and action. "
        "The optical simulator was used only to create and score labels; it was not callable at inference time.",
        "",
        "The LLM is the best pre-tool native checkpoint (`corrective_v2_seed314`). The numerical component is a "
        "three-seed ensemble of small 128-64 MLPs. This first round is tabular/text-only; visual-only rows were "
        "excluded rather than supplying hidden numerical state.",
        "",
        "## Results",
        "",
        "| Split | System | Skill vs zero | Strict joint pass | Centroid-x MAE | Centroid-y MAE | Sigma-x MAE | Sigma-y MAE | Peak MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("iid", "ood"):
        for system, metrics in summary[split]["systems"].items():
            mae = metrics["mae"]
            lines.append(
                f"| {split.upper()} ({summary[split]['count']}) | {system} | {metrics['skill_vs_zero']:.3f} | "
                f"{metrics['strict_joint_pass_rate']:.3f} | {mae['centroid_x_px']:.3f} | "
                f"{mae['centroid_y_px']:.3f} | {mae['sigma_x_px']:.3f} | {mae['sigma_y_px']:.3f} | "
                f"{mae['peak_intensity']:.3f} |"
            )
    lines += ["", "## Direction stage", ""]
    for split in ("iid", "ood"):
        for source, metrics in summary[split]["direction"].items():
            valid = f", schema-valid {metrics['schema_valid_rate']:.3f}" if "schema_valid_rate" in metrics else ""
            lines.append(
                f"- {split.upper()} {source}: equal-field macro-F1 {metrics['equal_field_macro_f1']:.3f}, "
                f"all-five direction exact {metrics['joint_exact']:.3f}{valid}."
            )
    if not promoted:
        iid_pred = summary["iid"]["direction"]["llm"]["predicted_distribution"]
        ood_pred = summary["ood"]["direction"]["llm"]["predicted_distribution"]
        lines += [
            "",
            "The LLM collapsed to strong defaults: it predicted `centroid_x=decrease` for "
            f"{iid_pred['centroid_x'].get('decrease', 0)}/{summary['iid']['count']} IID and "
            f"{ood_pred['centroid_x'].get('decrease', 0)}/{summary['ood']['count']} OOD cases; "
            "it similarly predicted decreasing peak intensity for "
            f"{iid_pred['peak_intensity'].get('decrease', 0)}/{summary['iid']['count']} IID and "
            f"{ood_pred['peak_intensity'].get('decrease', 0)}/{summary['ood']['count']} OOD cases.",
        ]
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- `strict_joint_pass` requires centroid vector error <= 1 px, both width errors <= 2 px, and peak error <= 5% of the initial peak simultaneously.",
        "- `skill_vs_zero` is computed from tolerance-normalized squared error; positive is better than predicting no change, zero is tied, and negative is worse.",
        "- `oracle_direction_plus_magnitude` is an upper bound for this factorization, not a deployable result.",
        "- A hybrid should be promoted only if the real LLM-direction system beats both zero-change and direct signed regression on fresh IID and OOD setups.",
        "",
        "## Decision",
        "",
        *[f"- {reason}" for reason in summary["promotion"]["reasons"]],
        "",
    ]
    return "\n".join(lines)


def command_evaluate(args: argparse.Namespace) -> None:
    with (args.model_dir / "ensemble.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    iid_rows = read_jsonl(args.data_dir / "eval_iid.jsonl")
    ood_rows = read_jsonl(args.data_dir / "eval_ood.jsonl")
    iid_summary, iid_details = evaluate_split(iid_rows, bundle, args.llm_predictions)
    ood_summary, ood_details = evaluate_split(ood_rows, bundle, args.llm_predictions)
    reasons = []
    promoted = args.llm_predictions is not None
    for split_name, split_summary in (("IID", iid_summary), ("OOD", ood_summary)):
        llm_system = split_summary["systems"].get("llm_direction_plus_magnitude")
        if llm_system is None:
            promoted = False
            reasons.append(f"{split_name}: no LLM direction predictions were supplied.")
            continue
        zero = split_summary["systems"]["zero_change"]
        direct = split_summary["systems"]["direct_mlp"]
        if llm_system["skill_vs_zero"] <= 0:
            promoted = False
            reasons.append(
                f"{split_name}: LLM hybrid skill versus zero was {llm_system['skill_vs_zero']:.3f}, not positive."
            )
        if llm_system["strict_joint_pass_rate"] <= max(
            zero["strict_joint_pass_rate"], direct["strict_joint_pass_rate"]
        ):
            promoted = False
            reasons.append(
                f"{split_name}: strict joint pass {llm_system['strict_joint_pass_rate']:.3f} did not beat both "
                f"zero-change ({zero['strict_joint_pass_rate']:.3f}) and direct MLP ({direct['strict_joint_pass_rate']:.3f})."
            )
    summary = {
        "version": VERSION,
        "checkpoint_role": "native qualitative direction prediction",
        "simulator_at_inference": False,
        "iid": iid_summary,
        "ood": ood_summary,
        "promotion": {"promoted": promoted, "reasons": reasons},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(args.output_dir / "details.jsonl", iid_details + ood_details)
    (args.output_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "build":
        command_build(args)
    elif args.command == "train":
        command_train(args)
    elif args.command == "evaluate":
        command_evaluate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
