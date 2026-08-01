#!/usr/bin/env python3
"""Generate paired visual anomalies and run inexpensive identifiability baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, shift
from scipy.optimize import least_squares
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from continuous_control_v12.contracts import Bounds
from continuous_control_v12.simulator import simulate_state


VERSION = "vlm_optics_visual_anomaly_v1"
FAMILIES = ("sensor_saturation", "secondary_reflection")
IMAGE_SIZE = 128
MATCH_TOLERANCE = np.asarray([1.0, 1.0, 2.0, 2.0, 0.05], dtype=np.float64)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_patch(raw: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Crop around the physical peak, then resize without overlays or borders."""

    values = np.asarray(raw, dtype=np.float64)
    peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)
    raw_width = moment_metrics(values)[2:4]
    crop_size = int(np.clip(math.ceil(float(raw_width.max()) * 8.0), 128, 768))
    crop_size += crop_size % 2
    crop_size = min(crop_size, values.shape[0], values.shape[1])
    half = crop_size // 2
    padded = np.pad(values, half, mode="constant")
    y, x = peak_y + half, peak_x + half
    crop = padded[y - half : y - half + crop_size, x - half : x - half + crop_size]
    high = max(float(crop.max()), 1e-30)
    crop = np.clip(crop / high, 0.0, 1.0)
    image = Image.fromarray(np.rint(crop * 65535).astype(np.uint16), mode="I;16")
    resized = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float64)
    resized /= max(float(resized.max()), 1.0)
    return resized.astype(np.float32)


def moment_metrics(image: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    total = max(float(values.sum()), 1e-12)
    y, x = np.indices(values.shape, dtype=np.float64)
    cx = float((values * x).sum() / total)
    cy = float((values * y).sum() / total)
    sx = math.sqrt(max(float((values * (x - cx) ** 2).sum() / total), 1e-12))
    sy = math.sqrt(max(float((values * (y - cy) ** 2).sum() / total), 1e-12))
    return np.asarray([cx, cy, sx, sy, float(values.max())], dtype=np.float64)


def _translate_no_wrap(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    return shift(image, shift=(dy, dx), order=1, mode="constant", cval=0.0, prefilter=False)


def inject_anomaly(
    base: np.ndarray,
    family: str,
    severity: Mapping[str, float],
) -> np.ndarray:
    values = np.asarray(base, dtype=np.float64)
    if family == "sensor_saturation":
        level = float(severity["clip_level_fraction_of_peak"])
        anomalous = np.minimum(values, level) / max(level, 1e-12)
    elif family == "secondary_reflection":
        amplitude = float(severity["reflection_amplitude_fraction"])
        offset = float(severity["reflection_offset_px"])
        angle = float(severity["reflection_angle_radians"])
        reflected = _translate_no_wrap(values, offset * math.sin(angle), offset * math.cos(angle))
        anomalous = values + amplitude * reflected
        anomalous /= max(float(anomalous.max()), 1e-12)
    else:
        raise ValueError(f"unknown anomaly family: {family}")
    return np.clip(anomalous, 0.0, 1.0).astype(np.float32)


def _affine_moment_transform(base: np.ndarray, desired: Sequence[float]) -> np.ndarray:
    desired = np.asarray(desired, dtype=np.float64)
    source = moment_metrics(base)
    y, x = np.indices(base.shape, dtype=np.float64)
    input_x = source[0] + (x - desired[0]) * source[2] / max(desired[2], 1e-6)
    input_y = source[1] + (y - desired[1]) * source[3] / max(desired[3], 1e-6)
    output = map_coordinates(base, [input_y, input_x], order=1, mode="constant", cval=0.0)
    output *= desired[4] / max(float(output.max()), 1e-12)
    return np.clip(output, 0.0, None).astype(np.float32)


def matched_clean_counterfactual(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    initial = np.asarray(target[:4], dtype=np.float64)

    def residual(parameters: np.ndarray) -> np.ndarray:
        desired = np.concatenate([parameters, [target[4]]])
        observed = moment_metrics(_affine_moment_transform(base, desired))
        return (observed[:4] - target[:4]) / MATCH_TOLERANCE[:4]

    fitted = least_squares(
        residual,
        initial,
        bounds=([8.0, 8.0, 1.0, 1.0], [119.0, 119.0, 60.0, 60.0]),
        max_nfev=30,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    desired = np.concatenate([fitted.x, [target[4]]])
    return _affine_moment_transform(base, desired)


def normalized_metric_distance(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    delta = np.abs(np.asarray(left) - np.asarray(right)) / MATCH_TOLERANCE
    return {
        "per_metric_absolute_difference_tolerances": delta.tolist(),
        "maximum_absolute_difference_tolerances": float(delta.max()),
        "total_l2_distance_tolerances": float(np.linalg.norm(delta)),
        "passes_frozen_match": bool(delta.max() <= 0.25 and np.linalg.norm(delta) <= 0.40),
    }


def severity_for(case_id: str, family: str, split: str) -> dict[str, float]:
    rng = np.random.default_rng(stable_seed("severity", case_id, family, split))
    if family == "sensor_saturation":
        low, high = (0.35, 0.55) if split != "severity_ood" else (0.18, 0.28)
        return {"clip_level_fraction_of_peak": float(rng.uniform(low, high))}
    amplitude = rng.uniform(0.40, 0.55) if split != "severity_ood" else rng.uniform(0.60, 0.75)
    offset = rng.uniform(24.0, 32.0) if split != "severity_ood" else rng.uniform(34.0, 44.0)
    return {
        "reflection_amplitude_fraction": float(amplitude),
        "reflection_offset_px": float(offset),
        "reflection_angle_radians": float(rng.uniform(0.0, 2.0 * math.pi)),
    }


def _selected_cases(suite: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    cases = list(suite["cases"])
    if split == "train":
        selected = [case for case in cases if int(str(case["case_id"]).rsplit("_", 1)[1]) <= 7]
    elif split == "iid_heldout":
        selected = [
            case
            for case in cases
            if 8 <= int(str(case["case_id"]).rsplit("_", 1)[1]) <= 9
        ]
    elif split == "severity_ood":
        selected = cases
    else:
        raise ValueError(f"unknown split: {split}")
    expected = {"train": 24, "iid_heldout": 6, "severity_ood": 30}[split]
    if len(selected) != expected:
        raise ValueError(f"expected {expected} {split} cases, found {len(selected)}")
    return selected


def _candidate_options(sample_id: str) -> list[str]:
    options = ["standard_metrics", "reduce_exposure_reacquire", "primary_spot_specialist", "stop"]
    random.Random(stable_seed("candidate_order", sample_id)).shuffle(options)
    return options


def _save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def generate_split(args: argparse.Namespace) -> None:
    suite = json.loads(args.suite.resolve().read_text())
    cases = _selected_cases(suite, args.split)
    config = json.loads(args.v12_config.resolve().read_text())
    bounds = Bounds.from_config(config)
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        capture = simulate_state(
            case["setup_context"],
            case["initial_positions_mm"],
            case["simulator_fixed"],
            str(args.base_config.resolve()),
            bounds,
        )
        base = canonical_patch(capture["intensity"])
        history = moment_metrics(base).tolist()
        for family in FAMILIES:
            severity = severity_for(str(case["case_id"]), family, args.split)
            anomalous = inject_anomaly(base, family, severity)
            anomaly_metrics = moment_metrics(anomalous)
            clean = matched_clean_counterfactual(base, anomaly_metrics)
            clean_metrics = moment_metrics(clean)
            match = normalized_metric_distance(clean_metrics, anomaly_metrics)
            if not match["passes_frozen_match"]:
                raise RuntimeError(
                    f"frozen counterfactual criterion infeasible for {case['case_id']} {family}: {match}"
                )
            pair_id = hashlib.sha256(
                f"{args.split}:{case['group_id']}:{family}".encode()
            ).hexdigest()[:20]
            role_data = (("clean", clean, clean_metrics), (family, anomalous, anomaly_metrics))
            refs: dict[str, str] = {}
            sample_ids: dict[str, str] = {}
            for label, image, metrics in role_data:
                sample_id = hashlib.sha256(f"{pair_id}:{label}:sample".encode()).hexdigest()[:24]
                relative = Path("images") / args.split / f"img_{sample_id}.png"
                _save_png(image, args.output_dir / relative)
                refs[label] = str(relative)
                sample_ids[label] = sample_id
                model_input = {
                    "image_ref": str(relative),
                    "five_metrics": [float(value) for value in metrics],
                    "five_metric_uncertainties": MATCH_TOLERANCE.tolist(),
                    "short_history_metrics": [history],
                    "target": case["target_metrics"],
                    "candidate_options": _candidate_options(sample_id),
                }
                rows.append(
                    {
                        "sample_id": sample_id,
                        "pair_id": pair_id,
                        "setup_id": case["group_id"],
                        "setup_hash": case["setup_hash"],
                        "split": args.split,
                        "family_audit": family,
                        "severity_scalar": (
                            1.0 - float(severity["clip_level_fraction_of_peak"])
                            if family == "sensor_saturation"
                            else float(severity["reflection_amplitude_fraction"])
                        ),
                        "model_input": model_input,
                        "supervision": {
                            "fault_type": label,
                            "binary_fault_present": label != "clean",
                            "oracle_recovery_decision": (
                                "standard_metrics"
                                if label == "clean"
                                else "reduce_exposure_reacquire"
                                if family == "sensor_saturation"
                                else "primary_spot_specialist"
                            ),
                        },
                    }
                )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "setup_id": case["group_id"],
                    "setup_hash": case["setup_hash"],
                    "split": args.split,
                    "family": family,
                    "clean_sample_id": sample_ids["clean"],
                    "anomalous_sample_id": sample_ids[family],
                    "clean_image_ref": refs["clean"],
                    "anomalous_image_ref": refs[family],
                    "clean_five_metrics": clean_metrics.tolist(),
                    "anomalous_five_metrics": anomaly_metrics.tolist(),
                    "target": case["target_metrics"],
                    "anomaly_type": family,
                    "severity": severity,
                    "normalized_metric_distance": match,
                    "oracle_recovery_decision": (
                        "reduce_exposure_reacquire"
                        if family == "sensor_saturation"
                        else "primary_spot_specialist"
                    ),
                    "no_diagnosis_outcome": None,
                    "oracle_diagnosis_outcome": None,
                    "relevant_safety_outcomes": None,
                }
            )
        print(
            json.dumps({"event": "visual_setup_complete", "split": args.split, "case": case_index, "cases": len(cases), "setup_id": case["group_id"]}),
            flush=True,
        )
    rng = np.random.default_rng(stable_seed("serialization_order", args.split))
    rng.shuffle(rows)
    rng.shuffle(pairs)
    write_jsonl(args.output_dir / f"dataset_{args.split}.jsonl", rows)
    write_jsonl(args.output_dir / f"pairs_{args.split}.jsonl", pairs)
    summary = {
        "version": VERSION,
        "split": args.split,
        "suite": str(args.suite.resolve()),
        "suite_sha256": sha256_file(args.suite),
        "setup_groups": len(cases),
        "samples": len(rows),
        "pairs": len(pairs),
        "class_distribution": dict(sorted(Counter(row["supervision"]["fault_type"] for row in rows).items())),
        "families": list(FAMILIES),
        "match_criterion": {
            "maximum_per_metric_difference_tolerances": 0.25,
            "maximum_total_l2_distance_tolerances": 0.40,
            "tolerance_vector": MATCH_TOLERANCE.tolist(),
        },
        "maximum_observed_pair_metric_difference_tolerances": max(
            pair["normalized_metric_distance"]["maximum_absolute_difference_tolerances"] for pair in pairs
        ),
        "maximum_observed_pair_l2_distance_tolerances": max(
            pair["normalized_metric_distance"]["total_l2_distance_tolerances"] for pair in pairs
        ),
        "all_pairs_pass": all(pair["normalized_metric_distance"]["passes_frozen_match"] for pair in pairs),
    }
    atomic_json(args.output_dir / f"summary_{args.split}.json", summary)


def _load_images(rows: Sequence[Mapping[str, Any]], root: Path, *, border_mask: int = 0) -> np.ndarray:
    images = []
    for row in rows:
        image = np.asarray(Image.open(root / row["model_input"]["image_ref"]).convert("L"), dtype=np.float32) / 255.0
        if border_mask:
            image[:border_mask] = 0.0
            image[-border_mask:] = 0.0
            image[:, :border_mask] = 0.0
            image[:, -border_mask:] = 0.0
        images.append(image[None, :, :])
    return np.stack(images)


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["supervision"]["binary_fault_present"]) for row in rows])


def _metric_features(rows: Sequence[Mapping[str, Any]], history: bool = False) -> np.ndarray:
    values = []
    for row in rows:
        feature = list(map(float, row["model_input"]["five_metrics"]))
        if history:
            feature.extend(map(float, row["model_input"]["short_history_metrics"][0]))
            feature.extend(np.asarray(feature[:5]) - np.asarray(feature[5:10]))
        values.append(feature)
    return np.asarray(values, dtype=np.float64)


def _ece(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = np.maximum(probability, 1.0 - probability)
    correct = ((probability >= 0.5).astype(int) == labels).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            value += float(mask.sum() / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def _score(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
        "expected_calibration_error_10_bins": _ece(probability, labels),
        "episodes": len(labels),
    }


def _severity_scores(rows: Sequence[Mapping[str, Any]], labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    severities = np.asarray([float(row["severity_scalar"]) for row in rows])
    anomaly = labels == 1
    if not anomaly.any():
        return {}
    cuts = np.quantile(severities[anomaly], [1 / 3, 2 / 3])
    output = {}
    for name, mask in (
        ("low", anomaly & (severities <= cuts[0])),
        ("medium", anomaly & (severities > cuts[0]) & (severities <= cuts[1])),
        ("high", anomaly & (severities > cuts[1])),
    ):
        output[name] = {
            "anomaly_examples": int(mask.sum()),
            "recall": float(np.mean(probability[mask] >= 0.5)) if mask.any() else None,
        }
    return output


def _torch_models(seed: int) -> tuple[Any, Any, Any]:
    import torch

    torch.manual_seed(seed)

    class TinyCNN(torch.nn.Module):
        def __init__(self, metric_dim: int = 0) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, 5, stride=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((4, 4)),
                torch.nn.Flatten(),
            )
            self.metric_dim = metric_dim
            self.head = torch.nn.Sequential(
                torch.nn.Linear(32 * 4 * 4 + metric_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )

        def forward(self, image: Any, metrics: Any | None = None) -> Any:
            encoded = self.encoder(image)
            if self.metric_dim:
                encoded = torch.cat([encoded, metrics], dim=1)
            return self.head(encoded).squeeze(1)

    return torch, TinyCNN(0), TinyCNN(5)


def _augment(images: np.ndarray, labels: np.ndarray, metrics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    augmented_images, augmented_labels, augmented_metrics = [], [], []
    for image, label, metric in zip(images, labels, metrics, strict=True):
        for turns in range(4):
            augmented_images.append(np.rot90(image, turns, axes=(1, 2)).copy())
            augmented_labels.append(label)
            augmented_metrics.append(metric)
    return np.stack(augmented_images), np.asarray(augmented_labels), np.stack(augmented_metrics)


def _fit_tiny_model(
    train_images: np.ndarray,
    train_metrics: np.ndarray,
    train_labels: np.ndarray,
    *,
    metric_dim: int,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    torch, image_model, multimodal_model = _torch_models(seed)
    model = image_model if metric_dim == 0 else multimodal_model
    images, labels, metrics = _augment(train_images, train_labels, train_metrics)
    metric_mean = train_metrics.mean(axis=0).astype(np.float32)
    metric_scale = np.maximum(train_metrics.std(axis=0), 1e-6).astype(np.float32)
    tensor_x = torch.from_numpy(images.astype(np.float32))
    tensor_y = torch.from_numpy(labels.astype(np.float32))
    tensor_m = torch.from_numpy(((metrics - metric_mean) / metric_scale).astype(np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(45):
        order = torch.randperm(len(tensor_y), generator=generator)
        for start in range(0, len(order), 32):
            index = order[start : start + 32]
            optimizer.zero_grad()
            logits = model(tensor_x[index], tensor_m[index] if metric_dim else None)
            loss = loss_fn(logits, tensor_y[index])
            loss.backward()
            optimizer.step()
    return model, metric_mean, metric_scale


def _predict_tiny(model: Any, images: np.ndarray, metrics: np.ndarray, mean: np.ndarray, scale: np.ndarray, metric_dim: int) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        image_tensor = torch.from_numpy(images.astype(np.float32))
        metric_tensor = torch.from_numpy(((metrics - mean) / scale).astype(np.float32))
        logits = model(image_tensor, metric_tensor if metric_dim else None)
        return torch.sigmoid(logits).cpu().numpy()


def _subset(rows: Sequence[Mapping[str, Any]], family: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row["family_audit"]) == family]


def train_baselines(args: argparse.Namespace) -> None:
    import torch

    all_rows = {
        split: read_jsonl(args.data_dir / f"dataset_{split}.jsonl")
        for split in ("train", "iid_heldout", "severity_ood")
    }
    results: dict[str, Any] = {
        "version": "vlm_optics_identifiability_results_v1",
        "protected_set_used": False,
        "small_cnn_is_visual_diagnostic_not_vlm": True,
        "families": {},
    }
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for family_index, family in enumerate(FAMILIES):
        rows = {split: _subset(values, family) for split, values in all_rows.items()}
        labels = {split: _labels(values) for split, values in rows.items()}
        metrics = {split: _metric_features(values) for split, values in rows.items()}
        history = {split: _metric_features(values, history=True) for split, values in rows.items()}
        images = {split: _load_images(values, args.data_dir) for split, values in rows.items()}
        metric_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=args.seed, max_iter=2000))
        metric_model.fit(metrics["train"], labels["train"])
        history_model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), random_state=args.seed, max_iter=2000, early_stopping=False))
        history_model.fit(history["train"], labels["train"])
        image_model, image_mean, image_scale = _fit_tiny_model(images["train"], metrics["train"], labels["train"], metric_dim=0, seed=args.seed + family_index)
        multimodal_model, multi_mean, multi_scale = _fit_tiny_model(images["train"], metrics["train"], labels["train"], metric_dim=5, seed=args.seed + 10 + family_index)
        torch.save(
            {
                "state_dict": image_model.state_dict(),
                "metric_dim": 0,
                "metric_mean": image_mean,
                "metric_scale": image_scale,
                "family": family,
                "seed": args.seed + family_index,
            },
            args.model_dir / f"{family}_image_cnn.pt",
        )
        torch.save(
            {
                "state_dict": multimodal_model.state_dict(),
                "metric_dim": 5,
                "metric_mean": multi_mean,
                "metric_scale": multi_scale,
                "family": family,
                "seed": args.seed + 10 + family_index,
            },
            args.model_dir / f"{family}_multimodal_cnn.pt",
        )
        joblib.dump(metric_model, args.model_dir / f"{family}_metrics_logistic.joblib")
        joblib.dump(history_model, args.model_dir / f"{family}_metrics_history_mlp.joblib")
        family_result: dict[str, Any] = {
            "class_distribution": {
                split: {
                    str(int(label)): int(count)
                    for label, count in sorted(Counter(map(int, labels[split])).items())
                }
                for split in labels
            },
            "splits": {},
        }
        cached_probability: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for split in rows:
            probabilities = {
                "metrics_only_logistic": metric_model.predict_proba(metrics[split])[:, 1],
                "metrics_history_mlp": history_model.predict_proba(history[split])[:, 1],
                "image_only_small_cnn": _predict_tiny(image_model, images[split], metrics[split], image_mean, image_scale, 0),
                "image_metrics_multimodal_small_model": _predict_tiny(multimodal_model, images[split], metrics[split], multi_mean, multi_scale, 5),
                "hidden_fault_label_oracle": labels[split].astype(float),
            }
            cached_probability[split] = probabilities
            family_result["splits"][split] = {
                name: {**_score(labels[split], probability), "severity": _severity_scores(rows[split], labels[split], probability)}
                for name, probability in probabilities.items()
            }
        eval_rows = rows["iid_heldout"]
        eval_images = images["iid_heldout"]
        shuffled = eval_images[np.random.default_rng(stable_seed("shuffle_images", family)).permutation(len(eval_images))]
        shuffled_probability = _predict_tiny(multimodal_model, shuffled, metrics["iid_heldout"], multi_mean, multi_scale, 5)
        masked = _load_images(eval_rows, args.data_dir, border_mask=8)
        masked_probability = _predict_tiny(image_model, masked, metrics["iid_heldout"], image_mean, image_scale, 0)
        metadata = np.asarray(
            [
                [IMAGE_SIZE, IMAGE_SIZE, 1, int(row["sample_id"][:2], 16) / 255.0]
                for row in rows["train"]
            ],
            dtype=np.float64,
        )
        metadata_eval = np.asarray(
            [[IMAGE_SIZE, IMAGE_SIZE, 1, int(row["sample_id"][:2], 16) / 255.0] for row in eval_rows],
            dtype=np.float64,
        )
        metadata_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=args.seed))
        metadata_model.fit(metadata, labels["train"])
        metadata_probability = metadata_model.predict_proba(metadata_eval)[:, 1]
        train_hashes = {sha256_file(args.data_dir / row["model_input"]["image_ref"]) for row in rows["train"]}
        eval_hashes = {sha256_file(args.data_dir / row["model_input"]["image_ref"]) for row in eval_rows}
        train_setups = {row["setup_id"] for row in rows["train"]}
        iid_setups = {row["setup_id"] for row in eval_rows}
        ood_setups = {row["setup_id"] for row in rows["severity_ood"]}
        flat_train = images["train"].reshape(len(images["train"]), -1)
        flat_eval = eval_images.reshape(len(eval_images), -1)
        nearest_mse = np.min(np.mean((flat_eval[:, None, :] - flat_train[None, :, :]) ** 2, axis=2), axis=1)
        family_result["leakage_tests"] = {
            "shuffle_images_keep_metrics": _score(labels["iid_heldout"], shuffled_probability),
            "mask_eight_pixel_borders": _score(labels["iid_heldout"], masked_probability),
            "metadata_only_classifier": _score(labels["iid_heldout"], metadata_probability),
            "setup_disjointness": {
                "train_iid_overlap": sorted(train_setups & iid_setups),
                "train_severity_ood_overlap": sorted(train_setups & ood_setups),
                "iid_severity_ood_overlap": sorted(iid_setups & ood_setups),
                "passed": not (train_setups & iid_setups or train_setups & ood_setups or iid_setups & ood_setups),
            },
            "nearest_neighbors_across_train_iid": {
                "exact_image_hash_overlap": len(train_hashes & eval_hashes),
                "minimum_pixel_mse": float(nearest_mse.min()),
                "median_nearest_pixel_mse": float(np.median(nearest_mse)),
            },
            "filenames_contain_fault_labels": False,
            "serialization_order_randomized": True,
        }
        iid = family_result["splits"]["iid_heldout"]
        best_metrics = max(iid["metrics_only_logistic"]["balanced_accuracy"], iid["metrics_history_mlp"]["balanced_accuracy"])
        best_image = max(iid["image_only_small_cnn"]["balanced_accuracy"], iid["image_metrics_multimodal_small_model"]["balanced_accuracy"])
        family_result["visual_identifiability_gate"] = {
            "metrics_only_at_most_65_percent_or_matched_pair_failure": best_metrics <= 0.65,
            "image_at_least_80_percent": best_image >= 0.80,
            "image_minus_metrics_at_least_15_points": best_image - best_metrics >= 0.15,
            "leakage_checks_pass": (
                family_result["leakage_tests"]["setup_disjointness"]["passed"]
                and family_result["leakage_tests"]["nearest_neighbors_across_train_iid"]["exact_image_hash_overlap"] == 0
                and family_result["leakage_tests"]["mask_eight_pixel_borders"]["balanced_accuracy"] >= 0.80
                and family_result["leakage_tests"]["shuffle_images_keep_metrics"]["balanced_accuracy"] <= 0.65
                and family_result["leakage_tests"]["metadata_only_classifier"]["balanced_accuracy"] <= 0.65
            ),
        }
        family_result["visual_identifiability_gate"]["passed"] = all(family_result["visual_identifiability_gate"].values())
        results["families"][family] = family_result
    atomic_json(args.output, results)
    print(json.dumps({family: value["visual_identifiability_gate"] for family, value in results["families"].items()}, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--suite", type=Path, required=True)
    generate.add_argument("--split", choices=("train", "iid_heldout", "severity_ood"), required=True)
    generate.add_argument("--v12-config", type=Path, required=True)
    generate.add_argument("--base-config", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    train = subparsers.add_parser("train-baselines")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=2026081201)
    train.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate_split(args)
    else:
        train_baselines(args)


if __name__ == "__main__":
    main()
