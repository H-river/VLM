"""Deterministic state-evidence extraction from calibrated instrumented beam images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


STATE_FIELDS = (
    "centroid_horizontal_region",
    "centroid_vertical_region",
    "sigma_x_band",
    "sigma_y_band",
)


def grayscale_without_overlays(
    image_path: Path, *, overlay_handling: str = "mask_zero"
) -> np.ndarray:
    """Return beam grayscale while removing known colored calibration aids."""

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64)
    if overlay_handling == "mask_zero":
        return rgb.min(axis=2)
    if overlay_handling != "interpolate_colored_aids":
        raise ValueError(f"Unsupported overlay handling: {overlay_handling}")
    mask = np.ptp(rgb, axis=2) > 1.0
    values = rgb.mean(axis=2)
    values[mask] = 0.0
    unresolved = mask.copy()
    for _ in range(8):
        if not unresolved.any():
            break
        valid = ~unresolved
        padded_values = np.pad(values, 1, mode="edge")
        padded_valid = np.pad(valid.astype(np.float64), 1, mode="constant")
        neighbor_sum = np.zeros_like(values)
        neighbor_count = np.zeros_like(values)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ys = slice(1 + dy, 1 + dy + values.shape[0])
            xs = slice(1 + dx, 1 + dx + values.shape[1])
            neighbor_sum += padded_values[ys, xs] * padded_valid[ys, xs]
            neighbor_count += padded_valid[ys, xs]
        fillable = unresolved & (neighbor_count > 0)
        values[fillable] = neighbor_sum[fillable] / neighbor_count[fillable]
        unresolved[fillable] = False
    if unresolved.any():
        values[unresolved] = 0.0
    return values


def neighbor_indices_with_ties(distances: np.ndarray, k: int) -> np.ndarray:
    """Return at least k neighbors and include every exact-distance boundary tie."""

    ordered = np.argsort(distances, kind="stable")
    cutoff = float(distances[ordered[min(k, len(ordered)) - 1]])
    return np.flatnonzero(np.isclose(distances, cutoff, rtol=1e-12, atol=1e-15) | (distances < cutoff))


def extract_features(
    image_path: Path,
    *,
    sensor_crop_px: int = 512,
    background_percentile: float = 25.0,
    overlay_handling: str = "mask_zero",
    intensity_power: float = 1.0,
    noise_floor_sigma: float = 0.0,
    relative_floor: float = 0.0,
    denoise_passes: int = 0,
) -> dict[str, float]:
    """Measure centroid and rendered second moments while masking colored aids."""

    grayscale = grayscale_without_overlays(image_path, overlay_handling=overlay_handling)
    if denoise_passes < 0:
        raise ValueError("denoise_passes must be nonnegative")
    for _ in range(denoise_passes):
        padded_x = np.pad(grayscale, ((0, 0), (1, 1)), mode="reflect")
        grayscale = (
            padded_x[:, :-2] + 2.0 * padded_x[:, 1:-1] + padded_x[:, 2:]
        ) / 4.0
        padded_y = np.pad(grayscale, ((1, 1), (0, 0)), mode="reflect")
        grayscale = (
            padded_y[:-2, :] + 2.0 * padded_y[1:-1, :] + padded_y[2:, :]
        ) / 4.0
    if noise_floor_sigma < 0.0 or not 0.0 <= relative_floor < 1.0:
        raise ValueError("noise floor parameters are outside their supported range")
    background = float(np.percentile(grayscale, background_percentile))
    threshold = background
    if noise_floor_sigma > 0.0 or relative_floor > 0.0:
        border_width = max(1, min(grayscale.shape) // 10)
        border = np.concatenate(
            [
                grayscale[:border_width, :].reshape(-1),
                grayscale[-border_width:, :].reshape(-1),
                grayscale[border_width:-border_width, :border_width].reshape(-1),
                grayscale[border_width:-border_width, -border_width:].reshape(-1),
            ]
        )
        background = float(np.median(border))
        deviations = np.abs(border - background)
        robust_sigma = max(
            1.4826 * float(np.median(deviations)),
            float(np.percentile(deviations, 84.0)),
        )
        threshold = max(
            background + noise_floor_sigma * robust_sigma,
            background + relative_floor * max(float(grayscale.max()) - background, 0.0),
        )
    if intensity_power <= 0.0:
        raise ValueError("intensity_power must be positive")
    signal = np.maximum(grayscale - background, 0.0)
    signal[grayscale < threshold] = 0.0
    weights = np.power(signal, intensity_power)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"No measurable beam signal: {image_path}")
    yy, xx = np.indices(weights.shape)
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    sigma_x = float(np.sqrt((weights * (xx - cx) ** 2).sum() / total))
    sigma_y = float(np.sqrt((weights * (yy - cy) ** 2).sum() / total))
    height, width = weights.shape
    crop_origin = (1024.0 - sensor_crop_px) / 2.0
    return {
        "centroid_x_px": (cx + 0.5) * sensor_crop_px / width - 0.5 + crop_origin,
        "centroid_y_px": (cy + 0.5) * sensor_crop_px / height - 0.5 + crop_origin,
        "rendered_sigma_x_px": sigma_x * sensor_crop_px / width,
        "rendered_sigma_y_px": sigma_y * sensor_crop_px / height,
    }


def extract_projected_features(
    image_path: Path,
    *,
    sensor_crop_px: int = 512,
    overlay_handling: str = "mask_zero",
    intensity_power: float = 1.0,
    noise_floor_sigma: float = 0.0,
    relative_floor: float = 0.0,
    denoise_passes: int = 0,
) -> dict[str, float]:
    """Measure centroids and widths after signed pixels are projected to 1-D profiles.

    Summing before nonnegative clipping allows zero-mean sensor noise to cancel instead
    of accumulating a positive full-frame moment bias.
    """
    if intensity_power <= 0.0:
        raise ValueError("intensity_power must be positive")
    if noise_floor_sigma < 0.0 or not 0.0 <= relative_floor < 1.0:
        raise ValueError("noise floor parameters are outside their supported range")
    grayscale = grayscale_without_overlays(image_path, overlay_handling=overlay_handling)
    if denoise_passes < 0:
        raise ValueError("denoise_passes must be nonnegative")
    for _ in range(denoise_passes):
        padded_x = np.pad(grayscale, ((0, 0), (1, 1)), mode="reflect")
        grayscale = (
            padded_x[:, :-2] + 2.0 * padded_x[:, 1:-1] + padded_x[:, 2:]
        ) / 4.0
        padded_y = np.pad(grayscale, ((1, 1), (0, 0)), mode="reflect")
        grayscale = (
            padded_y[:-2, :] + 2.0 * padded_y[1:-1, :] + padded_y[2:, :]
        ) / 4.0
    border_width = max(1, min(grayscale.shape) // 10)
    border = np.concatenate(
        [
            grayscale[:border_width, :].reshape(-1),
            grayscale[-border_width:, :].reshape(-1),
            grayscale[border_width:-border_width, :border_width].reshape(-1),
            grayscale[border_width:-border_width, -border_width:].reshape(-1),
        ]
    )
    signal = grayscale - float(np.median(border))

    def profile_moments(profile: np.ndarray) -> tuple[float, float]:
        edge_width = max(1, len(profile) // 10)
        edges = np.concatenate([profile[:edge_width], profile[-edge_width:]])
        background = float(np.median(edges))
        deviations = np.abs(edges - background)
        robust_sigma = max(
            1.4826 * float(np.median(deviations)),
            float(np.percentile(deviations, 84.0)),
        )
        threshold = max(
            background + noise_floor_sigma * robust_sigma,
            background + relative_floor * max(float(profile.max()) - background, 0.0),
        )
        positive = np.maximum(profile - background, 0.0)
        positive[profile < threshold] = 0.0
        weights = np.power(positive, intensity_power)
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError(f"No measurable projected beam signal: {image_path}")
        coordinate = np.arange(len(profile), dtype=np.float64)
        centroid = float((weights * coordinate).sum() / total)
        sigma = float(np.sqrt((weights * np.square(coordinate - centroid)).sum() / total))
        return centroid, sigma

    cx, sigma_x = profile_moments(signal.sum(axis=0))
    cy, sigma_y = profile_moments(signal.sum(axis=1))
    height, width = grayscale.shape
    crop_origin = (1024.0 - sensor_crop_px) / 2.0
    return {
        "centroid_x_px": (cx + 0.5) * sensor_crop_px / width - 0.5 + crop_origin,
        "centroid_y_px": (cy + 0.5) * sensor_crop_px / height - 0.5 + crop_origin,
        "rendered_sigma_x_px": sigma_x * sensor_crop_px / width,
        "rendered_sigma_y_px": sigma_y * sensor_crop_px / height,
    }


def fit_ordered_thresholds(
    values_and_labels: Sequence[tuple[float, str]],
    labels: Sequence[str],
    *,
    objective: str = "accuracy",
) -> tuple[float, float, float]:
    """Fit two monotonic thresholds with an exact quadratic cumulative search."""

    if len(labels) != 3:
        raise ValueError("Ordered threshold fitting requires exactly three labels")
    if objective not in {"accuracy", "macro_f1"}:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    values = sorted({float(value) for value, _ in values_and_labels})
    if len(values) < 3:
        raise ValueError("Not enough distinct features to fit thresholds")
    value_index = {value: index for index, value in enumerate(values)}
    label_index = {str(label): index for index, label in enumerate(labels)}
    counts = np.zeros((len(values), 3), dtype=np.int64)
    for value, label in values_and_labels:
        counts[value_index[float(value)], label_index[str(label)]] += 1
    cumulative = counts.cumsum(axis=0)
    total = cumulative[-1]
    candidates = [(left + right) / 2.0 for left, right in zip(values, values[1:])]
    best = (-1.0, 0.0, 0.0)
    for first_index, first in enumerate(candidates):
        low_correct = int(cumulative[first_index, 0])
        for second_index in range(first_index + 1, len(candidates)):
            middle_correct = int(
                cumulative[second_index, 1] - cumulative[first_index, 1]
            )
            high_correct = int(total[2] - cumulative[second_index, 2])
            if objective == "accuracy":
                score = (low_correct + middle_correct + high_correct) / len(values_and_labels)
            else:
                confusion = np.zeros((3, 3), dtype=np.int64)
                confusion[:, 0] = cumulative[first_index]
                confusion[:, 1] = cumulative[second_index] - cumulative[first_index]
                confusion[:, 2] = total - cumulative[second_index]
                f1_values = []
                for class_index in range(3):
                    tp = int(confusion[class_index, class_index])
                    fp = int(confusion[:, class_index].sum()) - tp
                    fn = int(confusion[class_index, :].sum()) - tp
                    denominator = 2 * tp + fp + fn
                    f1_values.append(2 * tp / denominator if denominator else 0.0)
                score = float(np.mean(f1_values))
            if score > best[0]:
                best = (score, first, candidates[second_index])
    if best[0] < 0.0:
        raise ValueError("Not enough distinct features to fit thresholds")
    return best


def classify(features: Mapping[str, float], calibration: Mapping[str, Any]) -> dict[str, str]:
    if calibration.get("classifier") == "group_cv_knn_v1":
        return classify_knn(features, calibration)

    def ordered(value: float, spec: Mapping[str, Any]) -> str:
        low, high = map(float, spec["thresholds"])
        labels = list(spec["labels"])
        return labels[0] if value < low else labels[2] if value > high else labels[1]

    return {
        "centroid_horizontal_region": ordered(
            float(features["centroid_x_px"]), calibration["centroid_x"]
        ),
        "centroid_vertical_region": ordered(
            float(features["centroid_y_px"]), calibration["centroid_y"]
        ),
        "sigma_x_band": ordered(
            float(features["rendered_sigma_x_px"]), calibration["sigma_x"]
        ),
        "sigma_y_band": ordered(
            float(features["rendered_sigma_y_px"]), calibration["sigma_y"]
        ),
    }


def classify_knn(
    features: Mapping[str, float], calibration: Mapping[str, Any]
) -> dict[str, str]:
    """Classify state fields from train-only image-feature prototypes."""

    feature_order = list(calibration["feature_order"])
    center = np.asarray(calibration["normalization"]["mean"], dtype=np.float64)
    scale = np.asarray(calibration["normalization"]["scale"], dtype=np.float64)
    query = (np.asarray([features[key] for key in feature_order]) - center) / scale
    prototypes = list(calibration["prototypes"])
    matrix = np.asarray([row["features"] for row in prototypes], dtype=np.float64)
    matrix = (matrix - center) / scale
    distances = np.square(matrix - query).sum(axis=1)
    result: dict[str, str] = {}
    for field, spec in calibration["fields"].items():
        indices = neighbor_indices_with_ties(distances, int(spec["k"]))
        labels = [str(prototypes[index]["labels"][field]) for index in indices]
        counts = {label: labels.count(label) for label in set(labels)}
        result[str(field)] = min(counts, key=lambda label: (-counts[label], label))
    return result


def load_calibration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Calibration must be a JSON object")
    return value


def extract_differential_features(
    first_path: Path,
    second_path: Path,
    *,
    sensor_crop_px: int = 512,
    overlay_handling: str = "mask_zero",
    noise_floor_sigma: float = 3.0,
) -> dict[str, float]:
    """Extract signed pair-difference geometry after common overlays cancel."""
    first = grayscale_without_overlays(first_path, overlay_handling=overlay_handling)
    second = grayscale_without_overlays(second_path, overlay_handling=overlay_handling)
    if first.shape != second.shape:
        raise ValueError("paired images must have identical dimensions")
    difference = second - first
    border_width = max(1, min(difference.shape) // 10)
    border = np.concatenate(
        [
            difference[:border_width, :].reshape(-1),
            difference[-border_width:, :].reshape(-1),
            difference[border_width:-border_width, :border_width].reshape(-1),
            difference[border_width:-border_width, -border_width:].reshape(-1),
        ]
    )
    background = float(np.median(border))
    deviations = np.abs(border - background)
    robust_sigma = max(
        1.4826 * float(np.median(deviations)),
        float(np.percentile(deviations, 84.0)),
    )
    centered = difference - background
    floor = noise_floor_sigma * robust_sigma
    positive = np.where(centered >= floor, centered, 0.0)
    negative = np.where(centered <= -floor, -centered, 0.0)
    height, width = centered.shape
    yy, xx = np.indices(centered.shape)
    sensor_scale_x = sensor_crop_px / width
    sensor_scale_y = sensor_crop_px / height
    crop_origin = (1024.0 - sensor_crop_px) / 2.0

    def moments(weights: np.ndarray) -> tuple[float, float, float, float, float]:
        total = float(weights.sum())
        if total <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)
        sigma_x = float(np.sqrt((weights * np.square(xx - cx)).sum() / total))
        sigma_y = float(np.sqrt((weights * np.square(yy - cy)).sum() / total))
        return (
            total,
            (cx + 0.5) * sensor_scale_x - 0.5 + crop_origin,
            (cy + 0.5) * sensor_scale_y - 0.5 + crop_origin,
            sigma_x * sensor_scale_x,
            sigma_y * sensor_scale_y,
        )

    pos_total, pos_cx, pos_cy, pos_sx, pos_sy = moments(positive)
    neg_total, neg_cx, neg_cy, neg_sx, neg_sy = moments(negative)
    combined = max(pos_total + neg_total, 1e-12)
    return {
        "diff_positive_total_fraction": pos_total / combined,
        "diff_positive_centroid_x": pos_cx,
        "diff_positive_centroid_y": pos_cy,
        "diff_positive_sigma_x": pos_sx,
        "diff_positive_sigma_y": pos_sy,
        "diff_negative_centroid_x": neg_cx,
        "diff_negative_centroid_y": neg_cy,
        "diff_negative_sigma_x": neg_sx,
        "diff_negative_sigma_y": neg_sy,
        "diff_sigma_x_contrast": pos_sx - neg_sx,
        "diff_sigma_y_contrast": pos_sy - neg_sy,
    }


def extract_pair_features(
    first_path: Path,
    second_path: Path,
    *,
    sensor_crop_px: int = 512,
    peak_feature: str = "energy_squared_ratio",
    overlay_handling: str = "mask_zero",
    sigma_x_power: float = 1.0,
    sigma_y_power: float = 1.0,
    noise_floor_sigma: float = 0.0,
    relative_floor: float = 0.0,
    width_estimator: str = "moments_2d",
    include_differential_features: bool = False,
    differential_floor_sigma: float = 3.0,
    denoise_passes: int = 0,
) -> dict[str, float]:
    """Extract calibrated pair deltas used by the deterministic threshold tool."""

    first = extract_features(
        first_path,
        sensor_crop_px=sensor_crop_px,
        overlay_handling=overlay_handling,
        noise_floor_sigma=noise_floor_sigma,
        relative_floor=relative_floor,
        denoise_passes=denoise_passes,
    )
    second = extract_features(
        second_path,
        sensor_crop_px=sensor_crop_px,
        overlay_handling=overlay_handling,
        noise_floor_sigma=noise_floor_sigma,
        relative_floor=relative_floor,
        denoise_passes=denoise_passes,
    )
    if width_estimator not in {"moments_2d", "projected_1d"}:
        raise ValueError(f"Unsupported width estimator: {width_estimator}")

    def extract_width(path: Path, power: float) -> dict[str, float]:
        extractor = extract_features if width_estimator == "moments_2d" else extract_projected_features
        return extractor(
            path,
            sensor_crop_px=sensor_crop_px,
            overlay_handling=overlay_handling,
            intensity_power=power,
            noise_floor_sigma=noise_floor_sigma,
            relative_floor=relative_floor,
            denoise_passes=denoise_passes,
        )

    if sigma_x_power == sigma_y_power and (
        sigma_x_power != 1.0 or width_estimator != "moments_2d"
    ):
        first_x = first_y = extract_width(
            first_path,
            sigma_x_power,
        )
        second_x = second_y = extract_width(
            second_path,
            sigma_x_power,
        )
    elif (
        sigma_x_power != 1.0
        or sigma_y_power != 1.0
        or width_estimator != "moments_2d"
    ):
        first_x = (
            first
            if sigma_x_power == 1.0 and width_estimator == "moments_2d"
            else extract_width(first_path, sigma_x_power)
        )
        second_x = (
            second
            if sigma_x_power == 1.0 and width_estimator == "moments_2d"
            else extract_width(second_path, sigma_x_power)
        )
        first_y = (
            first
            if sigma_y_power == 1.0 and width_estimator == "moments_2d"
            else extract_width(first_path, sigma_y_power)
        )
        second_y = (
            second
            if sigma_y_power == 1.0 and width_estimator == "moments_2d"
            else extract_width(second_path, sigma_y_power)
        )
    else:
        first_x = first_y = first
        second_x = second_y = second

    def peak_proxy(path: Path) -> float:
        grayscale = grayscale_without_overlays(path, overlay_handling=overlay_handling)
        if peak_feature == "energy_squared_ratio":
            return float(np.square(grayscale).sum())
        if peak_feature == "top10_mean_ratio":
            flattened = grayscale.reshape(-1)
            count = min(10, flattened.size)
            return float(np.partition(flattened, -count)[-count:].mean())
        raise ValueError(f"Unsupported paired peak feature: {peak_feature}")

    first_peak_raw = peak_proxy(first_path)
    second_peak = peak_proxy(second_path)
    first_peak = max(first_peak_raw, 1e-12)
    result = {
        "centroid_x": second["centroid_x_px"] - first["centroid_x_px"],
        "centroid_y": second["centroid_y_px"] - first["centroid_y_px"],
        "sigma_x": second_x["rendered_sigma_x_px"] - first_x["rendered_sigma_x_px"],
        "sigma_y": second_y["rendered_sigma_y_px"] - first_y["rendered_sigma_y_px"],
        "peak_intensity": second_peak / first_peak - 1.0,
        "centroid_x_first": first["centroid_x_px"],
        "centroid_x_second": second["centroid_x_px"],
        "centroid_y_first": first["centroid_y_px"],
        "centroid_y_second": second["centroid_y_px"],
        "sigma_x_first": first_x["rendered_sigma_x_px"],
        "sigma_x_second": second_x["rendered_sigma_x_px"],
        "sigma_y_first": first_y["rendered_sigma_y_px"],
        "sigma_y_second": second_y["rendered_sigma_y_px"],
        "peak_proxy_first": first_peak_raw,
        "peak_proxy_second": second_peak,
    }
    if include_differential_features:
        result.update(
            extract_differential_features(
                first_path,
                second_path,
                sensor_crop_px=sensor_crop_px,
                overlay_handling=overlay_handling,
                noise_floor_sigma=differential_floor_sigma,
            )
        )
    return result


def classify_pair(
    features: Mapping[str, float], calibration: Mapping[str, Any]
) -> dict[str, str]:
    if calibration.get("classifier") == "group_cv_hybrid_v1":
        feature_order = list(calibration["feature_order"])
        center = np.asarray(calibration["normalization"]["mean"], dtype=np.float64)
        scale = np.asarray(calibration["normalization"]["scale"], dtype=np.float64)
        query = (np.asarray([features[key] for key in feature_order]) - center) / scale
        prototypes = list(calibration["prototypes"])
        matrix = np.asarray([row["features"] for row in prototypes], dtype=np.float64)
        distances = np.square((matrix - center) / scale - query).sum(axis=1)
        result: dict[str, str] = {}
        for field, spec in calibration["fields"].items():
            if spec["method"] == "threshold":
                low, high = map(float, spec["thresholds"])
                value = float(features[field])
                result[field] = (
                    "decrease" if value < low else "increase" if value > high else "no_change"
                )
            else:
                indices = neighbor_indices_with_ties(distances, int(spec["k"]))
                labels = [str(prototypes[index]["labels"][field]) for index in indices]
                counts = {label: labels.count(label) for label in set(labels)}
                result[field] = min(counts, key=lambda label: (-counts[label], label))
        return result
    result: dict[str, str] = {}
    for field in ("centroid_x", "centroid_y", "sigma_x", "sigma_y", "peak_intensity"):
        low, high = map(float, calibration[field]["thresholds"])
        value = float(features[field])
        result[field] = "decrease" if value < low else "increase" if value > high else "no_change"
    return result
