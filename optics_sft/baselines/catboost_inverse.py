"""Simple CatBoost regressors for inverse-control control_plan prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from optics_sft.eval.control_metrics import CONTROL_KEYS, LENS_KEYS


NUMERIC_METADATA_KEYS = (
    "wavelength_nm",
    "beam_waist_mm",
    "power_w",
    "lens_focal_length_mm",
    "lens_aperture_mm",
    "source_to_lens_mm",
    "lens_to_camera_mm",
    "pixel_size_um",
    "grid_size",
    "grid_extent_mm",
)

DEFAULT_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 400,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
}


def _float(value: Any, name: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"Expected numeric {name}, got {value!r}")


def featurize_row(row: Mapping[str, Any]) -> dict[str, float]:
    """Build a flat numeric feature dict from a text SFT row."""
    prompt_inputs = row.get("prompt_inputs")
    if not isinstance(prompt_inputs, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} missing prompt_inputs")
    observations = prompt_inputs.get("text_observations")
    metadata = prompt_inputs.get("safe_setup_metadata")
    if not isinstance(observations, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} missing text_observations or safe_setup_metadata")

    current = observations.get("current")
    target = observations.get("target")
    if not isinstance(current, Mapping) or not isinstance(target, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} missing current/target observations")

    cur_x = _float(current["x_px"], "current.x_px")
    cur_y = _float(current["y_px"], "current.y_px")
    cur_wx = _float(current["width_x_px"], "current.width_x_px")
    cur_wy = _float(current["width_y_px"], "current.width_y_px")
    tgt_x = _float(target["x_px"], "target.x_px")
    tgt_y = _float(target["y_px"], "target.y_px")
    tgt_wx = _float(target["width_x_px"], "target.width_x_px")
    tgt_wy = _float(target["width_y_px"], "target.width_y_px")

    delta_x = cur_x - tgt_x
    delta_y = cur_y - tgt_y
    shift_x = tgt_x - cur_x
    shift_y = tgt_y - cur_y

    features: dict[str, float] = {
        "cur_x_px": cur_x,
        "cur_y_px": cur_y,
        "cur_width_x_px": cur_wx,
        "cur_width_y_px": cur_wy,
        "tgt_x_px": tgt_x,
        "tgt_y_px": tgt_y,
        "tgt_width_x_px": tgt_wx,
        "tgt_width_y_px": tgt_wy,
        "delta_x_px": delta_x,
        "delta_y_px": delta_y,
        "shift_x_px": shift_x,
        "shift_y_px": shift_y,
        "delta_width_x_px": cur_wx - tgt_wx,
        "delta_width_y_px": cur_wy - tgt_wy,
        "error_norm_px": (delta_x**2 + delta_y**2) ** 0.5,
        "width_ratio_x": cur_wx / tgt_wx if tgt_wx else 1.0,
        "width_ratio_y": cur_wy / tgt_wy if tgt_wy else 1.0,
    }

    for key in NUMERIC_METADATA_KEYS:
        if key in metadata:
            features[f"meta_{key}"] = _float(metadata[key], key)

    resolution = metadata.get("sensor_resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        features["meta_sensor_height_px"] = _float(resolution[0], "sensor_resolution[0]")
        features["meta_sensor_width_px"] = _float(resolution[1], "sensor_resolution[1]")

    return features


def feature_names(sample_row: Mapping[str, Any]) -> list[str]:
    return list(featurize_row(sample_row).keys())


def label_control_plan(row: Mapping[str, Any]) -> dict[str, float]:
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} missing target")
    plan = target.get("control_plan")
    if not isinstance(plan, Mapping):
        raise ValueError(f"Row {row.get('sample_id')} missing target.control_plan")
    return {key: _float(plan[key], key) for key in CONTROL_KEYS}


def rows_to_matrices(
    rows: list[dict[str, Any]],
) -> tuple[list[list[float]], list[str], dict[str, list[float]]]:
    if not rows:
        raise ValueError("Need at least one row to build feature matrices")
    names = feature_names(rows[0])
    x_matrix: list[list[float]] = []
    y_by_key: dict[str, list[float]] = {key: [] for key in CONTROL_KEYS}
    for row in rows:
        features = featurize_row(row)
        x_matrix.append([features[name] for name in names])
        plan = label_control_plan(row)
        for key in CONTROL_KEYS:
            y_by_key[key].append(plan[key])
    return x_matrix, names, y_by_key


def require_catboost() -> Any:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is not installed. Run: pip install catboost"
        ) from exc
    return CatBoostRegressor


def train_models(
    train_rows: list[dict[str, Any]],
    *,
    val_rows: list[dict[str, Any]] | None = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    CatBoostRegressor = require_catboost()
    x_train, feature_cols, y_train = rows_to_matrices(train_rows)
    eval_set = None
    if val_rows:
        x_val, _, y_val = rows_to_matrices(val_rows)
        eval_set = (x_val, y_val)

    merged_params = dict(DEFAULT_PARAMS)
    if params:
        merged_params.update(params)

    models: dict[str, Any] = {}
    for key in LENS_KEYS:
        targets = y_train[key]
        if len(set(round(value, 12) for value in targets)) < 2:
            raise ValueError(f"Training targets for {key} are constant; cannot fit CatBoost")
        model = CatBoostRegressor(**merged_params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = (eval_set[0], eval_set[1][key])
            fit_kwargs["use_best_model"] = True
        model.fit(x_train, targets, **fit_kwargs)
        models[key] = model
    return models, feature_cols


def predict_control_plans(
    models: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    x_matrix, names, _ = rows_to_matrices(rows)
    if names != feature_cols:
        raise ValueError("Feature columns do not match trained model")

    predictions: list[dict[str, Any]] = []
    per_key_preds = {key: models[key].predict(x_matrix) for key in LENS_KEYS}
    for index, row in enumerate(rows):
        plan = {key: 0.0 for key in CONTROL_KEYS}
        for key in LENS_KEYS:
            plan[key] = round(float(per_key_preds[key][index]), 8)
        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "parsed_json": {
                    "task": "physics_aware_beam_alignment",
                    "control_plan": plan,
                },
            }
        )
    return predictions


def save_models(
    models: Mapping[str, Any],
    output_dir: Path,
    *,
    feature_cols: list[str],
    params: Mapping[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, model in models.items():
        model.save_model(str(output_dir / f"{key}.cbm"))
    meta = {
        "feature_cols": feature_cols,
        "control_keys": list(LENS_KEYS),
        "fixed_zero_keys": ["camera_x_delta_mm", "camera_y_delta_mm"],
        "params": dict(params or DEFAULT_PARAMS),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_models(model_dir: Path) -> tuple[dict[str, Any], list[str]]:
    CatBoostRegressor = require_catboost()
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_cols = list(meta["feature_cols"])
    models: dict[str, Any] = {}
    for key in meta.get("control_keys", list(LENS_KEYS)):
        model = CatBoostRegressor()
        model.load_model(str(model_dir / f"{key}.cbm"))
        models[key] = model
    return models, feature_cols
