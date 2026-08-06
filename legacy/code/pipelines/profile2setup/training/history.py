"""Training history persistence and plotting helpers for profile2setup v2."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from profile2setup.schema import VARIABLE_ORDER


BASE_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_absolute_loss",
    "train_delta_loss",
    "train_change_loss",
    "train_constraint_loss",
    "val_absolute_loss",
    "val_delta_loss",
    "val_change_loss",
    "val_constraint_loss",
]


def _clean_scalar(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().item()
    except Exception:
        pass

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return float(value) if math.isfinite(value) else None


def _add_scalar(row: dict, key: str, value: Any) -> None:
    cleaned = _clean_scalar(value)
    if cleaned is not None:
        row[key] = cleaned


def _add_metric_dict(row: dict, prefix: str, values: Any) -> None:
    if not isinstance(values, dict):
        return
    for name in VARIABLE_ORDER:
        if name in values:
            _add_scalar(row, f"{prefix}_{name}", values[name])


def append_history_row(
    history: list[dict],
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
) -> dict:
    """Flatten one epoch of metrics, append it to history, and return the row."""
    row: dict[str, Any] = {"epoch": int(epoch)}

    for key, value in train_metrics.items():
        if isinstance(value, (dict, list, tuple)):
            continue
        _add_scalar(row, key, value)

    for key, value in val_metrics.items():
        if key in {"examples", "routed"}:
            continue
        if key in {"absolute_mae", "delta_mae", "routed_setup_mae"}:
            _add_metric_dict(row, f"val_{key}", value)
        elif key == "active_counts" and isinstance(value, dict):
            for count_key, count_value in value.items():
                _add_scalar(row, f"val_active_count_{count_key}", count_value)
        elif not isinstance(value, (dict, list, tuple)):
            _add_scalar(row, key, value)

    history.append(row)
    return row


def _history_columns(history: list[dict]) -> list[str]:
    seen = set()
    columns = []
    for key in BASE_COLUMNS:
        if any(key in row for row in history):
            columns.append(key)
            seen.add(key)
    for row in history:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    return columns


def save_history_csv(history: list[dict], path) -> None:
    """Write training history to CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _history_columns(history)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in columns})


def save_history_json(history: list[dict], path) -> None:
    """Write training history to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2, sort_keys=True)


def _series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            xs.append(int(row["epoch"]))
            ys.append(float(value))
    return xs, ys


def _plot_lines(ax, history: list[dict], keys: list[str]) -> bool:
    plotted = False
    for key in keys:
        xs, ys = _series(history, key)
        if xs:
            ax.plot(xs, ys, marker="o", label=key)
            plotted = True
    if plotted:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()
    return plotted


def _save_loss_curve(history: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    if _plot_lines(ax, history, ["train_loss", "val_loss"]):
        ax.set_title("Loss")
        ax.set_ylabel("loss")
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


def _save_component_curves(history: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = [
        "train_absolute_loss",
        "val_absolute_loss",
        "train_delta_loss",
        "val_delta_loss",
        "train_change_loss",
        "val_change_loss",
        "train_constraint_loss",
        "val_constraint_loss",
    ]
    if _plot_lines(ax, history, keys):
        ax.set_title("Component Losses")
        ax.set_ylabel("loss")
        fig.tight_layout()
        fig.savefig(out_dir / "component_loss_curves.png", dpi=160)
    plt.close(fig)


def _save_mae_curves(history: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metric_prefixes = [
        "val_routed_setup_mae",
        "val_absolute_mae",
        "val_delta_mae",
    ]
    keys = [
        f"{prefix}_{name}"
        for prefix in metric_prefixes
        for name in VARIABLE_ORDER
        if any(f"{prefix}_{name}" in row for row in history)
    ]
    if not keys:
        return

    fig, axes = plt.subplots(len(metric_prefixes), 1, figsize=(9, 9), sharex=True)
    for ax, prefix in zip(axes, metric_prefixes):
        plotted = _plot_lines(ax, history, [f"{prefix}_{name}" for name in VARIABLE_ORDER])
        if plotted:
            ax.set_title(prefix)
            ax.set_ylabel("normalized MAE")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "mae_curves.png", dpi=160)
    plt.close(fig)


def plot_training_history(history: list[dict], out_dir) -> list[str]:
    """Generate available training-history plots with matplotlib."""
    if not history:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        return []

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    before = set(out_path.glob("*.png"))
    _save_loss_curve(history, out_path)
    _save_component_curves(history, out_path)
    _save_mae_curves(history, out_path)
    after = set(out_path.glob("*.png"))
    return [str(path) for path in sorted(after - before)]
