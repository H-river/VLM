"""Interactive predicted-vs-groundtruth profile viewer for LLM API outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from profile2setup.evaluation.llm_api_visualization import simulate_api_prediction
from profile2setup.evaluation.profile_metrics import load_intensity
from profile2setup.llm_api.image_rendering import normalize_intensity
from profile2setup.schema import VARIABLE_ORDER, validate_setup_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show LLM API simulated predicted profiles next to ground-truth target "
            "profiles. Press 'e' to advance to the next pair."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="LLM API prediction JSONL, e.g. profile2setup/results/llm_api_predictions/finetuned_small.jsonl",
    )
    parser.add_argument(
        "--data",
        default="profile2setup/data/all_modes/test.jsonl",
        help="Ground-truth profile2setup JSONL used for the predictions",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root for resolving relative dataset paths",
    )
    parser.add_argument(
        "--simulation-policy",
        choices=("target_base", "current_base", "auto"),
        default="target_base",
        help="Simulator base-config policy used before applying the LLM predicted setup",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to show")
    parser.add_argument("--start-index", type=int, default=0, help="Prediction-row offset to start from")
    parser.add_argument("--image-size", type=int, default=320, help="Displayed panel size in pixels")
    parser.add_argument("--save-dir", default=None, help="Optional directory to save left/right PNG pairs")
    parser.add_argument("--no-window", action="store_true", help="Only write --save-dir outputs; do not open a GUI")
    parser.add_argument("--strict", action="store_true", help="Raise on the first skipped row instead of continuing")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            obj.setdefault("_line_number", line_number)
            records.append(obj)
    return records


def _parse_prediction(row: dict[str, Any]) -> dict[str, Any] | None:
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        return prediction
    raw = row.get("raw_response")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _canonical_setup(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    if not validate_setup_dict(value):
        return None
    return {name: float(value[name]) for name in VARIABLE_ORDER}


def _predicted_setup(prediction: dict[str, Any], data_record: dict[str, Any]) -> dict[str, float] | None:
    setup = _canonical_setup(prediction.get("predicted_setup"))
    if setup is not None:
        return setup
    delta = _canonical_setup(prediction.get("predicted_delta"))
    current = _canonical_setup(data_record.get("current_setup"))
    if delta is None or current is None:
        return None
    return {name: float(current[name] + delta[name]) for name in VARIABLE_ORDER}


def _record_id(row: dict[str, Any]) -> str | None:
    value = row.get("record_id")
    if value:
        return str(value)
    prediction = _parse_prediction(row)
    if isinstance(prediction, dict) and prediction.get("record_id"):
        return str(prediction["record_id"])
    return None


def _safe_filename(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "record")).strip("._")
    return safe or "record"


def _target_profile_path(record: dict[str, Any], repo_root: Path) -> Path | None:
    value = record.get("target_profile_path")
    if not value:
        ref = record.get("profile_loss_reference") or {}
        if isinstance(ref, dict):
            value = ref.get("target_profile_path")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _resize_gray(array: np.ndarray, size: int) -> Image.Image:
    display = normalize_intensity(array, mode="max")
    pixels = np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    if image.size != (size, size):
        resample = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((size, size), resample=resample)
    return image.convert("RGB")


def _labeled_panel(image: Image.Image, label: str) -> Image.Image:
    label_height = 30
    canvas = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    x = max((canvas.width - (bbox[2] - bbox[0])) // 2, 0)
    y = max((label_height - (bbox[3] - bbox[1])) // 2 - 1, 0)
    draw.text((x, y), label, fill=(0, 0, 0), font=font)
    return canvas


def _pair_image(predicted: np.ndarray, groundtruth: np.ndarray, size: int) -> Image.Image:
    panels = [
        _labeled_panel(_resize_gray(predicted, size), "PREDICTED"),
        _labeled_panel(_resize_gray(groundtruth, size), "GROUNDTRUTH"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def _show_window(pairs: list[tuple[Image.Image, str]]) -> None:
    import matplotlib.pyplot as plt

    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    state = {"index": 0}
    image_artist = ax.imshow(pairs[0][0])
    ax.set_title(f"{pairs[0][1]}    press e: next    q/escape: quit", fontsize=10)
    ax.axis("off")

    def on_key(event) -> None:
        if event.key == "e":
            next_index = state["index"] + 1
            if next_index >= len(pairs):
                plt.close(fig)
                return
            state["index"] = next_index
            pair, title = pairs[next_index]
            image_artist.set_data(pair)
            ax.set_title(f"{title}    press e: next    q/escape: quit", fontsize=10)
            fig.canvas.draw_idle()
        elif event.key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()


def _build_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    repo_root = Path(args.repo_root).resolve()
    prediction_rows = _load_jsonl(Path(args.predictions))
    data_records = _load_jsonl(Path(args.data))
    data_by_id = {str(record.get("id")): record for record in data_records if record.get("id")}
    selected_rows = prediction_rows[int(args.start_index) :]
    if args.limit is not None:
        selected_rows = selected_rows[: int(args.limit)]

    examples: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, row in enumerate(selected_rows, start=int(args.start_index)):
        rid = _record_id(row)
        if rid is None:
            reason = f"row {index}: missing record_id"
            if args.strict:
                raise ValueError(reason)
            skipped.append(reason)
            continue
        data_record = data_by_id.get(rid)
        if data_record is None:
            reason = f"{rid}: no matching data record in {args.data}"
            if args.strict:
                raise ValueError(reason)
            skipped.append(reason)
            continue
        prediction = _parse_prediction(row)
        if prediction is None:
            reason = f"{rid}: prediction is not valid JSON"
            if args.strict:
                raise ValueError(reason)
            skipped.append(reason)
            continue
        setup = _predicted_setup(prediction, data_record)
        if setup is None:
            reason = f"{rid}: missing canonical predicted_setup or predicted_delta"
            if args.strict:
                raise ValueError(reason)
            skipped.append(reason)
            continue
        target_path = _target_profile_path(data_record, repo_root)
        if target_path is None or not target_path.exists():
            reason = f"{rid}: missing groundtruth target_profile_path"
            if args.strict:
                raise ValueError(reason)
            skipped.append(reason)
            continue
        try:
            sim_result = simulate_api_prediction(
                data_record=data_record,
                predicted_setup=setup,
                simulation_policy=args.simulation_policy,
                repo_root=repo_root,
            )
            predicted = np.asarray(sim_result["predicted_intensity"], dtype=np.float64)
            groundtruth = np.asarray(load_intensity(target_path), dtype=np.float64)
        except Exception as exc:
            reason = f"{rid}: simulator failed: {exc}"
            if args.strict:
                raise RuntimeError(reason) from exc
            skipped.append(reason)
            continue
        examples.append(
            {
                "index": index,
                "record_id": rid,
                "task_type": data_record.get("task_type"),
                "predicted": predicted,
                "groundtruth": groundtruth,
                "normalized_mse": (sim_result.get("profile_metrics") or {}).get("normalized_mse"),
            }
        )
    return examples, skipped


def main() -> None:
    args = parse_args()
    if args.no_window and not args.save_dir:
        raise ValueError("--no-window requires --save-dir")
    examples, skipped = _build_examples(args)
    if not examples:
        raise RuntimeError("No viewable prediction/profile pairs were built")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Image.Image, str]] = []
    for offset, example in enumerate(examples, start=1):
        title = (
            f"{offset}/{len(examples)} {example['record_id']} "
            f"({example.get('task_type')})"
        )
        if example.get("normalized_mse") is not None:
            title += f" normalized_mse={float(example['normalized_mse']):.6g}"
        pair = _pair_image(example["predicted"], example["groundtruth"], int(args.image_size))
        if save_dir is not None:
            out_path = save_dir / f"{example['index']:06d}_{_safe_filename(example['record_id'])}.png"
            pair.save(out_path)
            print(f"saved {out_path}")
        pairs.append((pair, title))

    if not args.no_window:
        _show_window(pairs)

    if skipped:
        print("skipped:")
        for reason in skipped:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
