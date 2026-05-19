"""Orchestrate the physics-understanding diagnostic experiment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "profile2setup/experiments/physics_understanding/config.yaml"
DEFAULT_PROBES = "profile2setup/data/physics_understanding/probes.jsonl"
DEFAULT_VARIABLES_CONFIG = "profile2setup/configs/variables.yaml"
DEFAULT_VOCAB = "profile2setup/data/all_modes/vocab.json"

METRIC_KEYS = (
    ("valid_json_rate", ("format_metrics", "valid_json_rate")),
    ("schema_valid_rate", ("format_metrics", "schema_valid_rate")),
    ("canonical_variable_rate", ("format_metrics", "canonical_variable_rate")),
    (
        "paraphrase_consistency_score",
        ("language_physics_metrics", "paraphrase_consistency_score", "score"),
    ),
    (
        "prompt_sensitivity_score",
        ("language_physics_metrics", "prompt_sensitivity_score", "score"),
    ),
    ("fixed_variable_violation_rate", ("language_physics_metrics", "fixed_variable_violation_rate")),
    ("allowed_variable_violation_rate", ("language_physics_metrics", "allowed_variable_violation_rate")),
    ("contradiction_detection_accuracy", ("language_physics_metrics", "contradiction_detection_accuracy")),
    (
        "prompt_image_conflict_detection_accuracy",
        ("language_physics_metrics", "prompt_image_conflict_detection_accuracy"),
    ),
    ("direction_accuracy", ("language_physics_metrics", "direction_accuracy")),
    ("direction_flip_accuracy", ("language_physics_metrics", "direction_flip_accuracy", "accuracy")),
    ("forced_prediction_rate", ("language_physics_metrics", "forced_prediction_rate")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full physics-understanding diagnostic experiment."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Experiment YAML config")
    parser.add_argument("--base-model", default=None, help="Base LLM model name")
    parser.add_argument("--fine-tuned-model", default=None, help="Fine-tuned LLM model name")
    parser.add_argument("--checkpoint", default=None, help="Optional local PyTorch checkpoint")
    parser.add_argument("--data", default=None, help="Input all_modes test JSONL for probe building")
    parser.add_argument("--probes", default=DEFAULT_PROBES, help="Probe JSONL path")
    parser.add_argument("--out-dir", default=None, help="Physics-understanding results directory")
    parser.add_argument("--variables-config", default=DEFAULT_VARIABLES_CONFIG)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--max-base-records", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional LLM/local probe limit")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--max-text-len", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--run-local", action="store_true", help="Run local PyTorch baseline")
    parser.add_argument("--skip-probe-build", action="store_true")
    parser.add_argument("--force-rebuild-probes", action="store_true")
    parser.add_argument("--reuse-predictions", action="store_true", help="Reuse existing prediction JSONLs")
    parser.add_argument("--reuse-evals", action="store_true", help="Reuse existing eval JSONs")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the experiment config") from exc
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"config must be a dict: {path}")
    return obj


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else None


def _get_nested(obj: dict | None, path: tuple[str, ...]) -> Any:
    value: Any = obj
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _sanitize_label(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe[:80] or "model"


def _cmd_to_text(cmd: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in cmd)


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=@+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _run_command(cmd: list[str], *, dry_run: bool, steps: list[dict[str, Any]], name: str) -> bool:
    command_text = _cmd_to_text(cmd)
    if dry_run:
        steps.append({"name": name, "status": "planned", "command": command_text})
        return True
    try:
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        steps.append(
            {
                "name": name,
                "status": "completed",
                "command": command_text,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        return True
    except subprocess.CalledProcessError as exc:
        steps.append(
            {
                "name": name,
                "status": "failed",
                "command": command_text,
                "returncode": exc.returncode,
                "stdout_tail": (exc.stdout or "")[-4000:],
                "stderr_tail": (exc.stderr or "")[-4000:],
            }
        )
        raise


def _build_probe_command(
    *,
    data: Path,
    probes: Path,
    variables_config: Path,
    max_base_records: int,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "legacy.physics_understanding.scripts.build_physics_understanding_probes_cli",
        "--data",
        str(data),
        "--out",
        str(probes),
        "--variables-config",
        str(variables_config),
        "--max-base-records",
        str(max_base_records),
        "--seed",
        str(seed),
    ]


def _llm_command(
    *,
    model: str,
    probes: Path,
    predictions: Path,
    image_dir: Path,
    limit: int | None,
    dry_run_api: bool,
    temperature: float,
    max_output_tokens: int | None,
    image_detail: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "legacy.physics_understanding.scripts.run_physics_understanding_llm_cli",
        "--model",
        model,
        "--probes",
        str(probes),
        "--out",
        str(predictions),
        "--image-out-dir",
        str(image_dir),
        "--temperature",
        str(temperature),
        "--image-detail",
        image_detail,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if dry_run_api:
        cmd.append("--dry-run")
    if max_output_tokens is not None:
        cmd.extend(["--max-output-tokens", str(max_output_tokens)])
    return cmd


def _local_command(
    *,
    checkpoint: Path,
    probes: Path,
    predictions: Path,
    variables_config: Path,
    vocab: Path,
    input_size: int,
    max_text_len: int,
    limit: int | None,
    device: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "legacy.physics_understanding.scripts.run_physics_understanding_local_cli",
        "--checkpoint",
        str(checkpoint),
        "--probes",
        str(probes),
        "--out",
        str(predictions),
        "--variables-config",
        str(variables_config),
        "--vocab",
        str(vocab),
        "--input-size",
        str(input_size),
        "--max-text-len",
        str(max_text_len),
        "--device",
        device,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return cmd


def _eval_command(*, predictions: Path, probes: Path, out: Path, variables_config: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "legacy.physics_understanding.scripts.evaluate_physics_understanding_cli",
        "--predictions",
        str(predictions),
        "--probes",
        str(probes),
        "--out",
        str(out),
        "--variables-config",
        str(variables_config),
    ]


def _metric_summary(report: dict | None) -> dict[str, Any]:
    return {name: _get_nested(report, path) for name, path in METRIC_KEYS}


def _delta(new: Any, old: Any) -> float | None:
    if isinstance(new, (int, float)) and isinstance(old, (int, float)):
        return float(new) - float(old)
    return None


def _comparison_table(reports: dict[str, dict | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, path in METRIC_KEYS:
        row = {"metric": metric_name}
        for label, report in reports.items():
            row[label] = _get_nested(report, path)
        rows.append(row)
    return rows


def _probe_breakdown(reports: dict[str, dict | None]) -> dict[str, dict[str, Any]]:
    probe_types: set[str] = set()
    for report in reports.values():
        metrics = (((report or {}).get("probe_level_metrics") or {}).get("per_probe_type_metrics") or {})
        probe_types.update(metrics)
    out: dict[str, dict[str, Any]] = {}
    for probe_type in sorted(probe_types):
        out[probe_type] = {}
        for label, report in reports.items():
            metrics = (((report or {}).get("probe_level_metrics") or {}).get("per_probe_type_metrics") or {})
            item = metrics.get(probe_type) or {}
            out[probe_type][label] = {
                "count": item.get("rows"),
                "success_rate": item.get("success_rate"),
                "direction_accuracy": item.get("direction_accuracy"),
                "forced_prediction_rate": item.get("forced_prediction_rate"),
                "fixed_variable_violation_rate": item.get("fixed_variable_violation_rate"),
                "allowed_variable_violation_rate": item.get("allowed_variable_violation_rate"),
            }
    return out


def _interpretation(reports: dict[str, dict | None]) -> dict[str, Any]:
    base = _metric_summary(reports.get("base"))
    fine = _metric_summary(reports.get("fine_tuned"))
    improvements: list[str] = []
    worsens: list[str] = []
    neutral: list[str] = []
    lower_is_better = {
        "fixed_variable_violation_rate",
        "allowed_variable_violation_rate",
        "forced_prediction_rate",
    }
    for metric, fine_value in fine.items():
        base_value = base.get(metric)
        diff = _delta(fine_value, base_value)
        if diff is None:
            continue
        better = diff < 0 if metric in lower_is_better else diff > 0
        worse = diff > 0 if metric in lower_is_better else diff < 0
        text = f"{metric}: base={_fmt(base_value)}, fine_tuned={_fmt(fine_value)}, delta={diff:+.4f}"
        if better:
            improvements.append(text)
        elif worse:
            worsens.append(text)
        else:
            neutral.append(text)

    key_understanding = [
        fine.get("prompt_sensitivity_score"),
        fine.get("contradiction_detection_accuracy"),
        fine.get("prompt_image_conflict_detection_accuracy"),
    ]
    supports_understanding = any(isinstance(value, (int, float)) and value > 0.5 for value in key_understanding)
    schema_imitation_signal = (
        isinstance(fine.get("schema_valid_rate"), (int, float))
        and fine.get("schema_valid_rate") >= 0.95
        and not supports_understanding
    )
    return {
        "where_sft_improves": improvements,
        "where_sft_worsens": worsens,
        "unchanged_metrics": neutral,
        "supports_prompt_conditioned_physics_understanding": supports_understanding,
        "schema_imitation_or_image_regression_signal": schema_imitation_signal,
        "summary": (
            "Evidence supports prompt-conditioned physics understanding only if SFT improves "
            "prompt sensitivity, contradiction/conflict detection, and constraint following, "
            "not just JSON/schema validity. High schema validity with weak prompt/conflict "
            "metrics is more consistent with schema imitation or image-driven regression."
        ),
    }


def _write_comparison_markdown(comparison: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(comparison["reports"].keys())
    lines = [
        "# Physics Understanding Comparison",
        "",
        "## Metric Table",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---:" for _ in labels) + "|",
    ]
    for row in comparison["metric_table"]:
        lines.append(
            "| "
            + row["metric"]
            + " | "
            + " | ".join(_fmt(row.get(label)) for label in labels)
            + " |"
        )
    lines.extend(["", "## Per-Probe-Type Breakdown", ""])
    for probe_type, by_model in comparison["per_probe_type_breakdown"].items():
        lines.extend(
            [
                f"### {probe_type}",
                "",
                "| Model | Count | Success | Direction | Forced Prediction | Fixed Violation | Allowed Violation |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label in labels:
            item = by_model.get(label) or {}
            lines.append(
                f"| {label} | {_fmt(item.get('count'))} | {_fmt(item.get('success_rate'))} | "
                f"{_fmt(item.get('direction_accuracy'))} | {_fmt(item.get('forced_prediction_rate'))} | "
                f"{_fmt(item.get('fixed_variable_violation_rate'))} | {_fmt(item.get('allowed_variable_violation_rate'))} |"
            )
        lines.append("")

    interp = comparison["interpretation"]
    lines.extend(
        [
            "## Interpretation",
            "",
            "### Where SFT Improves",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in interp["where_sft_improves"]] or ["- No scored improvements over base were found."])
    lines.extend(["", "### Where SFT Worsens", ""])
    lines.extend([f"- {item}" for item in interp["where_sft_worsens"]] or ["- No scored regressions versus base were found."])
    lines.extend(
        [
            "",
            "### Understanding Evidence",
            "",
            f"- Supports prompt-conditioned physics understanding: `{interp['supports_prompt_conditioned_physics_understanding']}`",
            f"- Schema imitation or image-regression signal: `{interp['schema_imitation_or_image_regression_signal']}`",
            f"- {interp['summary']}",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _build_comparison(reports: dict[str, dict | None], paths: dict[str, str]) -> dict[str, Any]:
    compact_reports = {label: _metric_summary(report) for label, report in reports.items()}
    return {
        "paths": paths,
        "reports": compact_reports,
        "metric_table": _comparison_table(reports),
        "per_probe_type_breakdown": _probe_breakdown(reports),
        "interpretation": _interpretation(reports),
    }


def main() -> None:
    args = parse_args()
    config = _load_yaml(Path(args.config))
    models_cfg = config.get("models") or {}
    experiment_cfg = config.get("experiment") or {}
    data_cfg = config.get("data") or {}
    all_modes = data_cfg.get("all_modes") or {}

    base_model = args.base_model or models_cfg.get("base_model_name")
    fine_model = args.fine_tuned_model or models_cfg.get("fine_tuned_model_name")
    checkpoint = args.checkpoint or models_cfg.get("local_model_checkpoint")
    if not base_model:
        raise ValueError("--base-model is required when config has no models.base_model_name")
    if not fine_model:
        raise ValueError("--fine-tuned-model is required when config has no models.fine_tuned_model_name")

    out_dir = Path(args.out_dir or experiment_cfg.get("output_dir") or "profile2setup/results/physics_understanding")
    probes = Path(args.probes)
    data = Path(args.data or all_modes.get("test") or "profile2setup/data/all_modes/test.jsonl")
    variables_config = Path(args.variables_config)
    vocab = Path(args.vocab)
    seed = int(args.seed if args.seed is not None else experiment_cfg.get("random_seed", 42))
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "base": "base",
        "fine_tuned": "fine_tuned",
    }
    prediction_paths = {
        "base": out_dir / f"predictions_{labels['base']}.jsonl",
        "fine_tuned": out_dir / f"predictions_{labels['fine_tuned']}.jsonl",
    }
    image_dirs = {
        "base": out_dir / f"rendered_images_{labels['base']}",
        "fine_tuned": out_dir / f"rendered_images_{labels['fine_tuned']}",
    }
    eval_paths = {
        "base": out_dir / f"eval_{labels['base']}.json",
        "fine_tuned": out_dir / f"eval_{labels['fine_tuned']}.json",
    }
    if args.run_local:
        labels["local"] = "local"
        prediction_paths["local"] = out_dir / "predictions_local.jsonl"
        eval_paths["local"] = out_dir / "eval_local.json"

    steps: list[dict[str, Any]] = []
    planned_commands: list[str] = []

    probe_cmd = _build_probe_command(
        data=data,
        probes=probes,
        variables_config=variables_config,
        max_base_records=args.max_base_records,
        seed=seed,
    )
    if args.skip_probe_build:
        steps.append({"name": "build_probes", "status": "skipped"})
    elif args.force_rebuild_probes or not probes.is_file():
        _run_command(probe_cmd, dry_run=args.dry_run, steps=steps, name="build_probes")
        planned_commands.append(_cmd_to_text(probe_cmd))
    else:
        steps.append({"name": "build_probes", "status": "reused", "path": str(probes)})

    llm_specs = [
        ("base", str(base_model)),
        ("fine_tuned", str(fine_model)),
    ]
    for label, model in llm_specs:
        pred_path = prediction_paths[label]
        if args.reuse_predictions and pred_path.is_file():
            steps.append({"name": f"{label}_llm", "status": "reused", "path": str(pred_path)})
        else:
            cmd = _llm_command(
                model=model,
                probes=probes,
                predictions=pred_path,
                image_dir=image_dirs[label],
                limit=args.limit,
                dry_run_api=args.dry_run,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                image_detail=args.image_detail,
            )
            _run_command(cmd, dry_run=args.dry_run, steps=steps, name=f"{label}_llm")
            planned_commands.append(_cmd_to_text(cmd))

    if args.run_local:
        if not checkpoint:
            raise ValueError("--run-local requires --checkpoint or config models.local_model_checkpoint")
        pred_path = prediction_paths["local"]
        if args.reuse_predictions and pred_path.is_file():
            steps.append({"name": "local_baseline", "status": "reused", "path": str(pred_path)})
        else:
            cmd = _local_command(
                checkpoint=Path(checkpoint),
                probes=probes,
                predictions=pred_path,
                variables_config=variables_config,
                vocab=vocab,
                input_size=args.input_size,
                max_text_len=args.max_text_len,
                limit=args.limit,
                device=args.device,
            )
            _run_command(cmd, dry_run=args.dry_run, steps=steps, name="local_baseline")
            planned_commands.append(_cmd_to_text(cmd))

    for label, pred_path in prediction_paths.items():
        eval_path = eval_paths[label]
        if args.reuse_evals and eval_path.is_file():
            steps.append({"name": f"{label}_eval", "status": "reused", "path": str(eval_path)})
        else:
            cmd = _eval_command(
                predictions=pred_path,
                probes=probes,
                out=eval_path,
                variables_config=variables_config,
            )
            _run_command(cmd, dry_run=args.dry_run, steps=steps, name=f"{label}_eval")
            planned_commands.append(_cmd_to_text(cmd))

    comparison_json = out_dir / "comparison.json"
    comparison_md = out_dir / "comparison.md"
    if args.dry_run:
        plan = {
            "dry_run": True,
            "steps": steps,
            "commands": planned_commands,
            "note": "No commands were executed by the orchestrator. LLM commands include --dry-run.",
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    reports = {label: _load_json(path) for label, path in eval_paths.items()}
    paths = {
        "probes": str(probes),
        "comparison_json": str(comparison_json),
        "comparison_md": str(comparison_md),
        **{f"{label}_predictions": str(path) for label, path in prediction_paths.items()},
        **{f"{label}_eval": str(path) for label, path in eval_paths.items()},
    }
    comparison = _build_comparison(reports, paths)
    comparison["steps"] = steps
    _save_json(comparison, comparison_json)
    _write_comparison_markdown(comparison, comparison_md)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
