#!/usr/bin/env python3
"""End-to-end text-first SFT data pipeline for inverse-control optics tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optics_sft.data_quality.report import build_quality_report, write_report


DEFAULT_DATA_ROOT = ROOT.parent / "VLM_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, audit, split, and export a text-first inverse-control SFT dataset."
    )
    parser.add_argument("--dataset-version", type=str, default="text_inverse_v1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=Path("optical_sim/configs/base_config.yaml"),
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-ood-split", action="store_true")
    parser.add_argument("--skip-text-export", action="store_true")
    parser.add_argument(
        "--training-target",
        choices=("full", "compact", "control_plan_only"),
        default="compact",
        help="Text export: aligned compact JSON (default) or control_plan_only supervision.",
    )
    return parser.parse_args()


def run_step(label: str, command: list[str]) -> None:
    print(f"[pipeline] {label}")
    print("[pipeline] " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    version_dir = args.data_root / args.dataset_version
    physics_dir = version_dir / "physics_inverse"
    text_dir = version_dir / "text"
    reports_dir = version_dir / "reports"

    version_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        run_step(
            "generate inverse-control physics rows",
            [
                sys.executable,
                "optics_sft/scripts/generate_physics_inverse_dataset.py",
                "--output-dir",
                str(physics_dir),
                "--num-samples",
                str(args.num_samples),
                "--seed",
                str(args.seed),
                "--config",
                str(args.sim_config),
            ],
        )

    physics_quality = build_quality_report(
        train_jsonl=physics_dir / "train.jsonl",
        val_jsonl=physics_dir / "val.jsonl",
        extra_jsonls={"test": physics_dir / "test.jsonl"},
        dataset_name=f"{args.dataset_version}/physics_inverse",
    )
    write_report(reports_dir / "physics_quality_report.json", physics_quality)

    split_source_dir = physics_dir
    if not args.skip_ood_split:
        ood_dir = version_dir / "physics_inverse_ood"
        merged_input = version_dir / "physics_inverse_all.jsonl"
        merged_rows: list[dict[str, Any]] = []
        for split_name in ("train", "val", "test"):
            split_path = physics_dir / f"{split_name}.jsonl"
            if split_path.exists():
                merged_rows.extend(json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip())
        merged_input.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in merged_rows) + ("\n" if merged_rows else ""),
            encoding="utf-8",
        )
        run_step(
            "create ID/OOD splits",
            [
                sys.executable,
                "optics_sft/scripts/make_physics_ood_splits.py",
                "--input-jsonl",
                str(merged_input),
                "--output-dir",
                str(ood_dir),
                "--ood-parameter",
                "lens_focal_length_mm",
                "--train-min",
                "70",
                "--train-max",
                "130",
                "--ood-min",
                "40",
                "--ood-max",
                "60",
                "--seed",
                str(args.seed),
            ],
        )
        split_source_dir = ood_dir

    if not args.skip_text_export:
        text_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val", "test", "test_id", "test_ood"):
            source = split_source_dir / f"{split_name}.jsonl"
            if not source.exists():
                continue
            run_step(
                f"export text split {split_name}",
                [
                    sys.executable,
                    "optics_sft/scripts/build_text_sft_dataset.py",
                    "--input-jsonl",
                    str(source),
                    "--output-jsonl",
                    str(text_dir / f"{split_name}.jsonl"),
                    "--training-target",
                    args.training_target,
                ],
            )

    text_quality = build_quality_report(
        train_jsonl=text_dir / "train.jsonl",
        val_jsonl=text_dir / "val.jsonl",
        extra_jsonls={
            name: text_dir / f"{name}.jsonl"
            for name in ("test", "test_id", "test_ood")
            if (text_dir / f"{name}.jsonl").exists()
        },
        dataset_name=f"{args.dataset_version}/text",
    )
    write_report(reports_dir / "text_quality_report.json", text_quality)

    manifest = {
        "dataset_version": args.dataset_version,
        "data_root": str(version_dir),
        "physics_dir": str(physics_dir),
        "text_dir": str(text_dir),
        "reports_dir": str(reports_dir),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "sim_config": str(args.sim_config),
        "physics_quality_passed": physics_quality["quality_gates"]["passed"],
        "text_quality_passed": text_quality["quality_gates"]["passed"],
        "next_steps": [
            "Review reports/text_quality_report.json",
            "Run base vs SFT benchmark: optics_sft/scripts/eval_text_sft_benchmark.py",
            "Re-export text JSONL after prompt changes: re-run this pipeline or build_text_sft_dataset.py",
            "Train adapter: optics_sft/scripts/train_text_qlora.py --config optics_sft/configs/qwen25_3b_text_qlora_inverse_v3_aligned.yaml",
        ],
    }
    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not text_quality["quality_gates"]["passed"]:
        print("[warn] text quality gates did not fully pass; inspect reports/text_quality_report.json")


if __name__ == "__main__":
    main()
