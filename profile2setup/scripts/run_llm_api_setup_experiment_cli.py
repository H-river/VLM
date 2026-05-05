"""Run the multimodal LLM API setup-understanding experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile2setup.experiments.llm_api_setup_understanding import run_llm_api_setup_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end profile2setup LLM API setup-understanding experiment."
    )
    parser.add_argument("--provider", default="openai", choices=("openai",))
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--sft-model", default="", help="Fine-tuned model ID. Empty means skip post-SFT inference.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile2setup-checkpoint", default=None)
    parser.add_argument("--variables-config", default="profile2setup/configs/variables.yaml")
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--run-simulator", action="store_true")
    parser.add_argument(
        "--simulation-policy",
        choices=("target_base", "current_base", "auto"),
        default="target_base",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build request payloads without API calls")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument("--reuse-rendered-images", action="store_true")
    parser.add_argument("--include-composite", action="store_true")
    parser.add_argument(
        "--image-mode",
        choices=("base64", "relative", "public"),
        default="base64",
        help="Image reference mode for generated SFT JSONL",
    )
    parser.add_argument("--sft-train-limit", type=int, default=None, help="Optional SFT train rows to build")
    parser.add_argument("--sft-val-limit", type=int, default=None, help="Optional SFT val rows to build")

    parser.add_argument("--skip-sft-build", action="store_true")
    parser.add_argument("--reuse-sft-jsonl", action="store_true")
    parser.add_argument("--skip-base-inference", action="store_true")
    parser.add_argument("--base-predictions", default=None, help="Reuse existing base prediction JSONL")
    parser.add_argument("--skip-base-eval", action="store_true")

    parser.add_argument("--create-sft-job", action="store_true")
    parser.add_argument("--sft-job-metadata", default=None, help="Path to job metadata JSON")
    parser.add_argument("--check-sft-job", action="store_true")

    parser.add_argument("--skip-sft-inference", action="store_true")
    parser.add_argument("--sft-predictions", default=None, help="Reuse existing SFT prediction JSONL")
    parser.add_argument("--skip-sft-eval", action="store_true")

    parser.add_argument("--run-profile2setup-baseline", action="store_true")
    parser.add_argument("--profile2setup-baseline-json", default=None, help="Reuse existing local baseline eval JSON")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--strict-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_llm_api_setup_experiment(
        provider=args.provider,
        base_model=args.base_model,
        sft_model=args.sft_model,
        train_jsonl=Path(args.train_jsonl),
        val_jsonl=Path(args.val_jsonl),
        test_jsonl=Path(args.test_jsonl),
        out_dir=Path(args.out_dir),
        profile2setup_checkpoint=Path(args.profile2setup_checkpoint) if args.profile2setup_checkpoint else None,
        max_test_examples=args.max_test_examples,
        run_simulator=args.run_simulator,
        simulation_policy=args.simulation_policy,
        variables_config=Path(args.variables_config),
        dry_run=args.dry_run,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        image_detail=args.image_detail,
        reuse_rendered_images=args.reuse_rendered_images,
        include_composite=args.include_composite,
        image_mode=args.image_mode,
        sft_train_limit=args.sft_train_limit,
        sft_val_limit=args.sft_val_limit,
        build_sft=not args.skip_sft_build,
        reuse_sft_jsonl=args.reuse_sft_jsonl,
        skip_base_inference=args.skip_base_inference,
        base_predictions=Path(args.base_predictions) if args.base_predictions else None,
        skip_base_eval=args.skip_base_eval,
        create_sft=args.create_sft_job,
        sft_job_metadata=Path(args.sft_job_metadata) if args.sft_job_metadata else None,
        check_sft_job=args.check_sft_job,
        skip_sft_inference=args.skip_sft_inference,
        sft_predictions=Path(args.sft_predictions) if args.sft_predictions else None,
        skip_sft_eval=args.skip_sft_eval,
        run_profile2setup_baseline=args.run_profile2setup_baseline,
        profile2setup_baseline_json=Path(args.profile2setup_baseline_json)
        if args.profile2setup_baseline_json
        else None,
        device=args.device,
        strict_baseline=args.strict_baseline,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
