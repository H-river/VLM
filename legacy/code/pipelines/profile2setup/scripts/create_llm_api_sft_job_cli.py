"""Create a multimodal LLM API SFT job for profile2setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile2setup.llm_api.sft_jobs import (
    create_sft_job,
    save_job_metadata,
    upload_training_file,
    upload_validation_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a profile2setup LLM API SFT job.")
    parser.add_argument("--provider", default="openai", choices=("openai",))
    parser.add_argument("--base-model", required=True, help="Provider base model to fine-tune")
    parser.add_argument("--train-jsonl", required=True, help="Training SFT JSONL path")
    parser.add_argument("--val-jsonl", default=None, help="Validation SFT JSONL path")
    parser.add_argument("--out", required=True, help="Output job metadata JSON")
    parser.add_argument("--suffix", default=None, help="Optional fine-tuned model suffix")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and print intended requests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_file = upload_training_file(
        Path(args.train_jsonl),
        provider=args.provider,
        dry_run=args.dry_run,
    )
    val_file = None
    if args.val_jsonl:
        val_file = upload_validation_file(
            Path(args.val_jsonl),
            provider=args.provider,
            dry_run=args.dry_run,
        )

    job = create_sft_job(
        provider=args.provider,
        base_model=args.base_model,
        training_file_id=train_file["id"],
        validation_file_id=None if val_file is None else val_file["id"],
        suffix=args.suffix,
        dry_run=args.dry_run,
    )
    metadata = {
        "provider": args.provider,
        "base_model": args.base_model,
        "training_file": train_file,
        "validation_file": val_file,
        "job": job,
        "job_id": job.get("id"),
        "status": job.get("status"),
        "fine_tuned_model": job.get("fine_tuned_model"),
        "dry_run": args.dry_run,
    }
    saved = save_job_metadata(Path(args.out), metadata)
    print(json.dumps(saved, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
