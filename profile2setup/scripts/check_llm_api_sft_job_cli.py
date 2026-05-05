"""Check and update profile2setup LLM API SFT job metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile2setup.llm_api.sft_jobs import update_job_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a profile2setup LLM API SFT job.")
    parser.add_argument("--job-metadata", required=True, help="Path to saved job metadata JSON")
    parser.add_argument("--provider", default=None, choices=("openai",))
    parser.add_argument("--dry-run", action="store_true", help="Print intended retrieve request without API call")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updated = update_job_metadata(
        Path(args.job_metadata),
        provider=args.provider,
        dry_run=args.dry_run,
    )
    print(json.dumps(updated, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
