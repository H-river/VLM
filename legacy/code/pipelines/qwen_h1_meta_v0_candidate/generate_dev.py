#!/usr/bin/env python3
"""Generate raw strict-meta continuations for the complete candidate dev split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .inference import IndependentMetaAdapter, load_independent_adapter
from .training import (
    REPOSITORY_ROOT,
    TRAINING_SEEDS,
    _guard_candidate_path,
    _repo_path,
    base,
    sha256_path,
    validate_candidate_config,
)


EXPECTED_FULL_DEV_RECORDS = 24


def generate_dev_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    adapter: IndependentMetaAdapter,
    image_root: Path,
    seed: int,
    run_metadata: Mapping[str, Any],
    expected_count: int = EXPECTED_FULL_DEV_RECORDS,
) -> list[dict[str, Any]]:
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"seed must be one of {list(TRAINING_SEEDS)}")
    if len(rows) != expected_count:
        raise ValueError(
            f"full candidate dev reducer requires {expected_count} rows, got {len(rows)}"
        )
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("example_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"invalid or duplicate candidate dev example_id: {sample_id!r}")
        seen.add(sample_id)
        generated = adapter.generate_row(
            row,
            image_root=image_root,
            seed=seed,
            max_new_tokens=256,
        )
        output.append(
            {
                "sample_id": sample_id,
                "seed": seed,
                "prediction": generated.raw_text,
                "valid_json": generated.valid_json,
                "parse_error_code": generated.error_code,
                "parse_error_message": generated.error_message,
                "latency_seconds": generated.latency_seconds,
                "generation_metadata": dict(run_metadata),
            }
        )
    return output


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--local-rank", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = _guard_candidate_path(args.config, role="generation_config")
    data_path = _guard_candidate_path(args.data, role="generation_dev_data")
    adapter_path = _guard_candidate_path(args.adapter, role="generation_adapter")
    output_path = _guard_candidate_path(
        args.output, role="generation_output", must_exist=False
    )
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing candidate predictions: {output_path}"
        )
    config = base.load_yaml(config_path)
    validate_candidate_config(config)
    configured_dev = _repo_path(config["data"]["dev_jsonl"])
    if data_path != configured_dev:
        raise ValueError("--data must equal the hash-pinned config data.dev_jsonl")
    if sha256_path(data_path) != config["data"]["dev_sha256"]:
        raise ValueError("candidate dev JSONL differs from its config SHA-256")
    rows = base.read_jsonl(data_path)
    image_root = _repo_path(config["data"].get("image_root", "."))
    adapter = load_independent_adapter(
        config=config,
        adapter_path=adapter_path,
        local_rank=args.local_rank,
    )
    backend = adapter.backend
    run_metadata = {
        "adapter_name": adapter.adapter_name,
        "adapter_hashes": dict(getattr(backend, "adapter_hashes", {})),
        "config_sha256": sha256_path(config_path),
        "dev_jsonl_sha256": sha256_path(data_path),
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 256,
        "full_dev_reducer_records": EXPECTED_FULL_DEV_RECORDS,
        "formal_frozen_evaluation_enabled": False,
    }
    predictions = generate_dev_rows(
        rows=rows,
        adapter=adapter,
        image_root=image_root,
        seed=args.seed,
        run_metadata=run_metadata,
    )
    _atomic_jsonl(output_path, predictions)
    print(
        json.dumps(
            {
                "status": "candidate_dev_generation_complete",
                "output": str(output_path),
                "records": len(predictions),
                "seed": args.seed,
                "valid_json": sum(row["valid_json"] for row in predictions),
                "frozen_or_protected_predictions_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
