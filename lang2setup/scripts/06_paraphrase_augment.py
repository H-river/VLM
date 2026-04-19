#!/usr/bin/env python3
"""
06_paraphrase_augment.py
Generate paraphrases of template-based descriptions using an LLM.

Reads the training JSONL, sends batches of descriptions to the LLM
for paraphrasing, and writes an augmented JSONL with both originals
and paraphrases.

Usage:
    python -m lang2setup.scripts.06_paraphrase_augment \
        --input lang2setup/data/lang2setup_train.jsonl \
        --output lang2setup/data/lang2setup_train_augmented.jsonl \
        --paraphrases-per-sample 2 \
        --max-samples 500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lang2setup.llm_interface.api_caller import call_llm, load_llm_config

PARAPHRASE_SYSTEM = """\
You are a text rewriter. Given a technical description of a laser beam setup, \
rewrite it in a different natural style while preserving ALL the information \
about position, tilt, width, shape, and intensity.

Rules:
- Keep the same meaning (same position, same tilt, same width, etc.)
- Use different words, sentence structure, or phrasing
- Stay concise (1-2 sentences max)
- Do NOT add new information or change any parameter
- Output ONLY the rewritten text, nothing else.
"""

PARAPHRASE_BATCH_SYSTEM = """\
You are a text rewriter. Given multiple technical descriptions of laser beam \
setups (one per line), rewrite each one in a different natural style while \
preserving ALL the information about position, tilt, width, shape, and intensity.

Rules:
- Keep the same meaning for each line
- Use different words, sentence structure, or phrasing
- Stay concise (1-2 sentences per line)
- Output exactly one rewritten line per input line
- Do NOT add numbering, bullets, or extra text
"""


def paraphrase_batch(texts: list[str], config: dict) -> list[str]:
    """Send a batch of texts and get paraphrases back."""
    numbered_input = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    messages = [
        {"role": "system", "content": PARAPHRASE_BATCH_SYSTEM},
        {"role": "user", "content": numbered_input},
    ]
    raw = call_llm(messages, config)

    # Parse output lines
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    # Remove numbering if present
    cleaned = []
    for line in lines:
        # Strip "1. " or "1) " prefixes
        for prefix_len in range(1, 5):
            if len(line) > prefix_len and line[prefix_len] in ".)" and line[:prefix_len].isdigit():
                line = line[prefix_len + 1:].strip()
                break
        if line.startswith("- "):
            line = line[2:]
        cleaned.append(line)

    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="lang2setup/data/lang2setup_train.jsonl")
    parser.add_argument("--output", default="lang2setup/data/lang2setup_train_augmented.jsonl")
    parser.add_argument("--paraphrases-per-sample", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max original samples to paraphrase (to control API cost)")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    config = load_llm_config()

    # Load original training data
    with open(args.input) as f:
        records = [json.loads(line) for line in f]

    # Deduplicate by source_run to get unique samples, then pick texts
    seen = set()
    unique_records = []
    for rec in records:
        if rec["source_run"] not in seen:
            seen.add(rec["source_run"])
            unique_records.append(rec)

    to_paraphrase = unique_records[:args.max_samples]
    print(f"▶ Paraphrasing {len(to_paraphrase)} samples × {args.paraphrases_per_sample} variants")
    print(f"  Batch size: {args.batch_size}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write all originals first
    all_records = list(records)
    new_count = 0
    errors = 0

    for p_round in range(args.paraphrases_per_sample):
        print(f"\n  Round {p_round + 1}/{args.paraphrases_per_sample}")
        texts = [r["text"] for r in to_paraphrase]

        for batch_start in range(0, len(texts), args.batch_size):
            batch_texts = texts[batch_start:batch_start + args.batch_size]
            batch_records = to_paraphrase[batch_start:batch_start + args.batch_size]

            try:
                paraphrases = paraphrase_batch(batch_texts, config)

                # Pair paraphrases with records
                for j, (para, orig_rec) in enumerate(zip(paraphrases, batch_records)):
                    if para and len(para) > 5:
                        new_rec = {
                            "text": para,
                            "target": orig_rec["target"],
                            "target_continuous": orig_rec["target_continuous"],
                            "source_run": orig_rec["source_run"],
                            "split": orig_rec.get("split", "train"),
                            "augmented": True,
                        }
                        all_records.append(new_rec)
                        new_count += 1

            except Exception as e:
                errors += 1
                print(f"    ⚠ Batch error at {batch_start}: {e}")

            if (batch_start + args.batch_size) % 50 == 0:
                print(f"    [{batch_start + args.batch_size}/{len(texts)}] processed")

            time.sleep(0.2)

    # Write augmented dataset
    with open(output_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✓ Wrote {len(all_records)} records ({len(records)} original + {new_count} paraphrases)")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
