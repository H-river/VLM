#!/usr/bin/env python3
"""Verify that an adapter contains only the frozen LoRA tensor scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--expected-language", type=int, default=504)
    parser.add_argument("--expected-visual", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = (
        args.adapter / "adapter_model.safetensors"
        if args.adapter.is_dir()
        else args.adapter
    )
    with safe_open(path, framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
        shapes = {key: list(stream.get_slice(key).get_shape()) for key in keys}
    non_lora = [key for key in keys if ".lora_A." not in key and ".lora_B." not in key]
    visual = [key for key in keys if "visual" in key.lower()]
    visual_set = set(visual)
    language = [key for key in keys if key not in visual_set]
    report = {
        "adapter": str(path.resolve()),
        "tensor_count": len(keys),
        "language_lora_tensors": len(language),
        "visual_lora_tensors": len(visual),
        "non_lora_tensors": non_lora,
        "parameter_count": sum(
            __import__("math").prod(shape) for shape in shapes.values()
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if non_lora:
        raise SystemExit("adapter contains non-LoRA tensors")
    if len(language) != args.expected_language:
        raise SystemExit(
            f"expected {args.expected_language} language tensors, got {len(language)}"
        )
    if len(visual) != args.expected_visual:
        raise SystemExit(
            f"expected {args.expected_visual} visual tensors, got {len(visual)}"
        )


if __name__ == "__main__":
    main()
