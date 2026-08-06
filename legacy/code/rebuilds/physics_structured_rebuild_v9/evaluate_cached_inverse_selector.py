#!/usr/bin/env python3
"""Evaluate a frozen inverse enumerator selector on protected cached features."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument(
        "--blocks", nargs="+", default=("iid_clean", "difficult_clean")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric(success: np.ndarray, feasible: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(len(success)),
        "feasible_count": int(feasible.sum()),
        "success_all_count": int(success.sum()),
        "success_all": float(success.mean()),
        "success_feasible_count": int(success[feasible].sum()),
        "success_feasible": float(success[feasible].mean()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    with args.selector.resolve().open("rb") as stream:
        selector = pickle.load(stream)
    if selector.get("model") != "grouped_forward_inverse_hgb_selector_v9":
        raise ValueError("unexpected inverse selector artifact")
    feature_names = [str(name) for name in selector["feature_names"]]
    threshold = float(selector["threshold"])
    blocks = {}
    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        for block in args.blocks:
            features = np.column_stack(
                [
                    np.asarray(
                        cache[f"{block}_feature_{name}"], dtype=np.float32
                    )
                    for name in feature_names
                ]
            )
            primary = np.asarray(
                cache[f"{block}_primary_success"], dtype=np.bool_
            )
            secondary = np.asarray(
                cache[f"{block}_secondary_success"], dtype=np.bool_
            )
            feasible = np.asarray(
                cache[f"{block}_feasible"], dtype=np.bool_
            )
            probability = selector["classifier"].predict_proba(features)[:, 1]
            choose = probability >= threshold
            selected = np.where(choose, secondary, primary)
            blocks[str(block)] = {
                "primary": metric(primary, feasible),
                "secondary": metric(secondary, feasible),
                "selected": metric(selected, feasible),
                "oracle_union": metric(primary | secondary, feasible),
                "secondary_count": int(choose.sum()),
                "secondary_only_captured": int(
                    (choose & secondary & ~primary).sum()
                ),
                "primary_only_sacrificed": int(
                    (choose & primary & ~secondary).sum()
                ),
            }
    passed = all(
        values["selected"]["success_all_count"]
        >= values["primary"]["success_all_count"]
        for values in blocks.values()
    ) and any(
        values["selected"]["success_all_count"]
        > values["primary"]["success_all_count"]
        for values in blocks.values()
    )
    report = {
        "version": "cached_inverse_selector_protected_v9_one_seed",
        "blocks": blocks,
        "promotion_passed": bool(passed),
        "source_contract": {
            "selector_frozen_before_protected_cache_opened": True,
            "system_validation_used": False,
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
