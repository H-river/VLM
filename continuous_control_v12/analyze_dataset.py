#!/usr/bin/env python3
"""Audit v12 transition coverage, duplicates, leakage, and distributions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from continuous_control_v12.contracts import (
    ACTION_FIELDS,
    OUTPUT_FIELDS,
    POSITION_FIELDS,
    SETUP_CONTEXT_FIELDS,
    Bounds,
    action_vector,
    assert_no_q_star,
    canonical_json,
    metrics_vector,
    stable_hash,
    tolerance_vector,
)
from continuous_control_v12.schema import (
    deployed_transition_input,
    read_jsonl,
    validate_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "std": float(array.std()),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite analysis: {output_dir}")
    output_dir.mkdir(parents=True)
    validation = validate_dataset(data_dir, config)
    manifest = json.loads(
        (data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    bounds = Bounds.from_config(config)
    rows_by_split = {
        split: read_jsonl(data_dir / "transitions" / f"{split}.jsonl")
        for split in ("train", "development", "test")
    }
    all_rows = [
        row for split in ("train", "development", "test") for row in rows_by_split[split]
    ]
    actions = np.stack([action_vector(row["action_mm"]) for row in all_rows])
    requested_actions = np.stack(
        [
            action_vector(row.get("requested_action_mm", row["action_mm"]))
            for row in all_rows
        ]
    )
    positions = np.stack(
        [
            [float(row["positions_mm"][field]) for field in POSITION_FIELDS]
            for row in all_rows
        ]
        + [
            [
                float(row["next_positions_mm"][field])
                for field in POSITION_FIELDS
            ]
            for row in all_rows
        ]
    )
    current = np.stack([metrics_vector(row["metrics"]) for row in all_rows])
    next_metrics = np.stack(
        [metrics_vector(row["next_metrics"]) for row in all_rows]
    )
    normalized_delta = np.stack(
        [
            (metrics_vector(row["next_metrics"]) - metrics_vector(row["metrics"]))
            / tolerance_vector(row["metrics"])
            for row in all_rows
        ]
    )
    setups = np.stack(
        [
            [
                float(row["setup_context"][field])
                for field in SETUP_CONTEXT_FIELDS
            ]
            for row in all_rows
        ]
    )
    finite = bool(
        np.isfinite(actions).all()
        and np.isfinite(requested_actions).all()
        and np.isfinite(positions).all()
        and np.isfinite(current).all()
        and np.isfinite(next_metrics).all()
        and np.isfinite(normalized_delta).all()
        and np.isfinite(setups).all()
    )
    action_nonzero = np.abs(actions) > 1e-12
    active_axes = action_nonzero.sum(axis=1)
    normalized_action = actions / bounds.action_high[None, :]
    sampling_counts = Counter(
        str(row["sampling"]["kind"]) for row in all_rows
    )
    regime_counts = Counter(
        manifest["group_regimes"][str(row["group_id"])]
        for row in all_rows
    )
    transition_ids = [str(row["transition_id"]) for row in all_rows]
    content_hashes = [
        stable_hash(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "transition_id",
                    "simulator_seed",
                    "setup_hash",
                    "context_hash",
                }
            }
        )
        for row in all_rows
    ]
    deployable_hashes = [
        stable_hash(deployed_transition_input(row)) for row in all_rows
    ]
    targets_by_deployable_hash: defaultdict[str, set[str]] = defaultdict(set)
    for deployable_hash, row in zip(
        deployable_hashes, all_rows, strict=True
    ):
        targets_by_deployable_hash[deployable_hash].add(
            stable_hash(
                {
                    "next_positions_mm": row["next_positions_mm"],
                    "next_metrics": row["next_metrics"],
                }
            )
        )
    deployable_counts = Counter(deployable_hashes)
    for row in all_rows:
        assert_no_q_star(deployed_transition_input(row))
    oracle_rows = sum("oracle_metadata" in row for row in all_rows)
    unexpected_q_star = sum(
        "q_star" in canonical_json(deployed_transition_input(row)).lower()
        for row in all_rows
    )

    action_stats = {
        field: {
            **quantiles(actions[:, index]),
            "requested_min": float(requested_actions[:, index].min()),
            "requested_max": float(requested_actions[:, index].max()),
            "zero_fraction": float((~action_nonzero[:, index]).mean()),
            "at_low_bound_fraction": float(
                np.isclose(actions[:, index], bounds.action_low[index]).mean()
            ),
            "at_high_bound_fraction": float(
                np.isclose(actions[:, index], bounds.action_high[index]).mean()
            ),
        }
        for index, field in enumerate(ACTION_FIELDS)
    }
    position_stats = {
        field: quantiles(positions[:, index])
        for index, field in enumerate(POSITION_FIELDS)
    }
    setup_stats = {
        field: {
            **quantiles(setups[:, index]),
            "unique_values": int(np.unique(setups[:, index]).size),
            "constant": bool(np.ptp(setups[:, index]) == 0.0),
        }
        for index, field in enumerate(SETUP_CONTEXT_FIELDS)
    }
    metric_stats = {}
    for index, field in enumerate(OUTPUT_FIELDS):
        metric_stats[field] = {
            "current": quantiles(current[:, index]),
            "next": quantiles(next_metrics[:, index]),
            "normalized_delta": quantiles(normalized_delta[:, index]),
            "constant_current": bool(np.ptp(current[:, index]) == 0.0),
            "constant_next": bool(np.ptp(next_metrics[:, index]) == 0.0),
        }

    split_stats = {}
    setup_hash_sets = {}
    for split, rows in rows_by_split.items():
        groups = {str(row["group_id"]) for row in rows}
        setup_hashes = {str(row["setup_hash"]) for row in rows}
        context_hashes = {str(row["context_hash"]) for row in rows}
        setup_hash_sets[split] = setup_hashes
        split_stats[split] = {
            "groups": len(groups),
            "transitions": len(rows),
            "unique_setup_hashes": len(setup_hashes),
            "unique_context_hashes": len(context_hashes),
        }
    setup_overlap = {}
    split_names = list(rows_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            setup_overlap[f"{left}__{right}"] = len(
                setup_hash_sets[left] & setup_hash_sets[right]
            )

    projected_count = int(
        np.any(np.abs(requested_actions - actions) > 1e-12, axis=1).sum()
    )
    setup_attempt_by_group: dict[str, int] = {}
    initial_fraction_by_group: dict[str, float] = {}
    for row in all_rows:
        group_id = str(row["group_id"])
        setup_attempt_by_group[group_id] = int(
            row["sampling"].get("setup_resample_attempt", 0)
        )
        if "initial_captured_power_fraction" in row["sampling"]:
            initial_fraction_by_group[group_id] = float(
                row["sampling"]["initial_captured_power_fraction"]
            )
    next_captured_fractions = np.asarray(
        [
            float(row["auxiliary"]["captured_power_w"])
            / max(float(row["setup_context"]["power_w"]), 1e-30)
            for row in all_rows
            if row["auxiliary"].get("captured_power_w") is not None
        ],
        dtype=np.float64,
    )
    report: dict[str, Any] = {
        "version": "continuous_control_v12_dataset_analysis_v1",
        "dataset": str(data_dir),
        "schema_version": manifest["schema_version"],
        "simulator_semantics_version": manifest.get(
            "simulator_semantics_version"
        ),
        "validation": validation,
        "split_statistics": split_stats,
        "cross_split_setup_hash_overlap": setup_overlap,
        "rows": len(all_rows),
        "finite": finite,
        "duplicate_transition_ids": len(transition_ids)
        - len(set(transition_ids)),
        "duplicate_full_content_rows_ignoring_ids": len(content_hashes)
        - len(set(content_hashes)),
        "duplicate_deployable_inputs": len(deployable_hashes)
        - len(set(deployable_hashes)),
        "duplicate_deployable_input_keys": int(
            sum(count > 1 for count in deployable_counts.values())
        ),
        "duplicate_deployable_inputs_with_conflicting_targets": int(
            sum(
                len(targets) > 1
                for targets in targets_by_deployable_hash.values()
            )
        ),
        "action_statistics": action_stats,
        "action_regime_counts": {
            "no_op": int((active_axes == 0).sum()),
            "single_axis": int((active_axes == 1).sum()),
            "multi_axis": int((active_axes >= 2).sum()),
            "projected_requested_actions": projected_count,
        },
        "normalized_action_linf": quantiles(
            np.abs(normalized_action).max(axis=1)
        ),
        "position_statistics": position_stats,
        "setup_statistics": setup_stats,
        "dead_or_constant_setup_fields": [
            field for field, stats in setup_stats.items() if stats["constant"]
        ],
        "metric_statistics": metric_stats,
        "sampling_counts": dict(sorted(sampling_counts.items())),
        "regime_group_counts": dict(
            sorted(manifest["coverage"]["regime_group_counts"].items())
        ),
        "regime_transition_counts": dict(sorted(regime_counts.items())),
        "power_contract": {
            "manifest_power_affects_simulator": manifest.get(
                "power_affects_current_simulator"
            ),
            "unique_power_values": setup_stats["power_w"]["unique_values"],
            "power_is_constant_or_dead_in_rows": setup_stats["power_w"][
                "constant"
            ],
            "causal_ratios_verified_by": (
                "separate simulator power_scaling.csv"
            ),
        },
        "initial_signal_quality": {
            "configured_minimum_captured_power_fraction": (
                manifest["generator"]["simulator_fixed"].get(
                    "minimum_initial_captured_power_fraction", 0.0
                )
            ),
            "minimum_observed_initial_captured_power_fraction": (
                None
                if not initial_fraction_by_group
                else min(initial_fraction_by_group.values())
            ),
            "groups_resampled": int(
                sum(attempt > 0 for attempt in setup_attempt_by_group.values())
            ),
            "maximum_setup_resample_attempt_used": (
                0
                if not setup_attempt_by_group
                else max(setup_attempt_by_group.values())
            ),
            "floor_gate_passed": bool(
                not initial_fraction_by_group
                or min(initial_fraction_by_group.values())
                >= float(
                    manifest["generator"]["simulator_fixed"].get(
                        "minimum_initial_captured_power_fraction", 0.0
                    )
                )
            ),
        },
        "transition_signal_quality": {
            "observations": int(len(next_captured_fractions)),
            "next_captured_to_source_power_fraction": (
                None
                if not len(next_captured_fractions)
                else quantiles(next_captured_fractions)
            ),
            "below_1e_6": int((next_captured_fractions < 1e-6).sum()),
            "below_1e_3": int((next_captured_fractions < 1e-3).sum()),
            "below_configured_initial_floor": int(
                (
                    next_captured_fractions
                    < float(
                        manifest["generator"]["simulator_fixed"].get(
                            "minimum_initial_captured_power_fraction", 0.0
                        )
                    )
                ).sum()
            ),
        },
        "privileged_data": {
            "oracle_metadata_rows": oracle_rows,
            "q_star_in_deployable_inputs": unexpected_q_star,
            "deployable_input_guard_passed": unexpected_q_star == 0,
        },
        "bounds": {
            "actions_within_bounds": bool(
                np.all(actions >= bounds.action_low - 1e-12)
                and np.all(actions <= bounds.action_high + 1e-12)
            ),
            "positions_within_bounds": bool(
                np.all(positions >= bounds.position_low - 1e-12)
                and np.all(positions <= bounds.position_high + 1e-12)
            ),
        },
    }
    (output_dir / "dataset_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for index, field in enumerate(ACTION_FIELDS):
        axes.flat[index].hist(actions[:, index], bins=41)
        axes.flat[index].set_title(field)
        axes.flat[index].set_xlabel("mm")
    figure.tight_layout()
    figure.savefig(output_dir / "action_distributions.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 11))
    for index, field in enumerate(OUTPUT_FIELDS):
        axes.flat[index].hist(normalized_delta[:, index], bins=61)
        axes.flat[index].set_title(f"{field} normalized delta")
    axes.flat[-1].hist(np.abs(normalized_delta).max(axis=1), bins=61)
    axes.flat[-1].set_title("max absolute normalized delta")
    figure.tight_layout()
    figure.savefig(output_dir / "target_delta_distributions.png", dpi=150)
    plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
