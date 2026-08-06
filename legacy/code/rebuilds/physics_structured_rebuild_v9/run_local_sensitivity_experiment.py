#!/usr/bin/env python3
"""Measure same-action simulator sensitivity to visible setup perturbations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_structured_rebuild_v9.simulator_state_experiment_common import (
    REGIMES,
    action_grid_states,
    bootstrap_mean_ci,
    geometric_focus_residual,
    json_safe,
    lens_transmission,
    sample_visible_setup,
    setup_from_visible,
    stable_seed,
)
from specialist_rebuild_v2.common import SETUP_FIELDS, STATE_FIELDS
from control_rebuild_v3.common import tolerance_from_current

DEFAULT_OUTPUT = (
    REPO_ROOT.parent
    / "VLM_runs/physics_structured_rebuild_v9_state_experiments"
    / "local_sensitivity"
)
FRACTIONS = (0.001, 0.005, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def context_schedule(count: int) -> list[tuple[str, int]]:
    return [
        (REGIMES[index % len(REGIMES)], index // len(REGIMES))
        for index in range(count)
    ]


def variant_metrics(
    base_states: np.ndarray,
    perturbed_states: np.ndarray,
    fraction: float,
) -> dict[str, Any]:
    tolerance = tolerance_from_current(base_states[40]).astype(np.float64)
    output_signed = (perturbed_states - base_states) / tolerance[None, :]
    base_response = base_states - base_states[40][None, :]
    perturbed_response = (
        perturbed_states - perturbed_states[40][None, :]
    )
    response_signed = (
        perturbed_response - base_response
    ) / tolerance[None, :]
    output_absolute = np.abs(output_signed)
    response_absolute = np.abs(response_signed)
    scale_to_one_percent = 0.01 / fraction
    output_crossing = np.any(output_absolute > 1.0, axis=1)
    response_crossing = np.any(response_absolute > 1.0, axis=1)
    return {
        "fraction": fraction,
        "resulting_state_tolerance_crossing_rate": float(
            output_crossing.mean()
        ),
        "action_response_tolerance_crossing_rate": float(
            response_crossing.mean()
        ),
        "resulting_state_any_crossing": bool(output_crossing.any()),
        "action_response_any_crossing": bool(response_crossing.any()),
        "resulting_state_max_tolerance_units": float(output_absolute.max()),
        "action_response_max_tolerance_units": float(response_absolute.max()),
        "resulting_state_gradient_per_one_percent": {
            field: {
                "mean_signed_tolerance_units": float(
                    output_signed[:, field_index].mean()
                    * scale_to_one_percent
                ),
                "mean_absolute_tolerance_units": float(
                    output_absolute[:, field_index].mean()
                    * scale_to_one_percent
                ),
                "p95_absolute_tolerance_units": float(
                    np.quantile(output_absolute[:, field_index], 0.95)
                    * scale_to_one_percent
                ),
            }
            for field_index, field in enumerate(STATE_FIELDS)
        },
        "action_response_gradient_per_one_percent": {
            field: {
                "mean_signed_tolerance_units": float(
                    response_signed[:, field_index].mean()
                    * scale_to_one_percent
                ),
                "mean_absolute_tolerance_units": float(
                    response_absolute[:, field_index].mean()
                    * scale_to_one_percent
                ),
                "p95_absolute_tolerance_units": float(
                    np.quantile(response_absolute[:, field_index], 0.95)
                    * scale_to_one_percent
                ),
            }
            for field_index, field in enumerate(STATE_FIELDS)
        },
    }


def build_context(
    regime: str,
    regime_index: int,
    seed: int,
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    started = time.perf_counter()
    visible = sample_visible_setup(regime, regime_index, seed)
    setup = setup_from_visible(visible)
    base_states = action_grid_states(setup)
    variants: dict[str, dict[str, Any]] = {}
    for field in SETUP_FIELDS:
        variants[field] = {}
        for fraction in FRACTIONS:
            perturbed = dict(visible)
            base_value = float(perturbed[field])
            if base_value == 0.0:
                raise ValueError(
                    f"{regime}:{regime_index}: zero value cannot be "
                    f"multiplicatively perturbed for {field}"
                )
            perturbed[field] = base_value * (1.0 + fraction)
            states = action_grid_states(setup_from_visible(perturbed))
            variants[field][f"{100.0 * fraction:.1f}%"] = variant_metrics(
                base_states,
                states,
                fraction,
            )
    return {
        "version": "local_sensitivity_context_v9",
        "context_id": f"{regime}_{regime_index:04d}",
        "regime": regime,
        "regime_index": regime_index,
        "visible_setup": visible,
        "base_current_state": {
            field: float(base_states[40, field_index])
            for field_index, field in enumerate(STATE_FIELDS)
        },
        "regime_diagnostics": {
            "lens_transmission": lens_transmission(setup),
            "geometric_focus_residual": geometric_focus_residual(setup),
        },
        "variants": variants,
        "seconds": float(time.perf_counter() - started),
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize(
    contexts: list[dict[str, Any]],
    seed: int,
    draws: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in SETUP_FIELDS:
        summary[field] = {}
        for fraction in FRACTIONS:
            fraction_name = f"{100.0 * fraction:.1f}%"
            records = [
                context["variants"][field][fraction_name]
                for context in contexts
            ]
            item: dict[str, Any] = {
                "resulting_state_tolerance_crossing_rate": bootstrap_mean_ci(
                    np.asarray(
                        [
                            record[
                                "resulting_state_tolerance_crossing_rate"
                            ]
                            for record in records
                        ]
                    ),
                    stable_seed(seed, field, fraction, "output_cross"),
                    draws,
                ),
                "action_response_tolerance_crossing_rate": bootstrap_mean_ci(
                    np.asarray(
                        [
                            record[
                                "action_response_tolerance_crossing_rate"
                            ]
                            for record in records
                        ]
                    ),
                    stable_seed(seed, field, fraction, "response_cross"),
                    draws,
                ),
                "resulting_state_any_crossing_context_rate": bootstrap_mean_ci(
                    np.asarray(
                        [
                            record["resulting_state_any_crossing"]
                            for record in records
                        ],
                        dtype=np.float64,
                    ),
                    stable_seed(seed, field, fraction, "output_any"),
                    draws,
                ),
                "action_response_any_crossing_context_rate": bootstrap_mean_ci(
                    np.asarray(
                        [
                            record["action_response_any_crossing"]
                            for record in records
                        ],
                        dtype=np.float64,
                    ),
                    stable_seed(seed, field, fraction, "response_any"),
                    draws,
                ),
                "outputs": {},
                "by_regime": {},
            }
            for output in STATE_FIELDS:
                item["outputs"][output] = {
                    "resulting_state_mean_absolute_gradient_per_one_percent": (
                        bootstrap_mean_ci(
                            np.asarray(
                                [
                                    record[
                                        "resulting_state_gradient_per_one_percent"
                                    ][output][
                                        "mean_absolute_tolerance_units"
                                    ]
                                    for record in records
                                ]
                            ),
                            stable_seed(
                                seed,
                                field,
                                fraction,
                                output,
                                "output_gradient",
                            ),
                            draws,
                        )
                    ),
                    "action_response_mean_absolute_gradient_per_one_percent": (
                        bootstrap_mean_ci(
                            np.asarray(
                                [
                                    record[
                                        "action_response_gradient_per_one_percent"
                                    ][output][
                                        "mean_absolute_tolerance_units"
                                    ]
                                    for record in records
                                ]
                            ),
                            stable_seed(
                                seed,
                                field,
                                fraction,
                                output,
                                "response_gradient",
                            ),
                            draws,
                        )
                    ),
                }
            for regime in REGIMES:
                regime_records = [
                    record
                    for context, record in zip(contexts, records, strict=True)
                    if context["regime"] == regime
                ]
                item["by_regime"][regime] = {
                    "count": len(regime_records),
                    "resulting_state_crossing_rate": bootstrap_mean_ci(
                        np.asarray(
                            [
                                record[
                                    "resulting_state_tolerance_crossing_rate"
                                ]
                                for record in regime_records
                            ]
                        ),
                        stable_seed(
                            seed, field, fraction, regime, "output_cross"
                        ),
                        draws,
                    ),
                    "action_response_crossing_rate": bootstrap_mean_ci(
                        np.asarray(
                            [
                                record[
                                    "action_response_tolerance_crossing_rate"
                                ]
                                for record in regime_records
                            ]
                        ),
                        stable_seed(
                            seed, field, fraction, regime, "response_cross"
                        ),
                        draws,
                    ),
                }
            summary[field][fraction_name] = item
    return summary


def main() -> None:
    args = parse_args()
    if int(args.context_count) < 1:
        raise ValueError("context-count must be positive")
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "summary.json"
    if manifest_path.exists():
        print(manifest_path.read_text(encoding="utf-8"), end="")
        return
    started = time.perf_counter()
    schedule = context_schedule(int(args.context_count))
    contexts_by_position: dict[int, dict[str, Any]] = {}
    pending = []
    resumed = 0
    for position, (regime, regime_index) in enumerate(schedule):
        shard = shard_dir / f"{position:04d}_{regime}_{regime_index:04d}.json"
        if shard.exists():
            context = json.loads(shard.read_text(encoding="utf-8"))
            resumed += 1
            contexts_by_position[position] = context
        else:
            pending.append((position, regime, regime_index, shard))
    completed = resumed
    if int(args.workers) == 1:
        for position, regime, regime_index, shard in pending:
            context = build_context(regime, regime_index, int(args.seed))
            write_atomic(shard, context)
            contexts_by_position[position] = context
            completed += 1
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(schedule),
                        "regime": regime,
                        "resumed": resumed,
                        "workers": int(args.workers),
                        "context_seconds": context["seconds"],
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_jobs = {
                executor.submit(
                    build_context,
                    regime,
                    regime_index,
                    int(args.seed),
                ): (position, regime, shard)
                for position, regime, regime_index, shard in pending
            }
            for future in as_completed(future_jobs):
                position, regime, shard = future_jobs[future]
                context = future.result()
                write_atomic(shard, context)
                contexts_by_position[position] = context
                completed += 1
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(schedule),
                            "regime": regime,
                            "resumed": resumed,
                            "workers": int(args.workers),
                            "context_seconds": context["seconds"],
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    contexts = [
        contexts_by_position[position] for position in range(len(schedule))
    ]
    report = {
        "version": "local_sensitivity_experiment_v9",
        "context_count": len(contexts),
        "independent_unit": "one visible setup context",
        "regime_counts": {
            regime: sum(context["regime"] == regime for context in contexts)
            for regime in REGIMES
        },
        "perturbation": {
            "direction": "positive multiplicative, one feature at a time",
            "fractions": list(FRACTIONS),
            "action_count": 81,
            "visible_features": list(SETUP_FIELDS),
        },
        "tolerance": {
            "centroid_each_px": 1.0,
            "sigma_each_px": 2.0,
            "peak_relative": 0.05,
        },
        "regime_diagnostics": {
            regime: {
                "lens_transmission": bootstrap_mean_ci(
                    np.asarray(
                        [
                            context["regime_diagnostics"][
                                "lens_transmission"
                            ]
                            for context in contexts
                            if context["regime"] == regime
                        ]
                    ),
                    stable_seed(args.seed, regime, "transmission"),
                    int(args.bootstrap_draws),
                ),
                "geometric_focus_residual": bootstrap_mean_ci(
                    np.asarray(
                        [
                            context["regime_diagnostics"][
                                "geometric_focus_residual"
                            ]
                            for context in contexts
                            if context["regime"] == regime
                        ]
                    ),
                    stable_seed(args.seed, regime, "focus"),
                    int(args.bootstrap_draws),
                ),
            }
            for regime in REGIMES
        },
        "metrics": summarize(
            contexts,
            int(args.seed),
            int(args.bootstrap_draws),
        ),
        "source_contract": {
            "generated_contexts": len(contexts) - resumed,
            "resumed_contexts": resumed,
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
            "simulator_grid_size": 1024,
            "simulator_backend": "fresnel_numpy",
            "workers": int(args.workers),
        },
        "seconds": float(time.perf_counter() - started),
    }
    write_atomic(manifest_path, report)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
