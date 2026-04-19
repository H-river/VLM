#!/usr/bin/env python3
"""
05_closed_loop_eval.py
Stage D: Closed-loop evaluation.

For each test sample:
1. Load the ground-truth metadata (full setup config)
2. Convert predicted bins → physical x_offset, tilt_x values
3. Create a modified setup with predicted alignment params
4. Run the simulator
5. Compare predicted beam profile to ground-truth beam profile

Usage:
    python -m lang2setup.scripts.05_closed_loop_eval \
        --predictions lang2setup/data/llm_predictions.jsonl \
        --sim-dir outputs/random_5k \
        --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add VLM root so we can import both optical_sim and lang2setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lang2setup.data_prep.discretize import load_bins_config, bin_to_value
from lang2setup.evaluation.closed_loop import bins_to_physical, compute_beam_similarity

from optical_sim.src.optical_elements import setup_from_dict
from optical_sim.src.simulator import run_simulation
from optical_sim.src.metrics import compute_metrics


def load_ground_truth_setup(sim_dir: Path, run_name: str) -> dict:
    """Load the full metadata.json for a sample."""
    meta_path = sim_dir / run_name / "metadata.json"
    with open(meta_path) as f:
        return json.load(f)


def load_ground_truth_intensity(sim_dir: Path, run_name: str) -> np.ndarray:
    """Load the saved intensity array."""
    return np.load(sim_dir / run_name / "intensity.npy")


def build_predicted_setup(original_meta: dict, predicted_physical: dict) -> dict:
    """
    Take the original setup config and replace alignment params
    with predicted values. Keep everything else the same.
    """
    cfg = {
        "source": original_meta["setup"]["source"],
        "lens": original_meta["setup"]["lens"],
        "sensor": {
            "resolution": original_meta["setup"]["sensor"]["resolution"],
            "pixel_pitch": original_meta["setup"]["sensor"]["pixel_pitch"],
        },
        "geometry": {
            "laser_to_lens": original_meta["setup"]["geometry"]["laser_to_lens"],
            "lens_to_camera": original_meta["setup"]["geometry"]["lens_to_camera"],
        },
        "alignment": dict(original_meta["setup"]["alignment"]),
        "simulation": original_meta["setup"]["simulation"],
    }
    # Override with predicted values
    cfg["alignment"]["x_offset"] = predicted_physical["x_offset"]
    cfg["alignment"]["y_offset"] = predicted_physical["y_offset"]
    cfg["alignment"]["tilt_x"] = predicted_physical["tilt_x"]
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="lang2setup/data/llm_predictions.jsonl")
    parser.add_argument("--sim-dir", default="optical_sim/outputs/random_5k")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples to re-simulate (each takes ~1s)")
    parser.add_argument("--output", default="lang2setup/data/closed_loop_results.jsonl")
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    bins_cfg = load_bins_config()

    # Load predictions
    with open(args.predictions) as f:
        pred_records = [json.loads(line) for line in f]

    # Deduplicate by source_run (multiple text variants per sample)
    seen_runs = set()
    unique_records = []
    for rec in pred_records:
        if rec["source_run"] not in seen_runs:
            seen_runs.add(rec["source_run"])
            unique_records.append(rec)

    if args.max_samples and args.max_samples < len(unique_records):
        unique_records = unique_records[:args.max_samples]

    print(f"▶ Closed-loop evaluation: {len(unique_records)} samples")
    print(f"  Sim dir: {sim_dir}")

    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fout:
        for i, rec in enumerate(unique_records):
            run_name = rec["source_run"]
            prediction = rec["prediction"]
            target = rec["target"]

            try:
                # Load ground truth
                gt_meta = load_ground_truth_setup(sim_dir, run_name)
                gt_intensity = load_ground_truth_intensity(sim_dir, run_name)

                # Convert bins to physical values
                pred_physical = bins_to_physical(prediction, bins_cfg)
                gt_physical = bins_to_physical(target, bins_cfg)

                # Build predicted setup and simulate
                pred_cfg = build_predicted_setup(gt_meta, pred_physical)
                pred_setup = setup_from_dict(pred_cfg)
                pred_result = run_simulation(pred_setup)
                pred_intensity = pred_result["intensity"]

                # Compute beam metrics for predicted setup
                pred_metrics = compute_metrics(
                    pred_intensity, pred_result["sensor_X"], pred_result["sensor_Y"]
                )
                gt_metrics_dict = gt_meta["metrics"]

                # Profile similarity
                profile_sim = compute_beam_similarity(pred_intensity, gt_intensity)

                # Beam metric errors
                centroid_err = np.sqrt(
                    (pred_metrics.centroid_x - gt_metrics_dict["centroid_x"]) ** 2 +
                    (pred_metrics.centroid_y - gt_metrics_dict["centroid_y"]) ** 2
                )
                width_err_x = abs(pred_metrics.sigma_x - gt_metrics_dict["sigma_x"]) / (gt_metrics_dict["sigma_x"] + 1e-12)
                width_err_y = abs(pred_metrics.sigma_y - gt_metrics_dict["sigma_y"]) / (gt_metrics_dict["sigma_y"] + 1e-12)

                # Parameter errors (physical units)
                x_err_m = abs(pred_physical["x_offset"] - float(gt_meta["setup"]["alignment"]["x_offset"]))
                tilt_err_rad = abs(pred_physical["tilt_x"] - float(gt_meta["setup"]["alignment"]["tilt_x"]))

                record = {
                    "source_run": run_name,
                    "pred_bins": prediction,
                    "target_bins": target,
                    "pred_physical": pred_physical,
                    "param_errors": {
                        "x_offset_err_m": float(x_err_m),
                        "tilt_x_err_rad": float(tilt_err_rad),
                    },
                    "beam_errors": {
                        "centroid_err_m": float(centroid_err),
                        "rel_width_err_x": float(width_err_x),
                        "rel_width_err_y": float(width_err_y),
                    },
                    "profile_similarity": profile_sim,
                }
                results.append(record)
                fout.write(json.dumps(record) + "\n")

            except Exception as e:
                print(f"  ⚠ Error on {run_name}: {e}")
                continue

            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{len(unique_records)}] completed")

    # --- Aggregate statistics ---
    if not results:
        print("  No successful evaluations.")
        return

    print(f"\n  [{len(results)}/{len(unique_records)}] successful simulations")

    # Collect metrics
    x_errs = [r["param_errors"]["x_offset_err_m"] for r in results]
    tilt_errs = [r["param_errors"]["tilt_x_err_rad"] for r in results]
    centroid_errs = [r["beam_errors"]["centroid_err_m"] for r in results]
    width_errs_x = [r["beam_errors"]["rel_width_err_x"] for r in results]
    mses = [r["profile_similarity"]["profile_mse"] for r in results]
    psnrs = [r["profile_similarity"]["profile_psnr"] for r in results]
    ssims = [r["profile_similarity"].get("profile_ssim", None) for r in results]

    print("\n" + "=" * 60)
    print("  Closed-Loop Evaluation Results")
    print("=" * 60)

    print("\n  📐 Parameter Errors (predicted vs ground-truth setup)")
    print(f"    x_offset error:     mean={np.mean(x_errs)*1e3:.3f} mm   median={np.median(x_errs)*1e3:.3f} mm")
    print(f"    tilt_x error:       mean={np.mean(tilt_errs)*1e3:.2f} mrad  median={np.median(tilt_errs)*1e3:.2f} mrad")

    print("\n  🔬 Beam Profile Errors (simulated predicted vs ground-truth beam)")
    print(f"    centroid error:     mean={np.mean(centroid_errs)*1e6:.1f} µm   median={np.median(centroid_errs)*1e6:.1f} µm")
    print(f"    rel width err (x):  mean={np.mean(width_errs_x):.1%}   median={np.median(width_errs_x):.1%}")

    print("\n  🖼️  Profile Similarity")
    print(f"    MSE (normalized):   mean={np.mean(mses):.4f}   median={np.median(mses):.4f}")
    print(f"    PSNR:               mean={np.mean(psnrs):.1f} dB    median={np.median(psnrs):.1f} dB")
    if ssims[0] is not None:
        valid_ssims = [s for s in ssims if s is not None]
        print(f"    SSIM:               mean={np.mean(valid_ssims):.4f}   median={np.median(valid_ssims):.4f}")

    print("=" * 60)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
