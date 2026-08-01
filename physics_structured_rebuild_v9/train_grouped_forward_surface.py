#!/usr/bin/env python3
"""Train one model to correct the complete fixed 81-action forward surface."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control_rebuild_v3.common import action_basis, tolerance_from_current
from control_rebuild_v3.train_forward import configure
from physics_structured_rebuild_v9.grouped_forward_surface import (
    grouped_context_features,
    grouped_forward_surface_model,
)
from physics_structured_rebuild_v9.runtime import (
    DEFAULT_NATURAL_GRID_FORWARD_STATE,
)

DEFAULT_RUN = (
    REPO_ROOT.parent / "VLM_runs/physics_structured_rebuild_v9_one_seed"
)
DEFAULT_CACHE = DEFAULT_RUN / "combined_natural_inverse_forward_cache.npz"
DEFAULT_OUTPUT = DEFAULT_RUN / "grouped_forward_surface_v9.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--base-forward",
        type=Path,
        default=DEFAULT_NATURAL_GRID_FORWARD_STATE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=55)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split(group_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    validation = np.asarray(
        [
            int(
                hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16],
                16,
            )
            % 5
            == 0
            for group_id in group_ids
        ],
        dtype=np.bool_,
    )
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def metric(
    target: np.ndarray,
    prior: np.ndarray,
    correction: np.ndarray,
    blend: float,
) -> dict[str, Any]:
    error = np.abs(prior + float(blend) * correction - target)
    passed = error <= 1.0
    exact = np.all(passed, axis=2)
    return {
        "count": int(exact.size),
        "strict_all_five_count": int(exact.sum()),
        "strict_all_five_success": float(exact.mean()),
        "mae_in_tolerance_units": float(error.mean()),
        "per_field_tolerance_pass": passed.mean(axis=(0, 1)).tolist(),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    started = time.perf_counter()
    torch, device = configure(int(args.seed), args.device)
    with np.load(args.cache.resolve(), allow_pickle=False) as cache:
        group_ids = np.asarray(cache["group_ids"], dtype=np.str_)
        contexts = np.asarray(cache["contexts"][:, :17], dtype=np.float32)
        truth_states = np.asarray(
            cache["true_candidate_states"],
            dtype=np.float32,
        )
        base_states = np.asarray(
            cache["primary_states"],
            dtype=np.float32,
        )
    current = contexts[:, 12:17].copy()
    current[:, -1] = np.expm1(current[:, -1])
    tolerance = np.stack(
        [tolerance_from_current(values) for values in current]
    ).astype(np.float32)
    target = (
        truth_states - current[:, None, :]
    ) / tolerance[:, None, :]
    prior = (
        base_states - current[:, None, :]
    ) / tolerance[:, None, :]
    basis = np.asarray(action_basis(), dtype=np.float32)
    pseudoinverse = np.linalg.pinv(basis).astype(np.float32)
    base_coefficients = np.einsum(
        "ba,gaf->gbf",
        pseudoinverse,
        prior,
    )
    residual_coefficients = np.einsum(
        "ba,gaf->gbf",
        pseudoinverse,
        target - prior,
    )
    features = np.concatenate(
        [
            grouped_context_features(contexts),
            base_coefficients.reshape(len(contexts), -1),
        ],
        axis=1,
    ).astype(np.float32)
    training, validation = stable_split(group_ids, int(args.seed))
    input_mean = features[training].mean(axis=0)
    input_scale = features[training].std(axis=0)
    input_scale[input_scale < 1e-6] = 1.0
    coefficient_mean = residual_coefficients[training].mean(axis=0)
    coefficient_scale = residual_coefficients[training].std(axis=0)
    coefficient_scale[coefficient_scale < 1e-5] = 1.0
    scaled_features = (
        features - input_mean
    ) / input_scale
    scaled_coefficients = (
        residual_coefficients - coefficient_mean
    ) / coefficient_scale
    model = grouped_forward_surface_model(
        torch,
        features.shape[1],
        basis.shape[1],
        width=int(args.width),
        blocks=int(args.blocks),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
        eta_min=float(args.learning_rate) * 0.05,
    )
    basis_tensor = torch.as_tensor(
        basis,
        dtype=torch.float32,
        device=device,
    )
    coefficient_mean_tensor = torch.as_tensor(
        coefficient_mean,
        dtype=torch.float32,
        device=device,
    )
    coefficient_scale_tensor = torch.as_tensor(
        coefficient_scale,
        dtype=torch.float32,
        device=device,
    )
    residual_tensor = torch.as_tensor(
        target - prior,
        dtype=torch.float32,
        device=device,
    )
    feature_tensor = torch.as_tensor(
        scaled_features,
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(int(args.seed) + 501)
    best_key: tuple[Any, ...] | None = None
    best_state = None
    best_epoch = 0
    stale = 0
    trace = []
    field_weight = torch.as_tensor(
        [1.0, 1.0, 0.75, 0.75, 1.5],
        dtype=torch.float32,
        device=device,
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        shuffled = rng.permutation(training)
        running = 0.0
        for start in range(0, len(shuffled), int(args.batch_size)):
            indices = shuffled[start : start + int(args.batch_size)]
            values = model(feature_tensor[indices])
            coefficients = (
                values * coefficient_scale_tensor
                + coefficient_mean_tensor
            )
            correction = torch.einsum(
                "ab,gbf->gaf",
                basis_tensor,
                coefficients,
            )
            error = correction - residual_tensor[indices]
            smooth = torch.nn.functional.smooth_l1_loss(
                correction,
                residual_tensor[indices],
                reduction="none",
                beta=0.25,
            )
            boundary = torch.nn.functional.softplus(
                (error.abs() - 0.80) * 5.0
            ) / 5.0
            joint = torch.logsumexp(
                error.abs() * 3.0,
                dim=2,
            ) / 3.0
            loss = (
                (smooth * field_weight).mean()
                + 0.30 * (boundary * field_weight).mean()
                + 0.12 * joint.mean()
                + 0.002 * values.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(indices)
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            values = model(feature_tensor[validation])
            coefficients = (
                values * coefficient_scale_tensor
                + coefficient_mean_tensor
            )
            correction = torch.einsum(
                "ab,gbf->gaf",
                basis_tensor,
                coefficients,
            ).float().cpu().numpy()
        validation_metrics = metric(
            target[validation],
            prior[validation],
            correction,
            1.0,
        )
        key = (
            validation_metrics["strict_all_five_count"],
            sum(validation_metrics["per_field_tolerance_pass"]),
            -validation_metrics["mae_in_tolerance_units"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            record = {
                "epoch": epoch,
                "train_loss": running / len(training),
                "validation": validation_metrics,
                "best_epoch": best_epoch,
            }
            trace.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("grouped forward training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        values = model(feature_tensor[validation])
        coefficients = (
            values * coefficient_scale_tensor
            + coefficient_mean_tensor
        )
        correction = torch.einsum(
            "ab,gbf->gaf",
            basis_tensor,
            coefficients,
        ).float().cpu().numpy()
    blends = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    candidates = [
        {
            "blend": float(blend),
            "metrics": metric(
                target[validation],
                prior[validation],
                correction,
                blend,
            ),
        }
        for blend in blends
    ]
    selected = max(
        candidates,
        key=lambda row: (
            row["metrics"]["strict_all_five_count"],
            sum(row["metrics"]["per_field_tolerance_pass"]),
            -row["metrics"]["mae_in_tolerance_units"],
            -abs(row["blend"]),
        ),
    )
    artifact = {
        "version": "grouped_forward_surface_v9_one_seed",
        "model": "grouped_action_basis_forward_surface_v9",
        "seed": int(args.seed),
        "state_dict": best_state,
        "config": {
            "input_dim": int(features.shape[1]),
            "coefficient_count": int(basis.shape[1]),
            "width": int(args.width),
            "blocks": int(args.blocks),
            "dropout": float(args.dropout),
        },
        "input_mean": input_mean,
        "input_scale": input_scale,
        "coefficient_mean": coefficient_mean,
        "coefficient_scale": coefficient_scale,
        "field_blend": np.full(
            5,
            float(selected["blend"]),
            dtype=np.float32,
        ),
        "base_forward": str(args.base_forward.resolve()),
        "base_forward_sha256": sha256(args.base_forward.resolve()),
        "training_cache": str(args.cache.resolve()),
        "training_cache_sha256": sha256(args.cache.resolve()),
        "held_out_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    report = {
        "version": artifact["version"],
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "training_count": int(len(training)),
        "validation_count": int(len(validation)),
        "best_epoch": int(best_epoch),
        "validation_baseline": metric(
            target[validation],
            prior[validation],
            np.zeros_like(correction),
            0.0,
        ),
        "validation_candidates": candidates,
        "selected": selected,
        "trace": trace,
        "seconds": time.perf_counter() - started,
        "source_contract": {
            "training_cache": str(args.cache.resolve()),
            "validation_files_opened": [],
            "held_out_test_files_opened": [],
        },
        "held_out_test_used": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
