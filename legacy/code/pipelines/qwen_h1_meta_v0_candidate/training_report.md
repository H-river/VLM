# Qwen-H1 meta-controller candidate training report

CANDIDATE ONLY — NOT SEALED — FROZEN EVALUATION DISABLED

Reporting gate: **RED**. This is engineering evidence from candidate train/dev only, not a scientific or frozen-evaluation conclusion.
Downstream execution state: `{"ablation": "not_run_due_to_information_sufficiency_gate", "closed_loop": "not_run_due_to_information_sufficiency_gate", "offline": "not_run_due_to_information_sufficiency_gate", "training": "not_run_due_to_information_sufficiency_gate"}`.

## Preregistered training contract

- Adapter: `qwen_h1_meta_v0` (independent from the anomaly-supervisor adapter).
- Seeds: `2026080201, 2026080202, 2026080203`.
- Steps/eval/save: `200/25/25`.
- Batch/gradient accumulation: `1/4`.
- QLoRA: rank `16`, alpha `32`, dropout `0.05`, NF4 4-bit double-quant with BF16 compute, gradient checkpointing `True`.
- Checkpoint selection: lowest complete dev eval_loss; exact tie selects earlier step; never select by closed-loop result or best seed.

## Per-seed results

| Seed | Run | Steps | Train loss | Best dev loss | Final dev loss | Valid JSON | Decision macro-F1 | Config exact | Direction acc/F1 | Compiled valid | Wall s | Peak allocated/reserved bytes | Best checkpoint |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026080201 | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available |
| 2026080202 | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available |
| 2026080203 | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available | not available |

## Data and stability audit

- Information gate: `RED_STOP`; audited records: `72`; export permitted: `no`.
- Exact visible-label conflicts: `0`; rounded conflicts: `0`.
- Near-visible conflicting-pair rate: `1.0000` (maximum `0.2000`); visible-only grouped-classifier macro-F1: `0.0192` (minimum `0.5000`).
- Known identity overlap: `0`; candidate cross-split overlap: `0`.
- Data generation exit/elapsed/max RSS: `3` / `1082.7339` seconds / `933792` KiB.
- Three-seed valid-JSON mean/std: `not available` / `not available`.
- Three-seed configuration-regret mean/std: `not available` / `not available`.

Offline JSON and field accuracy are diagnostics only. They are not closed-loop success and do not establish frozen generalization.
When the information-sufficiency gate is RED, absent training/offline/closed-loop/ablation artifacts mean `not_run_due_to_information_sufficiency_gate`, not an unreported run.
