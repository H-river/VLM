# Qwen-H1 meta-controller candidate final report

CANDIDATE ONLY — NOT SEALED — FROZEN EVALUATION DISABLED

## 1. Gate

**RED**. RED reasons: `information_insufficient_or_data_gate_red, near_visible_conflict_rate_above_0.20, visible_classifier_macro_f1_below_0.50`. YELLOW reasons: `none`.

## 2. Files changed and added

Structured repository-change evidence: `{"commit_or_push_performed": false, "new_namespace": "qwen_h1_meta_v0_candidate", "preexisting_untracked_paths_preserved": ["qwen_vl_supervisor_v1/artifacts/pilot/", "qwen_vl_supervisor_v1/artifacts/training/pilot_qwen25vl_3b_full96_dev36_seed_2026080101_200step/", "qwen_vl_supervisor_v1/configs/training_pilot_local_200step.yaml", "supervisor_v1_1_candidate/"], "tracked_existing_files_modified": []}`. All candidate outputs are confined to `qwen_h1_meta_v0_candidate`; this aggregate does not claim that an unspecified existing file was unchanged.

## 3. Preservation of the existing structure

Existing-structure preservation evidence: `yes`. The intended additive path keeps the anomaly supervisor, learned forward ensemble, default Learned-H1 and state-machine authority separate.

## 4. Feature-flag-off equivalence

Off-mode equivalence: `yes`. Missing evidence is not a pass.

## 5. Qwen schema and authority boundary

Qwen emits only the frozen discrete JSON schema: decision, observation request, objective profile, allowlisted mask, per-canonical-actuator direction categories, step scale, risk mode, confidence and reason codes. Qwen never owns continuous actuator values, legal bounds, targets, safety decisions or dispatch.

## 6. Deterministic compiler

Compiler mapping: `protocol/compiler_mapping.json`. Balanced/default identity, trust-region shrinkage, fixed masks/directions, confidence fallback and H1-only dispatch remain deterministic and hash-pinned.

## 7. Candidate data

Audited records: `72`; preregistered setups are train/dev/eval `16/8/12`, with three target counterfactuals per setup. Known overlap: `0`; cross-split overlap: `0`.

## 8. Information sufficiency

Gate: `RED_STOP`. Exact/rounded conflicting collision groups: `0` / `0`. Near-visible conflicts: `54/54 = 1.000000` versus maximum `0.200000`. Visible-only grouped classifier macro-F1: `0.019166` versus minimum `0.500000`. Hidden-setup macro-F1 gain: `0.000000`. Known/cross-split identity overlap: `0` / `0`.

## 9. Three-seed training stability

Completed preregistered seeds: `none`. Offline valid-JSON mean/std: `not available` / `not available`. Configuration-regret mean/std: `not available` / `not available`.
Execution state: `not_run_due_to_information_sufficiency_gate` / `not_run_due_to_information_sufficiency_gate`. A RED information-sufficiency gate forbids starting QLoRA or downstream evaluation.

## 10. Controller comparisons

| Method | Complete | Strict success | Final normalized error | Config regret | Wall-clock s |
|---|---|---:|---:|---:|---:|
| not available | no | not available | not available | not available | not available |

Default H1 is the original compute budget. Dual-budget default is the required compute-fair control. If Qwen only beats original default and not dual-budget default, any gain may be extra search computation rather than Qwen reasoning.

## 11. Paired closed-loop evidence

Episodes/setups: `not available` / `not available`. Qwen stable candidate gain supported: `no`. Paired bootstrap evidence is preserved verbatim in `closed_loop_results.json`; no missing Qwen trace is relabeled as Qwen performance.
Closed-loop execution state: `not_run_due_to_information_sufficiency_gate`.
The closed-loop harness uses candidate-manifest/synthetic measurement-validity and supervisor state; it does not run the actual existing Qwen anomaly-supervisor inference path. The sequential reobserve recovery backend is unverified. This is neither full-stack nor end-to-end validation.

## 12. Reasoning ablations

Candidate reasoning criteria all supported: `no`. Ablation seed count: `not available`. Required target, history, image, metrics, actuator-semantics, uncertainty and reason-code ablations remain candidate simulator/shadow evidence only.
Ablation execution state: `not_run_due_to_information_sufficiency_gate`.

## 13. Safety, fallback and state machine

Regression status: `PASS`; dispatch gate: `yes`; state-machine tests: `yes`; H3 disabled: `yes`. Reobserve/stop are non-dispatch outcomes, anomaly recovery has priority, and fallback still passes the existing bounds/budget/validity path.

## 14. Remaining blockers

RED: `information_insufficient_or_data_gate_red, near_visible_conflict_rate_above_0.20, visible_classifier_macro_f1_below_0.50`. YELLOW/negative: `none`.
Old-source partial generation files, if present, remain isolated under `artifacts/interrupted_generation_old_import_20260802T143042` and are not treated as legal current candidate data or downstream evidence.

## 15. Claim boundary

The candidate implementation reached its preregistered information audit, which triggered a hard stop. Qwen training, inference, closed-loop evaluation and reasoning ablations were not run, so no Qwen reasoning or control-performance claim is supported.

This is not evidence that Qwen directly controls actuators, learned complete optical physics, replaces the forward model, converts offline accuracy into closed-loop success, generalizes to frozen data, validates temporal recovery, validates a full stack/end-to-end system or hardware, or enables H3.

The anomaly-supervisor dev diagnosis result (96.91% ± 1.67 pp) and the separate Learned-H1 candidate result (37/48, 77.1%) come from different experiments. They must not be multiplied and are not an end-to-end system success rate.

## 16. Commands, configs, checkpoints, manifests and hashes

Recorded commands: `6`. See `commands.md`. No QLoRA checkpoint was created because training was not run after the RED data gate. Protocol/config/manifest identities remain hash-pinned. Candidate-relative evidence SHA-256 values are in `machine_summary.json`; the complete deterministic inventory is in `artifact_hashes.json`, whose explicit rule excludes the hash manifest itself.

## 17. Runtime and memory

Data generation/audit pipeline elapsed: `1082.7339` seconds (`/usr/bin/time` wall `18:03.94`), maximum resident set `933792` KiB. Training wall time across supplied completed manifests: `not available` seconds. GPU training peak MiB (recorded JSON): `null`; QLoRA/inference start state is explained by `QLoRA and inference were not started after the hard data gate returned RED_STOP`. Closed-loop wall time: `not available` seconds. Recorded GPU identity: `NVIDIA GeForce RTX 4080 Laptop GPU`; reporter machine query: `{"driver_version": "580.173.02", "index": 0, "memory_total_mib": 12282, "name": "NVIDIA GeForce RTX 4080 Laptop GPU"}`.

Formal frozen evaluation was not run by this reporter, the protocol remains unsealed, no scientific conclusion is made, and no commit or push is performed.

QWEN-H1 META-CONTROLLER CANDIDATE — READY FOR HUMAN REVIEW
