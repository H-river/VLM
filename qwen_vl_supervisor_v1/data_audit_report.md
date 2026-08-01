# Qwen-VL Supervisor v1: Repository and Data Audit

Audit time: 2026-08-01 (Asia/Singapore)  
Repository commit: `aa7e1f66cfcb497cd5c5ff39c394520668f75584`  
Branch: `v13`  
Working tree: dirty before this task; all pre-existing changes are preserved.

## Decision

The available legal pool is sufficient. No images, labels, or controller episodes need to be generated for supervisor v1. The source manifest will use only the already-generated saturation training family and the already-generated width-relative reflection train/development families. It will not use fixed-pixel reflection.

The resulting records have valid supervision for a static anomaly-policy decision only. The stored `short_history_metrics` in the visual datasets is a clean primary capture at the same initial state, not a time-ordered action/observation history. It is therefore excluded from model-visible history and retained only as non-model provenance. Stored controller traces do not pair each real sequential observation with an anomaly image, so they are inventoried but excluded from v1 SFT rather than being converted into fabricated temporal examples.

## Source inventory

| Cohort | Source records | Setups/pairs | Contents | Legal use |
|---|---:|---:|---|---|
| Original visual train | 96 | 24 setups / 48 pairs | 24 saturation, 24 old fixed-pixel reflection, and 48 paired clean images | Filter saturation-family pairs only; split those 24 pairs setup-disjoint into 18 train and 6 dev pairs |
| Original visual IID heldout | 24 | 6 setups / 12 pairs | 6 saturation, 6 old fixed-pixel reflection, and paired clean images | Evaluation-only; retain the 6 saturation pairs only |
| Original visual severity OOD | 120 | 30 setups / 60 pairs | 30 saturation, 30 old fixed-pixel reflection, and paired clean images | Evaluation-only; retain the 30 saturation pairs only |
| Width-relative train | 60 | 30 setups / 30 pairs | 30 width-relative reflection and 30 paired clean images | Training |
| Width-relative development | 24 | 12 setups / 12 pairs | 12 width-relative reflection and 12 paired clean images | Development only |
| Width-relative IID heldout | 60 | 30 setups / 30 pairs | 30 width-relative reflection and 30 paired clean images | Frozen IID evaluation only |
| Width-relative severity OOD | no valid rendered dataset | 18 suite definitions only | Prior run did not produce a valid severity-OOD conclusion | Excluded; no reflection OOD claim |

All available beam images are grayscale `PNG`, `128 x 128`. The five current metrics use the 128-pixel diagnostic-image coordinate frame. Goal centroid/width values retain the 1024-pixel lab sensor frame and peak intensity retains the raw simulator/sensor scale; the exporter must state these frames explicitly.

Expected unified counts after deterministic construction:

- train: 96 records / 48 counterfactual pairs (48 nominal, 18 saturation, 30 width-relative reflection);
- dev: 36 records / 18 pairs (18 nominal, 6 saturation, 12 width-relative reflection);
- frozen IID: 72 records / 36 pairs (36 nominal, 6 saturation, 30 width-relative reflection);
- frozen OOD: 60 records / 30 pairs (30 nominal, 30 saturation); no width-relative reflection OOD records.

The preferred balanced smoke subset is feasible: 36 train records (12 per class) and 12 dev records (4 per class), with whole counterfactual pairs kept in their source split. Nominal examples are sampled across both legal family cohorts without moving their anomalous pair mates across splits.

## Exact cohort provenance

| Artifact | SHA-256 |
|---|---|
| `runs/vlm_optics_benchmark_20260801_154812/visual_data/dataset_train.jsonl` | `e5a437a0af27339ecf6958e2762a4fc8018f8b7358fb39d4fe1f3e3372958c7a` |
| `runs/vlm_optics_benchmark_20260801_154812/visual_data/dataset_iid_heldout.jsonl` | `4300c5aa770a94ac208049188ab23f8ba6fda043c1ab972cc6ba65464433e225` |
| `runs/vlm_optics_benchmark_20260801_154812/visual_data/dataset_severity_ood.jsonl` | `771a289a3b254c3fb99add1f0088c03aa868b55180fcaabe2d2f43a92c1929a1` |
| `runs/vlm_optics_benchmark_20260801_154812/paired_counterfactuals.jsonl` | `1bc370d80bff5a1c834d6dc43f82f0731c3d4d368011772076ba6ce8762bd31d` |
| `reflection_width_relative/data/dataset_train.jsonl` | `4c2e70767a4c017a2cb42cecc90c91dc7a7efb4e2e1ac50b8fca84335013c2ef` |
| `reflection_width_relative/data/dataset_development.jsonl` | `a4462f184308836425d9f8d98370d88f08dc9e27b7a1b17950e43d5031635de5` |
| `reflection_width_relative/data/dataset_iid_heldout.jsonl` | `2a3c273d252234f001f2b46a0feaec2c4758012e99273974677bc0c88e4133e8` |
| `reflection_width_relative/data/pairs_train.jsonl` | `d5c9b65fd523b9aabe4e2bcee232bade97df192d3a1d13cb6e2a3a813f87da4f` |
| `reflection_width_relative/data/pairs_development.jsonl` | `f10a1776c4598eb5e22f9dce19e9b7790911be348a72617366087644bfb6b75f` |
| `reflection_width_relative/data/pairs_iid_heldout.jsonl` | `990c10db150c849b6419d1bec8f090a8b7533549e8b42199aa3787448ac62a45` |

The manifest builder verifies both every source-dataset hash and every paired-counterfactual metadata hash before deriving records; the manifest index records both sets of pins. The width-relative generator reports `secondary_reflection_primary_sigma_direction_v1` and stores its parameters and counterfactual metric distances per pair. The old fixed-pixel family has source label `secondary_reflection` but is excluded by requiring `family_audit == secondary_reflection_width_relative` for every new reflection record.

## Existing code and frozen identities

- Existing Qwen trainer: `optics_sft/scripts/train_qwen25vl_qlora.py`, SHA-256 `38feb3cfb257ad2b5661d5174e8deec56c2dd63245478408efa3ec5c5eb4c6da`. It provides the repository's Qwen2.5-VL + TRL + PEFT + bitsandbytes framework, but its smoke mode is one step and it does not perform full Trainer checkpoint resume. It remains unmodified.
- Local checkpoint: `/home/jiamo/HF_models/Qwen2.5-VL-3B-Instruct`, complete two-shard model. A 4-bit load and LoRA attachment succeeded during the audit on the local RTX 4080 Laptop GPU.
- Frozen forward ensemble checkpoint: `runs/overnight_v12_semantics_20260731_002709/models/lc_128g_v2/continuous_forward_v12_128g.pt`, SHA-256 `d9b30627c80817f6ecade1959d8cc9e91e7a9de9cc51485153fdbfaa173aca2e`.
- Frozen v12 semantics: `continuous_control_v12/config_v12_semantics_v2.json`, SHA-256 `77cb8bfc8cc064e32ab3cbf473e10f94df5c4ca53869ae235bb5eee782b81ff8`.
- Frozen controller code: `continuous_control_v12/mpc.py` plus the v13 gain-aware probe/replanning and sequential stopping rule. The deployed numerical action source remains H1, one-step CEM, replanning from real observations, and maximum horizon 8.
- Per-step visible action bounds from the frozen v12 configuration are lens x/y `[-0.05, 0.05] mm` and camera x/y `[-0.02, 0.02] mm`; absolute repository sampling-domain limits are `[-3, 3] mm` per axis and are explicitly not claimed as hardware limits.
- Previous image diagnostics: the saturation tiny CNN/multimodal models under `runs/vlm_optics_benchmark_20260801_154812/models/` and the selected width-relative diagnostic `reflection_width_relative/models/selected_diagnostic.pt`. They are binary-family models, not a unified three-class supervisor.
- Metrics baselines: family-specific logistic/history-MLP artifacts beside those image diagnostics. Evaluation v1 will require a dev-frozen, non-oracle arbitration rule or a unified three-class metrics baseline trained on the legal train set.
- Existing split/leakage evidence: `reflection_width_relative/split_manifest.json` and `reflection_width_relative/leakage_audit.json`. Supervisor v1 adds a cross-family manifest validator because the prior audits do not cover the combined dataset.

## Canonical policy mapping

Supervisor v1 uses reversible external names and never emits continuous actions:

| Supervisor enum | Frozen repository policy |
|---|---|
| `standard` | `standard_metrics` |
| `lower_exposure_reacquire` | `reduce_exposure_reacquire` |
| `primary_spot` | `primary_spot_specialist` |

Static target mapping is nominal -> (`nominal`, `standard`, `execute`), saturation -> (`sensor_saturation`, `lower_exposure_reacquire`, `reacquire`), and width-relative reflection -> (`secondary_reflection`, `primary_spot`, `switch_measurement`). No static record is assigned `continue` or `stop`.

## Protection decisions

- Old fixed-pixel reflection is excluded everywhere from the primary supervisor manifests.
- Original IID-heldout and severity-OOD examples are evaluation-only and will not be opened by the smoke generation path.
- Width-relative IID-heldout is evaluation-only.
- No protected predictions will be generated in this task.
- No future observations, post-decision controller outcomes, generator names/parameters, severity labels, setup identifiers, pair identifiers, paths, or split names enter model-visible prompts.
- Formal evaluation remains disabled until its config, hashes, selection rule, metrics, and commands are frozen.

## Known research limitation

Sensor saturation is formally validated. Width-relative reflection is a provisional engineering GO only: Q1 accuracy was 68.75%, the width-quartile gap was 31.25 percentage points, boundary accuracy was 75%, and no valid reflection severity-OOD conclusion exists.
