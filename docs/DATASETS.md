# Datasets

## Policy

All current scientific datasets are simulated unless an individual manifest
proves otherwise. Dataset presence, generation, and redistribution are separate
claims: a local file may be valid evidence while still being unsuitable for
GitHub or public release.

Generated JSONL, arrays, images, videos, caches, and full manifests remain
ignored. Commit only compact metadata and the tiny synthetic fixture under
`data/examples/`. Use `data/manifests/dataset_manifest.template.json` when
moving a dataset to external storage.

## Inventory

| Family | Split information | Provenance | Repository representation | Redistribution |
|---|---|---|---|---|
| Optical simulator random/sweep output | Local `random_v2`, sweep, and single outputs | `optical_sim` configs and generator | Ignored local `optical_sim/outputs/` | `UNKNOWN` |
| Continuous v12 transitions | Corrected retained run uses 128 train, 16 development, 16 test groups; original full config declares 10,500/600/600 but completion is not assumed | `continuous_control_v12.generate_dataset` with corrected simulator semantics | Configs tracked; data/manifests local under `runs/` | `UNKNOWN` |
| V13 gain-fault controller suite | 30 development and 18 protected setup groups in the frozen v13 split | Existing v12 evaluation suite plus hidden simulated command gains | Compact final report tracked; raw controls local | `UNKNOWN` |
| External sequential validation | 30 new external groups × two planner seeds = 60 episodes | Preregistered external suite and frozen Branch-A controller | Compact summary tracked; manifest/results local | `UNKNOWN` |
| Visual saturation benchmark | Train/IID/severity-OOD simulated counterfactuals | Visual anomaly generator and paired clean/anomalous states | Compact summary tracked; raw images/data local | `UNKNOWN` |
| Width-relative reflection | 30 train setups, 12 development setups, 30 IID setups; attempted 18-setup severity-OOD did not produce a valid result | Width-relative clean-primary copy with frozen matching thresholds | Split/suite metadata tracked; JSONL/images ignored | `UNKNOWN` |
| Qwen supervisor | Progress log records 96 train, 36 dev, 72 frozen-IID, 60 saturation-only frozen-OOD records | Hash-checked saturation and provisional reflection sources | Manifest index/schema tracked; record JSONL/images local | `UNKNOWN`; base/model terms also apply |
| Candidate transition Protocol v2 | 320 train base states, 48 dev, 32 each IID/action-OOD/source-OOD/optics-OOD, plus 64 counterfactual pairs; full candidate evaluation reports 1,152 rows | Corrected simulator and frozen sampling contract | Local ignored candidate artifact; source currently untracked | `UNKNOWN` |
| Specialist rebuild v2 | README target: 3,700 setups and 299,700 grid transitions | Specialist builder | Source tracked; external dataset not verified in repo | `UNKNOWN` |
| Synthetic no-op fixture | One record | Hand-authored schema contract | Tracked `data/examples/synthetic_transition_v1.json` | Internal fixture; no external content |

## Split integrity

The retained active experiments generally split by setup/group identity before
derived images, counterfactuals, or episodes. Qwen supervisor manifests also
record image/setup hashes and pair containment. These properties are local
experiment evidence; a public dataset release must rerun the corresponding
validators against the exact released bytes.

## Invalid or excluded data

- Fixed-pixel reflection is excluded from the current supervisor cohort.
- Width-relative severity-OOD produced a serialized counterfactual mismatch
  above the frozen threshold. The threshold was not relaxed and no OOD result
  is reported.
- Partial candidate artifacts, failed Qwen runs, and invalid gain-dependent
  seed attempts are not substitutes for completed splits.
- The synthetic fixture is not eligible for benchmark metrics.

## External storage checklist

Before sharing any dataset, record its logical name, filename/tree convention,
version, SHA-256, byte size, generation command, config, exact code commit,
license/redistribution decision, evidence level, and whether the archive
contains paths or metadata identifying a local machine or person.

