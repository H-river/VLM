# Qwen-VL Optics Supervisor Dataset v1

Sensor saturation: formally validated.
Width-relative reflection: provisional GO for Qwen-VL training, with unresolved
narrow-beam and boundary generalization limitations.

## Summary

This setup-disjoint counterfactual dataset trains a Qwen2.5-VL model to act only as a high-level optics supervisor. Each record contains one current 128x128 grayscale beam image, current five metrics, goal metrics, an empty causally-valid history for the static records, an eight-step remaining budget, and visible frozen actuator constraints. The target is strict three-field JSON: diagnosis, measurement policy, and a high-level action. It never contains continuous actuator commands, rationales, or self-reported confidence.

The source of truth is the versioned manifest in `qwen_vl_supervisor_v1/manifests/`; Qwen chat JSONL under `qwen_vl_supervisor_v1/sft/` is deterministic derived data.

## Splits

| Split | Total | Nominal | Saturation | Width-relative reflection | Use |
|---|---:|---:|---:|---:|---|
| train | 96 | 48 | 18 | 30 | SFT |
| dev | 36 | 18 | 6 | 12 | Checkpoint/threshold selection and smoke generation |
| frozen IID | 72 | 36 | 6 | 30 | Future formal evaluation only |
| frozen OOD | 60 | 30 | 30 | 0 | Future formal saturation severity evaluation only |

The saturation train/dev rows are a deterministic whole-pair partition of the original legal saturation train pool: 18 pairs remain train and 6 pairs become dev. Width-relative reflection uses its existing 30 train pairs and 12 development pairs unchanged. No existing IID-heldout or severity-OOD pair is moved into train or dev.

The balanced smoke exports (36 train, 12 dev) are sampling views of the existing splits, not new splits. Exact class balancing does not imply that both members of every selected counterfactual pair appear in the small view; every member remains assigned to the same underlying manifest split.

## Targets and provenance

Diagnosis and measurement policy are derived directly and reversibly from the stored `fault_type` and `oracle_recovery_decision`. High-level static action uses the frozen `static_policy_action_mapping_v1`:

- `standard` -> `execute`;
- `lower_exposure_reacquire` -> `reacquire`;
- `primary_spot` -> `switch_measurement`.

This action projection is explicitly marked as derived target provenance. The sources do not directly supervise nominal `execute` as a temporal control outcome. No static example receives `continue` or `stop`; those actions remain part of the runtime enum but must be scored only when a future causally-valid temporal dataset supplies supervision.

The source `short_history_metrics` value is a clean same-state capture, not an action/observation history. It is excluded from model input. Existing controller episode artifacts lack paired per-step anomaly images and are also excluded from SFT.

## Coordinate frames

- Current centroid and width metrics: `diagnostic_image_128px` (the 128x128 diagnostic patch).
- Goal centroid and width metrics: `lab_sensor_1024px_and_raw_peak` (the 1024x1024 lab/simulator sensor frame).
- Goal peak intensity: raw simulator/sensor scale.
- Current peak intensity: normalized diagnostic-image scale used by the source visual benchmark.

These deliberately different frames are named in every model-visible metric object; values must not be directly subtracted across frames without the frozen control stack's transformations.

## Legal use and exclusions

Legal: high-level supervisor SFT, development-only engineering smoke tests, and future preregistered frozen evaluation after checkpoint/config freeze.

Excluded from training:

- every old fixed-pixel secondary-reflection record, including its paired nominal member;
- all existing IID-heldout/protected and severity-OOD records;
- 26 partial width-relative OOD PNGs that have no valid dataset manifest;
- controller traces that lack corresponding time-aligned anomaly images;
- generator parameters, setup identifiers, severity, boundary and width labels as model inputs.

Manifest metadata remains outside the model-visible chat. Filenames and filesystem paths never enter prompt text.

## Known limitations

Width-relative reflection is a provisional engineering family, not a strict preregistered pass. The frozen prior result was 68.75% accuracy for Q1 narrow beams, a 31.25 percentage-point width-quartile gap, 75% boundary accuracy, and no valid severity-OOD conclusion. The future evaluation keeps Q1 and boundary slices visible.

The data are simulated and static. A successful 20-step smoke proves pipeline integrity only; it does not validate model accuracy, closed-loop control, or the overall benchmark.
