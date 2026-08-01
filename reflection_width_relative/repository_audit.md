# Width-relative reflection repository audit

Audit completed: 2026-08-01T17:41:19+08:00  
Repository commit: `aa7e1f66cfcb497cd5c5ff39c394520668f75584`  
Branch: `v13`

## Evidence boundary and worktree

This continuation starts from a dirty worktree containing user-authored modified and untracked files. No existing file is reverted or overwritten. Reusable code for this experiment is added as a new module and artifacts are isolated under `reflection_width_relative/`.

The old fixed-pixel reflection cohorts are used only to audit implementation contracts and to perform the required deterministic replay. Their held-out predictions and protected outcomes are not used for parameter selection, model selection, debugging, or gate changes. The frozen numerical forward model and controller are not retrained or modified.

## Previous secondary-reflection generator

The implementation is `vlm_optics_benchmark/visual_anomalies.py` (audited SHA-256 `233f5c56676b5d1be6ee5f72b0caf8df9897e3f1b51cebe1720e5cd2519f1158`). `inject_anomaly` creates a bilinearly shifted, zero-filled copy of the already canonicalized 128x128 clean image, adds `reflection_amplitude_fraction * shifted_copy`, and divides the sum by its peak. Reflection power is added before peak normalization; it is neither redistributed from the primary nor conserved after normalization. The old path has no component-width parameter.

Old development/IID reflection parameters are amplitude 0.40-0.55, separation 24-32 canonical pixels, and direction uniform on 0 to 2 pi. Old severity-OOD parameters are amplitude 0.60-0.75 and separation 34-44 pixels. `severity_for` derives all values deterministically from the case ID, family, and split. The fixed-pixel path remains untouched by this experiment.

## Five-metric counterfactual construction

The five metrics are exactly centroid x, centroid y, width x, width y, and peak intensity. `matched_clean_counterfactual` applies a translation plus independent x/y scaling to the clean canonical beam, fits the first four moments with bounded nonlinear least squares, and scales the peak to the anomalous target. `normalized_metric_distance` divides absolute differences by `[1 px, 1 px, 2 px, 2 px, 0.05 peak]`. Frozen acceptance is maximum normalized field difference <= 0.25 and total normalized L2 <= 0.40.

The earlier files only recorded floating pre-serialization distances. The new experiment must additionally remeasure both saved 8-bit PNGs and record the post-serialization distance.

## Previous model protocol

`_torch_models` defines the prior small diagnostic CNN: grayscale 128x128 input; convolution channels 1->8 (5x5, stride 2), 8->16 (3x3, stride 2), and 16->32 (3x3, stride 2), ReLU after each, adaptive 4x4 average pooling, flattening, a 32-unit ReLU head, and one binary logit. The image+metrics variant concatenates five standardized metrics before the same head.

`_fit_tiny_model` uses deterministic 0/90/180/270-degree rotation augmentation, Adam with learning rate 0.002 and weight decay 0.0001, binary cross entropy with logits, batch size 32, and 45 epochs. The old base seed is 2026081201; the reflection image and image+metrics instances used seeds 2026081202 and 2026081212. The metrics baseline is standardized balanced logistic regression with maximum 2,000 iterations. The short-history baseline is a standardized one-hidden-layer MLP with 16 units, maximum 2,000 iterations, no early stopping, and the same base seed.

## Previous split and leakage protocol

The prior train and IID cohorts used setup suffixes 0-7 and 8-9 respectively from the old primary diagnostic suite; the prior severity-OOD cohort used the old external suite. Splits are setup-group disjoint, sample and candidate order are deterministically shuffled, image filenames contain only sample hashes, and labels/severity/setup identifiers/oracle decisions remain outside `model_input`.

Existing leakage checks cover shuffled images with metrics preserved, eight-pixel border masking, metadata-only classification, exact image hashes, nearest-neighbor pixel MSE, and setup overlap. The new audit extends these with irrelevant-background masking, previous-cohort setup/hash exclusion, and explicit fixed dimension/padding/normalization checks.

## Measurement specialist and recovery policy

`measurement_rebuild_v3/predict.py` is the audited image-measurement switch point. Its analytic path linearizes the image, applies the valid mask, and derives centroid, widths, peak, captured integral, valid fraction, and saturation fraction; its learned path predicts a calibrated correction. The existing reflection control audit implements the oracle `primary_spot_specialist` contract by replacing combined-spot metrics with clean-primary metrics before invoking the unchanged controller. This is an observation/measurement switch and adds no actuator actions.

## Frozen deterministic control replay

`vlm_optics_benchmark/visual_control.py` (audited SHA-256 `c309a29e96913730b6b0bf015c92bd865415f50d6350ef0f0287dde9397f2a50`) loads the frozen learned H1 ensemble, instantiates the unchanged H1-CEM controller, replans after each observation, and applies the frozen continuation rule through at most eight real control steps. The v13 config SHA-256 is `4f5652b0532c039d22684dc89e3c4c90aa2c34f5226c86a85db6f71ae65e3a18`.

At 2026-08-01T17:39+08:00, a clean-process regeneration of all old training pairs byte-matched `pairs_train.jsonl` (`cmp` status 0). The reproduced old reflection PNG `img_f365ee75da91a2177d4ab2dc.png` matched the stored image at SHA-256 `046ed619a39b383128aa71d2f48479c7b106ea0cd24dfb1519bb2a6b5ca9611b`. At 2026-08-01T17:40+08:00, the deterministic control replay test regenerated one oracle episode per stratum for both anomaly families and compared every stored trace exactly: 1 test passed.

## New cohort exclusions

Every new suite must exclude the forward-model training groups and the complete setup IDs/hashes in:

- `runs/v12_mpc_h1_h3_diagnosis_20260731_111829/suite/evaluation_suite_manifest.json` (all previous reflection train and IID setups);
- `runs/vlm_optics_benchmark_20260801_154812/external_suite_manifest.json` (all previous reflection severity-OOD/control setups);
- the three v13 fresh-holdout suite manifests inventoried by the prior external preregistration.

Each later new suite must also exclude every earlier new suite. Old protected result tables are not an exclusion source and will remain unopened for tuning.
