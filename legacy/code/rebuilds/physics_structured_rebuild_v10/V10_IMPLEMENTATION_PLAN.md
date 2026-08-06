# V10 response-surface specialist rebuild plan

Date: 2026-07-29

## Frozen baseline and audit

The accepted system remains the v9 artifact reported by
`physics_structured_rebuild_v9/CYCLE_RESULTS_2026-07-28.md`. No v9 source,
checkpoint, validation file, or deployment manifest will be modified.

The accepted metrics reproduced from the authoritative v9 JSON are:

| Task | Correct route | End to end |
|---|---:|---:|
| direction strict all-five | 54.00% | 54.00% |
| forward strict all-five | 61.67% | 60.00% |
| inverse physical success, state and visual combined | 61.00% | 59.33% |
| image measurement strict all-five | 76.67% | 76.67% |
| Qwen routing (frozen reported baseline) | 98.76% | not applicable |

The forward identifiability audit contains 14,705 independent fixed-source
groups and 1,191,105 transitions (81 per group), with no exact duplicate
visible contexts. The 17 visible context values are the 12 setup fields in
`specialist_rebuild_v2.common.SETUP_FIELDS` followed by the five current
measurements in `STATE_FIELDS`. The action grid is the Cartesian product
`(-0.05, 0, +0.05)^2 x (-0.02, 0, +0.02)^2` for lens x/y and camera x/y.
It contains one zero-, eight one-, 24 two-, 32 three-, and 16 four-actuator
actions.

The latest controlled diagnostics establish steep and regime-dependent
fixed-source behavior, including focus, aperture-clipping, camera-boundary,
and quantized sensor-extraction effects. They do not establish
non-identifiability for the present fixed-source distribution. Hidden source
phase/shape experiments are retained as a robustness warning only; hidden
state is not added to the v10 deployable context.

## Files

Only a new `physics_structured_rebuild_v10/` package and new
`/home/jiamo/VLM_data/physics_structured_rebuild_v10/` and
`/home/jiamo/VLM_runs/physics_structured_rebuild_v10_pilot/` outputs will be
created.

Planned package files:

- `config_v10.json`: frozen full-generation and pilot configuration.
- `contracts.py`: fields, action representation, hashes, tolerances, and
  success labels.
- `generate_dataset.py`: resumable stratified grouped simulation and
  training-only auxiliary/image labels.
- `validate_dataset.py`: schema, action order, finiteness, provenance, and
  cross-split hash guards.
- `models.py`: pointwise-ID and structured grouped response-surface models.
- `train_forward_ablation.py`: controlled A-D training and checkpointing.
- `evaluate.py`: grouped diagnostics, group bootstrap, paired bootstrap,
  direction derivation, and inverse success metrics.
- `train_inverse_ranker.py`: direct multi-positive 81-action numerical ranker.
- `train_visual_inverse.py`: direct current/target-image 81-action scorer.
- `train_direction_calibrator.py`: forward-derived threshold calibration.
- `freeze_and_evaluate.py`: immutable selection manifest and one-time locked
  evaluation.
- `run_v10_cycle.py`: reproducible gated pilot orchestrator.
- `tests/`: contract, model-shape, leakage, loss, and smoke tests.
- `V10_CYCLE_RESULTS.md`: final controlled evidence and decisions.

## Split policy and generation

The intended full dataset has 2,400 independent training groups, 320
system-like development groups, and 320 separately seeded locked-test groups.
All 81 actions inherit their parent group split. Setup and setup-plus-current
context hashes must be unique across splits. The locked-test JSONL is created
by the generator but is not loaded by training, development evaluation, or
model selection code.

The distribution is preregistered as:

| Regime | Fraction |
|---|---:|
| ordinary natural requests | 35% |
| focusing | 15% |
| aperture clipping | 15% |
| camera boundary | 15% |
| high offset / actuator interaction | 10% |
| tolerance-boundary mixture | 10% |

Every group contains the full action grid, so three- and four-actuator actions
make up 59.26% of transitions. Visual requests are drawn from non-zero legal
actions with cardinality-balanced source selection. Clean, noise, blur,
saturation, asymmetric gain/crop, and camera artifact conditions are
deterministic functions of the group/request seed.

Where available, the generator records lens transmission/clipping fraction,
captured sensor power, geometric focus residual, distance to sensor boundary,
regime, action cardinality, interaction category, and calibrated images for
the current state and selected visual targets. These are supervision and
diagnostic fields only; the numerical deployable model consumes only the 17
visible context values.

The full set is expected to require roughly 45-75 minutes on two pinned CPU
workers. The first cycle will run a 384/160/160 group pilot (57,024
transitions) if the full run cannot fit the current execution window. The
manifest and final report must label that limitation; it must not be presented
as the full experiment.

## Controlled forward matrix

All four experiments use seed 2026072901, equal numbers of training groups,
the same preprocessing, optimizer steps, early-stopping rule, and development
evaluator.

| ID | Data | Model | Objective |
|---|---|---|---|
| A | size-matched existing fixed-source training groups | pointwise action-ID grouped model | fieldwise Huber |
| B | new system-aligned training groups | pointwise action-ID grouped model | fieldwise Huber |
| C | new system-aligned training groups | pointwise action-ID grouped model | tolerance-aware joint loss |
| D | new system-aligned training groups | explicit structured 81-action surface | tolerance-aware joint loss |

All models predict tolerance-normalized five-output changes for every action.
The pointwise control uses a learned action-ID embedding. D exposes signed
motion, moved-actuator masks, cardinality, and all pairwise, third-order, and
fourth-order interaction terms. D has individual-actuator components, a
shared grouped interaction block, partially separate centroid/width/peak
heads, and a per-output log-variance head. All 81 actions are processed in one
forward pass.

The metric-aware loss is preregistered as:

`weighted Huber(r) + 0.35 * softplus(smooth_max(abs(r)) - 1) + 0.02 * NLL`

where `r` is already normalized by the production tolerance and smooth-max is
`logsumexp(6 * abs(r)) / 6`. No Gaussian-beam identity is imposed in clipping
or non-Gaussian regimes.

## Metrics

Forward primary: strict all-five tolerance success on system-like development
requests using their naturally sampled requested action.

Forward secondary:

- strict success across the full response surface;
- per-output tolerance success and normalized residual quantiles;
- centroid, width, and peak blocks;
- action cardinality and optical regime;
- uncertainty calibration/coverage for D;
- group-bootstrap 95% confidence intervals;
- paired group-bootstrap intervals for B-A, C-B, D-C, each candidate-A, and
  each candidate-frozen-v9.

Numerical inverse primary: top-1 simulator-derived physical target success.
Secondary: top-k, first-positive rank, available-positive count including
1/81 cases, action cardinality, and regime.

Visual inverse primary: top-1 physical success from images. Measurement
accuracy is diagnostic only.

Direction primary: strict all-five direction accuracy. Secondary: per-field
accuracy/macro-F1 and boundary-confusion rates.

## Preregistered decision gates

All confidence intervals resample independent groups/requests, never
transitions.

Forward acceptance requires all of:

1. primary development improvement over A of at least 3 percentage points;
2. paired-bootstrap 95% lower bound above zero versus A;
3. paired-bootstrap 95% lower bound above zero versus the frozen v9 forward
   evaluated on the same new development requests;
4. ordinary-regime paired lower bound above -2 percentage points;
5. no output tolerance accuracy regression larger than 3 percentage points.

If multiple candidates pass, select by primary success, then worst-output
accuracy, then smaller model. If none passes, retain frozen v9 forward. Training
loss cannot promote a model.

Numerical inverse acceptance requires a 3-point top-1 gain and paired lower
confidence bound above zero versus frozen v9 on development, with no
1/81-positive regression larger than 3 points. The listwise loss uses all
positives, inverse-positive-count request weights, and additional weight for
near misses that fail one tolerance.

Visual inverse acceptance requires a 3-point top-1 gain with paired lower
bound above zero versus the existing direct visual scorer and no clean-image
regression larger than 2 points.

Direction retains a separate specialist only if it beats direct thresholding
of the selected forward surface by at least 2 points with paired lower bound
above zero and does not regress any field by more than 2 points.

After all choices are frozen, the selected v10 components (or explicit v9
retentions) are hashed in a freeze manifest. The locked test may then be
loaded exactly once. The old 150-case system evaluation is reserved only for
the final comparability run and cannot change the freeze decision.

## Reproducibility and safety

All output writes refuse to overwrite completed artifacts. Configurations,
source hashes, split hashes, seeds, environment versions, checkpoints,
registries, and metrics are saved. Long generation/training commands use
`taskset -c 0,8` through the v9 safety wrapper with the existing 2 GiB
available-RAM, 8 GiB GPU-memory, and 80 C temperature stops. Every pipeline is
first exercised with two to eight groups and one to two epochs.
