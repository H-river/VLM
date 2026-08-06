# V11/V12 implementation report

Date: 2026-07-29

## Outcome

Two new isolated tracks are implemented:

- `physics_structured_rebuild_v11/`: legacy 81-action baseline-parity,
  scaling, matched ablations, group OOF prior/residual training, and an
  interpretation gate.
- `continuous_control_v12/`: versioned continuous transition schema,
  deterministic mixed sampling, simulator-derived targets, estimated
  reachability, numerical residual ensemble, CEM MPC, structured goal
  boundary, group-level evaluation, and smoke tests.

No file under `physics_structured_rebuild_v9/` or
`physics_structured_rebuild_v10/` was changed. No v10 JSONL, locked-test data,
checkpoint, or protected artifact was used by the implementation or smoke
runs.

## Files added or changed

Top-level:

- `V11_V12_AUDIT_AND_DESIGN.md`
- `V11_V12_IMPLEMENTATION_REPORT.md`

V11:

- `physics_structured_rebuild_v11/__init__.py`
- `physics_structured_rebuild_v11/config_v11.json`
- `physics_structured_rebuild_v11/contracts.py`
- `physics_structured_rebuild_v11/models.py`
- `physics_structured_rebuild_v11/generate_system_aligned.py`
- `physics_structured_rebuild_v11/train.py`
- `physics_structured_rebuild_v11/README.md`
- `physics_structured_rebuild_v11/tests/`

V12:

- `continuous_control_v12/__init__.py`
- `continuous_control_v12/config_v12.json`
- `continuous_control_v12/contracts.py`
- `continuous_control_v12/simulator.py`
- `continuous_control_v12/sampling.py`
- `continuous_control_v12/schema.py`
- `continuous_control_v12/generate_dataset.py`
- `continuous_control_v12/validate_dataset.py`
- `continuous_control_v12/generate_targets.py`
- `continuous_control_v12/reachability.py`
- `continuous_control_v12/world_model.py`
- `continuous_control_v12/train_forward_model.py`
- `continuous_control_v12/mpc.py`
- `continuous_control_v12/goal.py`
- `continuous_control_v12/evaluation.py`
- `continuous_control_v12/evaluate_legacy_grid.py`
- `continuous_control_v12/evaluate_controller.py`
- `continuous_control_v12/run_mpc.py`
- `continuous_control_v12/README.md`
- `continuous_control_v12/schemas/`
- `continuous_control_v12/tests/`

## Important decisions

- V11 retains the exact legacy action ordering and uses the exact repository
  46-feature transformation for structured runs.
- The v11 baseline has 988,037 parameters in the smoke configuration, close
  to the reported v9 neural capacity. It uses standardized five-fold
  group-OOF Ridge priors and learns normalized residuals.
- V11 baseline, ordinary-loss, opaque-action, and legacy-distribution
  configurations share group counts, capacity, epochs, early stopping, and
  seed. Conceptual comparisons remain `not_interpretable` until the common
  system-aligned baseline passes both fit and development parity checks.
- V12 deployable state contains eight setup fields, four absolute positions,
  five current metrics, and a physical continuous action. Simulator
  resolution/backend/defocus/source assumptions are fixed and recorded.
- V12 structured action features include normalized physical action values,
  squares, six pairwise products, and four position-action products.
- Metric prediction is a tolerance-normalized delta around current metrics.
  A no-op mask forces exact zero predicted delta.
- Ensemble spread plus learned log variance supplies uncertainty. Available
  captured-power, clipping, camera-boundary, and actuator-limit labels have
  configurable auxiliary losses.
- The optional image-conditioning ablation uses a fixed pooled intensity
  embedding for one-step prediction. It is not allowed in imagined multi-step
  MPC until a future-image model exists.
- CEM always projects candidates against per-step and absolute bounds,
  executes one action, observes the simulator, and replans. It returns
  best-effort/likely-unreachable profiles rather than claiming success.
- Reachable targets are generated from legal `q*` simulator states.
  Mismatched profiles come from another simulator-derived optical group and
  are labelled ambiguous until the continuous oracle verifies them.
- Group IDs, not transitions/actions, are used for splits, hashes, bootstrap
  intervals, and reported sample counts.

## Assumptions and unresolved simulator limitations

- The simulator has no documented hardware limits. ±3 mm is the repository
  setup-sampling domain and is explicitly not claimed as a hardware limit.
- `power_w` is recorded, but the current Gaussian source implementation
  normalizes the field and ignores power.
- Sensor extraction is nearest-neighbour and discontinuous under small camera
  motions.
- The five control metrics use the legacy base/lab-frame pseudo-pixel
  convention, while stored intensity arrays are sensor-frame images.
- Phase is available only from privileged complex simulator fields and is
  null by default. It is not a deployed input.
- Source phase/shape/alignment are fixed. If allowed to vary, v9 diagnostics
  show intensity-only current observations can be non-identifying.
- Full image storage at the preregistered group count is large. The numerical
  model does not require it; full generation defaults to null image refs.
- Continuous oracle results are estimates. Repeated optimizer agreement
  supports `candidate_infeasible` but is not a physical proof.

## Deterministic verification performed

Unit tests:

```text
15 passed in 0.60 s
```

V11 data smoke:

```text
32 train groups / 2,592 transitions
8 development groups / 648 transitions
cross-split group overlap: 0
```

V11 baseline smoke:

```text
parameters: 988,037
fit normalized MAE: 2.267
development normalized MAE: 1.952
fit full-surface strict: 4.71%
development full-surface strict: 2.16%
parity gate: not_interpretable (all four checks false)
```

The separate 32-group/60-epoch overfit diagnostic reached normalized MAE
0.450, 70.29% full-surface strict accuracy, and 78.12% natural-action
accuracy on its repeated diagnostic set. It confirms learning capacity but is
not eligible to unlock the scientific interpretation gate.

These diagnostic values do not support an ablation conclusion.

V12 dataset/schema smoke:

```text
2 train groups / 28 transitions
1 development group / 14 transitions
1 test group / 14 transitions
cross-split group overlap: 0
schema and manifest validation: passed
simulator-derived targets: 12
```

V12 16-transition overfit diagnostic:

```text
normalized MAE: 0.0494
strict all-five: 100%
no-op maximum predicted normalized residual: 0
```

V12 small ensemble smoke:

```text
fit normalized MAE: 1.249
fit strict all-five: 28.57%
development normalized MAE: 1.884
development strict all-five: 14.29%
serialization/inference: passed
no-op maximum predicted normalized residual: 0
```

The fixed pooled-image conditioning ablation also completed its smoke fit and
serialization/inference check. Its development normalized MAE was 1.712 on
one group, so it remains diagnostic-only.

V12 oracle-MPC smoke on one simulator-derived one-step target:

```text
initial normalized distance: 4.121
final normalized distance: 0.643
steps: 3
status: reached
illegal proposed/executed actions: 0
q* used by controller: false
```

The learned-model MPC smoke did not reach the same target and ended at
normalized distance 7.158. This is expected from a 2-group/28-transition
training smoke and is recorded as a limitation, not evidence about the full
method.

One-group legacy-grid probe:

```text
canonical actions: 81
legacy ordering preserved: true
strict all-five: 1/81 = 1.23%
no-op consistency: exact
```

No smoke result is a claim that v11/v12 outperforms v9.

## Exact next commands

Set:

```bash
PY=/home/jiamo/miniconda3/envs/optical_sim/bin/python
```

Generate and validate a fresh V12 small dataset:

```bash
$PY -m continuous_control_v12.generate_dataset \
  --smoke \
  --output-dir /home/jiamo/VLM_data/continuous_control_v12_smoke

$PY -m continuous_control_v12.validate_dataset \
  --data-dir /home/jiamo/VLM_data/continuous_control_v12_smoke

$PY -m continuous_control_v12.generate_targets \
  --data-dir /home/jiamo/VLM_data/continuous_control_v12_smoke
```

Overfit and train the small forward model:

```bash
$PY -m continuous_control_v12.train_forward_model \
  --overfit \
  --data-dir /home/jiamo/VLM_data/continuous_control_v12_smoke \
  --run-dir /home/jiamo/VLM_runs/continuous_control_v12_smoke/overfit \
  --device cpu

$PY -m continuous_control_v12.train_forward_model \
  --smoke \
  --data-dir /home/jiamo/VLM_data/continuous_control_v12_smoke \
  --run-dir /home/jiamo/VLM_runs/continuous_control_v12_smoke/model \
  --device cpu
```

Run oracle and learned MPC:

```bash
$PY -m continuous_control_v12.run_mpc \
  --smoke --mode oracle \
  --data-dir /home/jiamo/VLM_data/continuous_control_v12_smoke \
  --max-steps 3 \
  --output /home/jiamo/VLM_runs/continuous_control_v12_smoke/oracle_mpc.json

$PY -m continuous_control_v12.run_mpc \
  --smoke --mode learned \
  --data-dir /home/jiamo/VLM_data/continuous_control_v12_smoke \
  --checkpoint /home/jiamo/VLM_runs/continuous_control_v12_smoke/model/continuous_forward_v12_smoke.pt \
  --max-steps 3 \
  --output /home/jiamo/VLM_runs/continuous_control_v12_smoke/learned_mpc.json
```

Evaluate the legacy 81-action grid:

```bash
$PY -m continuous_control_v12.evaluate_legacy_grid \
  --data-dir /home/jiamo/VLM_data/continuous_control_v12_smoke \
  --checkpoint /home/jiamo/VLM_runs/continuous_control_v12_smoke/model/continuous_forward_v12_smoke.pt \
  --split development --max-groups 1 \
  --output /home/jiamo/VLM_runs/continuous_control_v12_smoke/legacy_grid.json
```

Run the V11 generator and required baseline diagnostic:

```bash
$PY -m physics_structured_rebuild_v11.generate_system_aligned \
  --smoke \
  --output-dir /home/jiamo/VLM_data/physics_structured_rebuild_v11_smoke/system_aligned

$PY -m physics_structured_rebuild_v11.train \
  --overfit-groups 32 --ablation baseline --device cuda \
  --training-source /home/jiamo/VLM_data/physics_structured_rebuild_v11_smoke/system_aligned/grids/train.jsonl \
  --run-dir /home/jiamo/VLM_runs/physics_structured_rebuild_v11_overfit
```

Produce the first gate-eligible v11 learning-curve point:

```bash
$PY -m physics_structured_rebuild_v11.generate_system_aligned \
  --max-train-groups 326 --max-development-groups 600 \
  --output-dir /home/jiamo/VLM_data/physics_structured_rebuild_v11_326/system_aligned

$PY -m physics_structured_rebuild_v11.train \
  --group-count 326 --development-groups 600 \
  --ablation baseline --device cuda \
  --training-source /home/jiamo/VLM_data/physics_structured_rebuild_v11_326/system_aligned/grids/train.jsonl \
  --development-source /home/jiamo/VLM_data/physics_structured_rebuild_v11_326/system_aligned/grids/development.jsonl \
  --run-dir /home/jiamo/VLM_runs/physics_structured_rebuild_v11
```

Do not launch ordinary-loss, opaque-action, or legacy-distribution
interpretation runs until the common baseline gate passes. Do not launch the
10,500-group curve until the 326/678 scaling behavior and storage/compute
budget have been reviewed.
