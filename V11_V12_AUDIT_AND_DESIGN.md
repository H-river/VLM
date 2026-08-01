# V11/V12 optical-control audit and design

Date: 2026-07-29

This document was written before the v11/v12 implementation. The audit did
not open any v10 JSONL, locked-test data, checkpoint, or protected model
artifact. No v9/v10 file is modified by the new pipelines.

## Repository contract

- There is no repository-local `AGENTS.md`.
- The canonical numerical field definitions are in
  `specialist_rebuild_v2/common.py`.
- The simulator is `optical_sim/src/simulator.py`, with dataclasses in
  `optical_sim/src/optical_elements.py`, moment metrics in
  `optical_sim/src/metrics.py`, and the established adapter in
  `optics_sft/physics/sim_adapter.py`.
- The existing 81-action optimized simulator path is
  `control_rebuild_v5/simulator_grid.py`.
- Frozen v9 remains the reference described by
  `physics_structured_rebuild_v9/CYCLE_RESULTS_2026-07-28.md`.
- V10 remains the immutable negative pilot described by
  `physics_structured_rebuild_v10/V10_CYCLE_RESULTS.md` and
  `physics_structured_rebuild_v10/V10_IDENTICAL_NATURAL_AUDIT.md`. Those
  reports show that A-D underfit and lacked v9 data/capacity/training parity.
  V11 therefore does not treat the v10 conceptual ablations as settled.

## Exact deployed numerical contract

The 12 setup fields, in order, are:

1. `wavelength_nm`
2. `beam_waist_mm`
3. `power_w`
4. `lens_focal_length_mm`
5. `lens_aperture_mm`
6. `source_to_lens_mm`
7. `lens_to_camera_mm`
8. `lens_x_offset_mm`
9. `lens_y_offset_mm`
10. `camera_x_offset_mm`
11. `camera_y_offset_mm`
12. `pixel_size_um`

The five outputs, in exact order, are:

1. `centroid_x_px`
2. `centroid_y_px`
3. `sigma_x_px`
4. `sigma_y_px`
5. `peak_intensity`

Forward/direction tolerances are 1 px, 1 px, 2 px, 2 px, and 5% of the
absolute current peak (with the existing numerical floor). These are the v9
forward tolerances, not the stricter inverse-matching contract. The inverse
contract is a 0.5 px centroid-vector distance, 1 px for each width, and 2%
relative peak error.

The legacy action order is Python Cartesian-product order over:

```text
(-0.05, 0, +0.05) lens x
(-0.05, 0, +0.05) lens y
(-0.02, 0, +0.02) camera x
(-0.02, 0, +0.02) camera y
```

It contains 81 unique actions and the all-zero action is index 40.

## Units and coordinates

- Repository datasets store setup offsets and actions in mm.
- `OpticalSetup` stores lengths in metres. The adapter applies the explicit
  conversion `mm * 1e-3 -> m`.
- Wavelength is stored in dataset metadata as nm; pixel pitch as micrometres.
- The five legacy control metrics use the historical base/lab-frame
  pseudo-pixel centroid convention. They are not sensor-array coordinates
  when the camera offset is nonzero. V12 keeps that convention for benchmark
  parity and records it in every manifest.
- Sensor-frame image references are linked separately. Image conditioning is
  optional and is not required by the first v12 numerical model.

## Simulator audit

Variables that can affect the current simulator output are wavelength, beam
waist, focal length, clear aperture, source-to-lens distance,
lens-to-camera/effective camera distance, lens x/y position, camera x/y
position, pixel pitch, sensor resolution, grid size/extent, propagation
backend, and alignment defocus. V12 either records these or fixes them in the
generator configuration.

Important limitations:

- `gaussian_source_field()` normalizes the source peak and does not use
  `source.power`; varying `power_w` currently does not change the simulated
  intensity. V12 records it for contract parity but does not claim it is
  causal in this simulator version.
- Sensor extraction uses nearest-neighbour lookup and is therefore
  piecewise/discontinuous with sub-grid camera motion.
- The simulator returns an intensity image and a pre-crop complex camera
  field. Intensity images are available. Phase is not an observable deployed
  input. Existing v9 diagnostics define phase descriptors, but V12 keeps phase
  optional and null by default.
- Existing v10 generator code defines captured power as sensor intensity sum
  times pixel area, clipping fraction as one minus lens-plane transmitted
  intensity fraction, and camera-boundary distance as the minimum centroid
  distance to a sensor edge. V12 uses these existing definitions when the
  simulator backend can supply the required arrays.
- There is no simulator or hardware actuator-limit enforcement. The only
  repository-wide absolute position domain is the `[-0.003, +0.003] m`
  range for lens x/y and camera x/y in
  `profile2setup/configs/variables.yaml` and
  `optical_sim/configs/random_config_v2.yaml`. V12 converts this configured
  dataset domain to `[-3, +3] mm`, labels it
  `repository_sampling_domain_not_hardware_limit`, and requires it to remain
  configurable. It must not be presented as a measured hardware limit.
- Source phase, source shape, source alignment, sensor tilt, and other hidden
  state are fixed by the default generator. V9 diagnostics show that if such
  omitted state is allowed to vary, an intensity-only current observation can
  be non-identifying.

## V9 feature and model audit

`specialist_rebuild_v2.common.forward_feature()` constructs 46 values:

- 12 setup values;
- 5 current-state values (peak is `log1p` transformed);
- 4 physical action values;
- 25 engineered quantities: next absolute offsets, offset-square changes,
  radial offsets, lens/camera-to-pixel response scales, position-action
  products, and absolute-offset changes.

The frozen forward lineage uses a strong neural prior and v9 residual
corrections/selectors. V9 residual training concatenates the engineered
features and prior predictions, learns residual targets, and protects
selection/calibration on independent group partitions. The reported neural
reference has approximately 0.98 million parameters and a zero-action anchor.
V11 therefore uses:

- the exact 46-feature preprocessing for structured-action runs;
- a matched-capacity residual MLP;
- deterministic group-fold out-of-fold Ridge prior predictions for training;
- a prior refit on all fit groups for development/inference;
- a zero-action consistency anchor;
- fit and development diagnostics before any conceptual interpretation.

The opaque-action ablation replaces physical action inputs with a learned
action-ID embedding but retains the same context, hidden width, residual
objective, optimizer steps, early stopping, and seed.

## Grouping, splits, and hashes

Existing grids contain one independent setup/current context per JSONL row and
81 correlated candidates within that row. V9 and v10 split/calibrate using
stable hashes of group IDs. V10 additionally records rounded setup and
setup-plus-current SHA-256 hashes and rejects cross-split overlap.

V11/V12 use group IDs as the only split/bootstrap unit. V12 records:

- deterministic group-ID assignment;
- setup and context SHA-256 hashes;
- per-split sorted group-ID hashes;
- cross-split overlap checks;
- generator seed/version and repository commit;
- transition counts as correlated observations, never as independent groups.

## V11 design and interpretation gate

V11 preserves the 81-action contract and exposes learning-curve sizes
326, 678, 1,200, 2,400, 5,000, and 10,500 groups. A deterministic 32-64 group
overfit mode is a required pipeline diagnostic. Ablations are matched on
groups, feature availability, parameter budget, optimizer steps, early
stopping, and seeds.

The v11 report marks ablations `not_interpretable` unless the common baseline
approaches the configured frozen-v9 fit and development references on both
normalized MAE and strict all-five accuracy. Smoke tests never pass or claim
that scientific gate.

## V12 design

V12 stores transitions `(s_t, a_t, s_{t+1})` with continuous actions in mm.
The default per-step limits are ±0.05 mm for each lens axis and ±0.02 mm for
each camera axis. Absolute positions are checked against configured limits.

Targets are generated from legal target actuator positions `q*` and simulator
runs. `q*` is stored only under oracle/supervision metadata and schema guards
explicitly reject it from deployable model state, controller observations, and
structured goals.

The first world model is numerical and predicts tolerance-normalized metric
deltas around the current metrics. It supports physical action squares,
pairwise action products, and position-action products. Image embeddings and
auxiliary losses are clean optional extensions.

The controller is receding-horizon CEM. It enforces step and absolute bounds,
returns best effort when no predicted candidate reaches tolerance, and has a
simulator-oracle mode to isolate planner behavior from learned-model error.

