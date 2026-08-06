# Overnight v12 simulator-semantics and forward-model report

Run: `overnight_v12_semantics_20260731_002709`
Starting commit: `aa7e1f66cfcb497cd5c5ff39c394520668f75584`
Schema: `v12.1.0`
Corrected simulator semantics: `v12_sensor_power_semantics_v1`
Immutable data-quality v2 generation-config SHA-256:
`77cb8bfc8cc064e32ab3cbf473e10f94df5c4ca53869ae235bb5eee782b81ff8`

This is the consolidated report for the bounded overnight run.

## Executive result

The corrected simulator path passes its expansion gate. It is opt-in,
numerically continuous in camera sampling, power-causal, explicit about
coordinate frames, converged against higher sampling and propagation
references, and exactly preserves the legacy default path.

The corrected local continuous dynamics are materially learnable. On fixed
nested 16/32/64/128-group prefixes, development normalized MAE decreases
monotonically from `0.6714` to `0.2362`; held-out test MAE decreases from
`0.7124` to `0.1579`; and 128-group strict all-five accuracy is `89.95%`.
The 128-group model also has `0.8053` paired direction-sign accuracy and
`0.03892` median relative directional-Jacobian error.

The model is not yet cleared for substantial learned MPC. Its horizon-3
rollout MAE is `0.4610`, or `2.837x` horizon-1 MAE `0.1625`, failing the
pre-registered `<=2x` accumulation gate. Five other gate checks pass. Learned
MPC was therefore skipped without post-result planner tuning. Oracle CEM
strongly improved all three fixed targets but reached strict tolerance on only
one, so the simulator is reachable and controllable by construction while
robust optimizer success remains unproven.

## Scope and safety

- No protected raw v10 locked-test data, frozen v9/v10 checkpoint, or
  preserved v9/v10 result was opened, copied, regenerated, or modified.
- No v11/v12 output was overwritten. Every generated result is under this
  versioned run directory or a fresh audit rerun directory.
- The run generates 128 train, 16 development, and 16 test optical groups,
  not the prohibited 10,500-group dataset.
- Train/development/test group IDs and setup hashes are disjoint.
- `q_star_mm` remains supervision-only `oracle_metadata`; the deployable input
  constructor and runtime guard reject privileged fields.
- The initial dirty worktree was preserved. Unrelated user changes were not
  edited, staged, committed, pushed, or published.

## What was implemented

### Opt-in sensor measurement

The legacy `optical_sim.run_simulation()` default still uses its unchanged
clipped left-`searchsorted` lookup. The corrected path must be selected
explicitly and uses:

1. sensor pixel centres at
   `camera_pose + (index - (count-1)/2) * declared_pitch`;
2. continuous fractional simulation-grid coordinates;
3. order-3 tensor Gauss-Legendre finite-pixel irradiance averaging;
4. bilinear interpolation of `abs(E)^2`, never wrapped phase;
5. explicit zero padding outside the propagated field;
6. a complete-pixel valid-region mask;
7. serialized grid pitch, sensor pitch, axis direction, origin, quadrature
   order, sampling method, and semantics version.

Finite-pixel irradiance integration was selected over the prompt’s default
complex-field point interpolation. On the original 1024 / +/-30 mm candidate,
real/imaginary point interpolation retained only 44.98% of the finite-pixel
reference power. On the final 1536 / +/-6.25 mm grid it retained 94.65%, but
still differed by 0.594 task tolerance. The selected order-3 area result
differed from order 9 by only 0.000108 tolerance.

### Active optical power

Only the corrected path normalizes the source field so

```text
sum(abs(E_source)^2) * simulation_grid_pitch^2 == power_w
```

`power_w` is integrated source optical power in W. `image_raw` and the
five-metric `peak_intensity` are absolute irradiance in W/m2;
`captured_power_w = sum(image_raw) * sensor_pixel_pitch^2`; and
`image_normalized = image_raw / max(image_raw)` is exposed separately.

Measured ratios relative to 1 W were exact to displayed precision:

| power | amplitude | raw sum | captured power | absolute peak |
|---:|---:|---:|---:|---:|
| 0.25 | 0.5 | 0.25 | 0.25 | 0.25 |
| 0.5 | 0.70710678 | 0.5 | 0.5 | 0.5 |
| 1 | 1 | 1 | 1 | 1 |
| 2 | 1.41421356 | 2 | 2 | 2 |
| 4 | 2 | 4 | 4 | 4 |

Centroids and widths were invariant, and the maximum normalized-image
difference was `5.55e-17`.

### Explicit coordinate semantics

The compatible five numerical targets remain lab-frame pseudo-pixel metrics.
Images are sensor-frame arrays. Every v12.1 row records both forms, camera
pose, pitch, first-pixel origin, axis directions, control frame, and:

```text
centroid_sensor_px = centroid_lab_px - camera_pose_m / pixel_pitch_m
sigma_sensor_px = sigma_lab_px
```

Independent metric recomputation from the stored float32 sensor image has a
maximum discrepancy of approximately `3.6e-15` tolerance in the production
audit, well below the requested 0.1 threshold. Camera pose is already present
in numerical model inputs, so the frame pairing is identifiable. The optional
image-conditioned ablation was not needed for tonight’s primary numerical
model and medium images were not stored.

### Data, model, and MPC diagnostics

- v12.1 rows record requested and effective actions and positions separately.
- Generation is group-checkpointed, config-hash locked, and resumable.
- Dataset analysis covers bounds, distributions, regimes, duplicates,
  constants, power variation, finite values, split leakage, and privileged
  fields.
- Forward training supports deterministic nested 16/32/64/128 train-group
  prefixes with fixed development/test groups, train-only statistics, and
  identical architecture/budget/seeds.
- Diagnostics report one-step and per-output errors, tolerance accuracy,
  action magnitude/axis count, regimes, paired directions, local directional
  Jacobian error, uncertainty, and autoregressive horizons 1/3/5.
- MPC traces now cross-evaluate every selected action in the corrected
  simulator and record predicted/actual improvement, distance gap, illegal
  actions, uncertainty, and explicit planner-exploitation events.
- Oracle and learned CEM use the same deterministic planner seed for a matched
  target.
- Discrete-81 target reachability uses the corrected simulator for corrected
  targets; the cached legacy grid remains available only to legacy semantics.

## Simulator quantitative acceptance

Final configuration: 1536 x 1536 simulation grid over +/-6.25 mm, 1024 x 1024
sensor, 1.5 um continuity steps.

| Check | Result | Gate |
|---|---:|---:|
| Order-3 pixel integration vs order 9 | 0.000108 tolerance max | <0.05 |
| Configured 1536 vs 2048 base state | 0.001116 tolerance max | <0.25 |
| Worst of 12 multi-regime 1536-vs-2048 cases | 0.067700 tolerance | <0.25 |
| Worst multi-regime captured-power disagreement | 0.006067% | <1% |
| Corrected interior camera x 1.5 um max | 0.001326 tolerance | <0.25 |
| Corrected interior camera y 1.5 um max | 0.001334 tolerance | <0.25 |
| Corrected clipping-boundary camera x max | 0.165810 tolerance | <1 |
| Corrected image plateau transitions | 0 | 0 |
| Corrected metric plateau transitions | 0 | 0 |
| Corrected camera effective-coordinate plateaus | 0 | 0 |
| Power ratios and invariances | pass | pass |
| Coordinate transform/recomputation | pass | <0.1 |
| Finite and deterministic | pass | pass |

The corrected 1.5 um camera maximum is about 628 times smaller than the old
0.838-tolerance maximum. The corrected boundary result is about 10.2 times
smaller than the old 1.695-tolerance maximum. Corrected lens translations
still reach about 0.409 tolerance per 1.5 um; both the selected sampler and
the higher-resolution reference show this response, so it is physical
thin-lens magnification rather than a sensor-index jump.

The longest corrected image and metric plateau is one point for every sweep.
The longest corrected camera-coordinate plateau is also one point. Lens
motion correctly leaves the sensor sampling coordinates stationary (121
points) while the propagated image and metrics change at every step. No
corrected sweep reports a derivative spike under the audit’s 10x-median rule.
Some nearly invariant cross-axis and peak traces have many derivative sign
changes (the largest count is 148) because the remaining continuous
piecewise-bilinear ripple alternates at simulation-grid scale. Its largest
camera-step effect is only 0.001334 tolerance. The physically responsive lens
centroid axes and the clipping-boundary centroid have zero derivative sign
changes.

The exact legacy audit rerun reproduced all nine CSVs and all eight plots
byte-for-byte. The deterministic 128-grid source, lens-plane, post-lens,
camera-plane, and sensor-image hashes also match their frozen baselines.

## Rejected simulator candidates

The initial 1024 / +/-30 mm candidate passed its local continuity checks but
differed from a 2048 reference by 20.26 task tolerances. The root cause was
thin-lens phase under-resolution at the legacy 58.65 um grid pitch. A 1024 /
+/-8 mm candidate improved the base state but exposed additional deterministic
multi-regime outliers. Neither rejected configuration was used for medium
data. The accepted 1536 / +/-6.25 mm configuration was chosen only after all
twelve multi-regime checks passed.

## Corrected dataset

### Smoke

The current-config smoke contains 4 train, 2 development, and 2 test groups
(152 transitions). It is finite, within action/position bounds, has no
cross-split group or setup overlap, no duplicate deployable inputs, no
constant setup fields, active power variation, all required sampling kinds,
and no privileged data in deployed inputs.

Two independent full builds reproduced all three transition JSONLs and all
160 sensor NPZ archives byte-for-byte. An intentional timeout after one
completed group was also resumed successfully and produced the same
byte-identical result.

After the main generation process started, resume bookkeeping was hardened
for the narrow crash window between writing a complete group file and updating
`generation_state.json`. This bookkeeping-only edit does not change group
content; `configs/source_sha256_at_generation.txt` preserves the exact v1
generator source that process loaded. The replacement v2 process is frozen by
`configs/source_sha256_at_v2_generation.txt`.

### Medium

The first v1 attempt was stopped after 64 train groups. Incremental analysis
found a rare near-dark camera-boundary setup, `v12_train_000055`, whose initial
captured/source-power fraction was `8.23e-10`. Although finite, its
ill-conditioned centroids produced deltas as large as 409.38 task tolerances.
The incomplete v1 directory is preserved as rejected data-quality evidence
and will not be resumed.

Data-quality v2 keeps the accepted simulator semantics unchanged but
deterministically resamples a complete setup when its initial
captured/source-power fraction is below 0.01, up to 16 attempts. The retry
index and accepted fraction are serialized. A v1 smoke build after this
change remained byte-identical to its earlier reference. The v2 smoke
exercised retry index 2, accepted a minimum fraction of 0.01717, and reproduced
all three JSONLs and all 160 NPZ tensors byte-for-byte on replay.
Strict negative checks reject a fraction of 0.009, retry index 16, and missing
policy metadata, while the v1 schema remains policy-inactive. Across all 152
smoke transitions, the minimum post-action captured/source-power fraction was
0.01016; none fell below 0.001 or the configured initial floor. A direct
production-grid v1/v2 state comparison produced bit-identical raw images,
normalized images, and valid masks plus equal metrics and auxiliaries,
confirming that v2 changes setup selection rather than optical semantics.

The 128/16/16 v2 grouped dataset completed in 1 h 38 min with 160 groups and
11,040 transitions: 8,832 train, 1,104 development, and 1,104 test. Each
group contains 69 transitions from 64 continuous probes and one five-step
trajectory. The immutable split JSONL hashes are:

- train: `80bb0862ba6f7862e68163b5f8372b2c94cea93f87c1f822d1e66776a529006f`;
- development: `e8d55fbc53fb2654aed6dab0f8b234f291cbc5516c1a31fecf4c5df78d826205`;
- test: `ebc26c9ecde7f8678878ebc57e795d15fc9e23c6cdb65699b2768bcce6508411`.

All rows are finite and within action/position bounds. Train, development,
and test have zero group or setup-hash overlap. All six planned regimes are
present: 30 camera-boundary, 24 clipping, 30 focusing, 30 high-offset, 26
ordinary, and 20 tolerance-boundary groups. Power has 160 unique values.
There are no privileged `q_star` fields in deployable inputs. Sixty-two
repeated deployable inputs occur across 51 keys, primarily terminal/no-op
structure; none has conflicting targets, and there are no identical full
content rows after IDs are ignored.

One of 160 setups required deterministic retry index 1. The minimum accepted
initial captured/source-power fraction is `0.15850`, well above the `0.01`
floor. The minimum across all post-action observations is `0.02853`; no
transition falls below `0.01`, `0.001`, or `1e-6`. The rejected v1 near-dark
pathology did not recur.

## Forward-model evidence

### Tiny overfit

The current smoke model nearly fit 32 deterministic transitions:

- best epoch: 137;
- normalized MAE: 0.0864;
- strict all-five accuracy: 96.875%;
- per-output MAE: centroid x 0.1187, centroid y 0.1138, sigma x 0.0405,
  sigma y 0.0425, peak 0.1165;
- exact no-op consistency: pass;
- saved/reloaded inference: finite.

An independent replay produced bitwise-identical model state tensors and an
identical report after excluding the output path and wall-clock duration.

A separate three-member v2 smoke ensemble also reproduced every state tensor
exactly. Its four-group test result is plumbing evidence, not the final
learnability claim: one-step normalized MAE was 1.391, paired direction-sign
accuracy was 0.65, median relative directional-Jacobian error was 0.818, and
horizon-3 rollout MAE was 5.346 versus 1.904 at horizon 1. This shows the full
gate is meaningful while confirming that four groups do not control rollout
accumulation.

### Production overfit

The production-grid capacity check fit the first 32 transitions from one
production group. Best epoch was 268, normalized MAE was `0.04133`, and strict
all-five accuracy was `100%`. Per-output normalized MAE was `0.09544`
centroid x, `0.04801` centroid y, `0.02262` sigma x, `0.02510` sigma y, and
`0.01547` peak. Predictions remained finite after serialization and no-op
residuals were exactly zero. This clears the capacity gate.

### Fixed nested learning curve

Architecture, optimizer budget, seeds, development groups, and test groups
were held fixed; only the nested training prefix changed.

| train groups | development MAE | development strict | test MAE | test strict |
|---:|---:|---:|---:|---:|
| 16 | 0.6714 | 35.24% | 0.7124 | 27.72% |
| 32 | 0.4800 | 43.75% | 0.3586 | 53.35% |
| 64 | 0.3284 | 70.11% | 0.2640 | 76.00% |
| 128 | 0.2362 | 83.88% | 0.1579 | 89.95% |

This is a strong, monotonic data-scaling result with no sign of a 128-group
plateau. The 128-group held-out per-output MAEs are `0.2436` centroid x,
`0.3740` centroid y, `0.04064` sigma x, `0.05673` sigma y, and `0.07444`
peak. Centroid y is the largest output bottleneck. By regime, clipping is
hardest (`0.3450` MAE, `55.80%` strict), followed by tolerance-boundary
(`0.2300`, `78.26%`). Ordinary is `0.1027` with `100%` strict accuracy.

Paired-action direction-sign accuracy stays well above chance and is `0.8053`
at 128 groups. Its per-output signs are strongest for centroid x/y
(`0.9896` each); peak sign is only `0.4815`, although absolute peak error is
small. Median relative directional-Jacobian error improves monotonically
`0.2296 -> 0.1164 -> 0.07715 -> 0.03892`, showing that local response is
learned accurately.

Autoregressive error also improves strongly with data, but accumulates:

| train groups | horizon 1 | horizon 3 | horizon 5 |
|---:|---:|---:|---:|
| 16 | 0.7359 | 2.0718 | 3.3999 |
| 32 | 0.3691 | 1.0423 | 1.6285 |
| 64 | 0.2439 | 0.6407 | 1.0123 |
| 128 | 0.1625 | 0.4610 | 0.7021 |

No focused longer-training retry was run: overfit and local-Jacobian evidence
already pass, whereas the remaining failure is multi-step compounding. A
same-objective epoch extension would not be a well-motivated use of the
single allowed retry.

## Conditional MPC evidence

The matched oracle/learned smoke exercise only validated plumbing; it is not
a scientific controller result because the tiny model was fit to a different
subset. It confirmed identical seeds/settings, real-simulator cross
evaluation, zero illegal actions, and correct aggregation.

Substantial learned MPC is pre-registered to require:

- production overfit strict accuracy at least 95%;
- finite predictions;
- 128-group development MAE below 16-group development MAE;
- paired direction-sign accuracy at least 0.65;
- median relative directional-Jacobian error below 1.0;
- finite horizon-3 rollout MAE no more than 2x horizon-1.

The formal gate passes production overfit, finiteness, learning-curve
improvement, paired directions, and local Jacobians. It fails only rollout
accumulation: `0.46097 > 2 * 0.16249`. The saved decision has
`learned_mpc_authorized=false`; substantial learned MPC was not run.

Twelve fixed held-out target records were generated from groups 3, 5, 8, and
14, covering one-step reachable, multi-step reachable, and ambiguous
mismatched categories. Mismatched candidates were not labeled infeasible
because repeated continuous-oracle verification was intentionally skipped
after the learned gate failed.

Three fixed oracle-only episodes used horizon 2, population 32, 8 elites,
three CEM iterations, and at most three executed steps:

| target | regime | initial cost | final cost | improvement | strict success |
|---|---|---:|---:|---:|---:|
| group 5 one-step | camera boundary | 5.606 | 0.845 | 84.9% | yes |
| group 8 one-step | ordinary | 4.595 | 1.162 | 74.7% | no |
| group 14 multi-step | clipping | 38.390 | 4.232 | 89.0% | no |

Every episode had zero illegal actions and zero exploitation events. The
targets have exact simulator construction witnesses; the clipping target also
has a three-step bounded-position lower bound. Thus one-of-three strict
optimizer success does not make the other targets physically infeasible. It
does show that this small CEM budget is not a robust oracle controller,
especially at clipping boundaries.

## Tests

Current suite: 36 passed, 3 xfailed. The retained legacy diagnostics cover
discontinuous sampling, dead power, and the intentionally false assertion
that a lab-frame centroid should directly equal a sensor-array centroid;
passing corrected-semantics equivalents cover deterministic replay, camera/lens
subpixel sweeps, signs, zero padding/mask, image metric recomputation, power
ratios/invariances, peak provenance, quadrature reference, and explicit
semantics.

## Failures, skipped work, and limitations

- The first pytest attempt was blocked by an auto-loaded system ROS Jazzy
  plugin from a different Python environment. Disabling third-party pytest
  auto-load fixed collection; no source workaround was introduced.
- The first metric regression exposed pre-serialization metric computation.
  Metrics now derive from the exact stored float32 image; the failing log was
  preserved.
- Two simulator-grid candidates failed convergence and were preserved.
- Point complex-field interpolation was rejected on measurement-model and
  quantitative-reference evidence.
- A smoke-only MPC plumbing review found and fixed a mixed-semantics
  reachability bug: corrected targets had initially been compared with the
  cached legacy 81-action simulator. Reachable target content was unaffected,
  the invalid candidate metadata was superseded before production target
  generation, and a corrected-semantics regression now covers dispatch.
- The optional v11 parity run was skipped so it could not consume resources
  needed by the primary v12 task.
- Full candidate-infeasible verification is expensive and will only be run on
  a small fixed diagnostic set after a future forward model clears the rollout
  gate. The four current mismatched targets remain explicitly ambiguous.
- The initial medium v1 dataset failed a newly added signal-quality audit
  because one of 64 generated train groups was effectively dark. It was
  stopped and preserved; the deterministic v2 retry policy addresses that
  failure without weakening or silently filtering individual transitions.
- The 128-group model fails the pre-registered rollout ratio despite strong
  one-step and local-Jacobian results. Production learned MPC was skipped.
- Oracle CEM used a deliberately bounded budget and reached strict tolerance
  on only one of three construction-reachable targets. This is not a claim of
  robust controller performance.

## Go/no-go decisions

1. Corrected simulator ready for medium/full v12 generation: **GO for the
   versioned v2 semantics and signal-floor policy.** The prohibited
   10,500-group build was not run; any future full build should retain the
   same frozen audit/config/hash checks.
2. Forward model ready for larger data scaling: **CONDITIONAL GO.** More
   groups are justified by the monotonic curve, but scaling should be paired
   with rollout-aware training/modeling rather than treated as the only fix.
3. Learned MPC evaluation meaningful: **NO for substantial production
   evaluation under the current checkpoint.** The pre-registered rollout
   gate failed; learned MPC was correctly skipped.
4. Next priority: **multi-step/rollout-aware modeling and clipping/tolerance
   boundary coverage, with centroid-y error as the primary output
   bottleneck.** Re-run the same preregistered gate before planner tuning.
5. v11 326 groups sufficient for 678: **NO DECISION; v11 was not executed.**

## Reproducibility and artifacts

- Exact commands: `commands/exact_commands.md`
- Generation-time causal source hashes:
  `configs/source_sha256_at_v2_generation.txt`
- Incremental status: `overnight_status.md`
- Accepted simulator audit:
  `audits/semantics_production_1536_extent6p25_v2/`
- Legacy byte comparison: `audits/legacy_audit_rerun_comparison.txt`
- Current v2 smoke analysis: `audits/dataset_smoke_8g_v2/`
- Full v2 dataset analysis: `audits/dataset_128_16_16_v2/`
- V2 smoke determinism: `audits/data_quality_v2_smoke_determinism.txt`
- Resume determinism: `audits/resume_smoke_test_comparison.txt`
- Learning curve and gate: `reports/learning_curve_v2/`
- Fixed production targets: `audits/mpc_targets_v2/`
- Oracle episodes and aggregate: `audits/mpc_oracle_v2/` and
  `reports/oracle_mpc_v2.json`
- Final causal source hashes: `configs/source_sha256_final.txt`
- Key artifact hashes: `reports/key_artifact_sha256.txt`
- Stdout/stderr inventory: `reports/stdout_stderr_inventory.tsv`
- Complete created-file inventory: `reports/files_created_inventory.txt`
- Final running-job snapshot: `reports/job_status_final.txt`
- Current git status/diff stat: `reports/git_diff_stat.txt`
- Run-attributed source inventory and concise diff summary:
  `reports/source_files_created_changed.md`
- Versioned semantics documentation:
  `continuous_control_v12/SIMULATOR_SEMANTICS_V12.md`
- Source/data-flow trace: `continuous_control_v12/SOURCE_DATA_FLOW_V12.md`

The final stdout/stderr inventory, files-created inventory, source inventory,
source hashes, job-state snapshot, and git diff/status summary are stored
under `reports/` and `configs/`.
