# Simulator three-issue audit

Date: 2026-07-30  
Version: `simulator_three_issue_audit_v1`  
Scope: diagnostic only; no production simulator default was changed.

## Executive verdicts

| Issue | Verdict | Evidence in one line | Recommendation |
|---|---|---|---|
| Camera sampling continuity | **Confirmed material discontinuity** | Current sampling is clipped left-`searchsorted` integer lookup on a 58.651 µm production grid; a 1.5 µm camera step caused up to 0.838 tolerance of centroid jump away from boundaries, 1.695 at clipping, and 6.938 peak tolerances in the 128-grid smoke configuration | **fix before medium/full v12 generation, but current v11 work may continue** |
| Lab versus sensor frame | **Deliberate dual representation but insufficiently documented at the schema/task boundary**; numerically consistent after an explicit transform | Stored centroid is lab-frame pseudo-pixel while image centroid is sensor-frame; after subtracting camera translation and correcting endpoint pitch, discrepancies are at most 0.060 tolerance and peak is exact | **documentation/schema clarification only** |
| `power_w` | **Dead/ignored input** | 0.25x, 0.5x, 1x, 2x, and 4x produced bitwise-identical source, propagated field, sensor image, five metrics, and captured power; all output ratios were exactly 1 | **fix before medium/full v12 generation, but current v11 work may continue** |

The three implementation-report hypotheses are therefore directionally
correct, but two need more exact wording:

1. The camera sampler is not true nearest-distance interpolation. It uses
   `np.searchsorted(..., side="left")`, choosing the first grid coordinate
   greater than or equal to each sensor coordinate.
2. The five metrics are not computed upstream on the unsampled lab field.
   They are computed **after** the sensor lookup from the same sampled
   intensity array, but on lab-coordinate arrays and then serialized in
   legacy lab-frame pseudo-pixels.

## Scope, safety, and reproducibility

- No repository-local `AGENTS.md` exists.
- No raw v10 locked-test data, frozen-v9 checkpoint, or preserved v9/v10
  artifact was opened, copied, regenerated, or modified.
- No full dataset generation or training was launched.
- Source code, newly generated deterministic audit samples, and three
  explicitly non-protected training JSONLs were used.
- The production experiment used the current 1024 grid and sensor. The
  separate 128-grid run matches the v11/v12 smoke resolution.
- The simulator contains no RNG in this path. Repeated identical inputs are
  bitwise deterministic; `stable_seed` checks are included for the audit
  harness.

The exact source trace is in
[`source_data_flow_trace.md`](source_data_flow_trace.md), and execution
commands are in [`commands_used.txt`](commands_used.txt).

## Fixed deterministic setup

| Parameter | Value |
|---|---:|
| wavelength | 632.8 nm |
| beam waist | 1.0 mm |
| declared power | 1.0 W |
| lens focal length / aperture | 100 mm / 25 mm |
| source-to-lens / lens-to-camera | 200 mm / 150 mm |
| declared sensor pitch | 5.5 µm |
| initial lens x/y | +0.11 / -0.08 mm |
| initial camera x/y | +0.07 / -0.04 mm |
| main sweep | +/-0.09 mm, 1.5 µm increments, 121 points |

Resolution:

| Configuration | Simulation pitch | Declared sensor pitch | Constructed sensor-coordinate step | Unique simulation samples selected per sensor axis |
|---|---:|---:|---:|---:|
| Production 1024 | 58.651 µm | 5.500 µm | 5.505376 µm | 97-98 of 1024 columns/rows |
| Smoke 128 | 472.441 µm | 5.500 µm | 5.543307 µm | only 2-3 of 128 columns/rows |

The smoke sensor is therefore formed mostly by duplicating two or three
simulation-grid values along each axis. This is not just a smaller version of
the production simulation; it is qualitatively under-resolved.

# Audit 1: camera sampling continuity

## Static answer

1. The field is propagated on the fixed lab grid. Camera displacement is
   applied only after propagation by translating the sensor coordinate arrays.
2. Each sensor coordinate is mapped by clipped left-`searchsorted` to an
   integer simulation-grid index. There is no sensor-pixel integration,
   nearest-distance rounding, bilinear interpolation, or Fourier shift.
3. The quantization pitch for each individual sensor sample is the simulation
   pitch: 58.651 µm in production and 472.441 µm in smoke. The sensor pitch
   only spaces requested coordinates.
4. Metrics are computed after the integer lookup, from the sampled intensity.
5. Images, captured power, centroid, width, and peak can all jump. Some
   quantities can plateau even while other image pixels change.
6. V12 actuator positions remain continuous floating-point values. The
   discontinuities are numerical sampling artifacts, except the separate,
   explicit plateau at the configured +/-3 mm position projection and real
   finite-sensor clipping behavior.

## Main camera-sweep evidence

Largest adjacent metric changes are shown in units of each field's task
tolerance. Each displacement is 1.5 µm.

| Configuration / sweep / sampler | centroid x | centroid y | sigma x | sigma y | peak |
|---|---:|---:|---:|---:|---:|
| Production camera x, current | **0.6364** | 0 | **0.2686** | ~0 | 0 |
| Production camera x, bilinear intensity | 0.0114 | ~0 | 0.00178 | ~0 | 0.1346 |
| Production camera y, current | 0 | **0.8377** | ~0 | **0.2136** | 0 |
| Production camera y, bilinear intensity | 0 | 0.0105 | ~0 | 0.00092 | 0.2069 |
| Production clipping-boundary x, current | **1.6948** | 0 | 0.3534 | ~0 | 0 |
| Production clipping-boundary x, bilinear intensity | 0.8103 | ~0 | 0.1066 | ~0 | 0.1563 |
| Smoke camera x, current | 0.5454 | ~0 | 0.0690 | ~0 | **6.9384** |
| Smoke camera x, bilinear intensity | 0.2709 | ~0 | 0.00549 | ~0 | 0.0458 |
| Smoke camera y, current | ~0 | 0.2727 | ~0 | 0.1031 | 0 |
| Smoke camera y, bilinear intensity | 0 | 0.2581 | ~0 | 0.00428 | 0.0339 |

The production current camera response is strongly jagged in the plotted
centroid and width, even away from clipping. Across the full +/-0.09 mm sweep:

| Quantity expected nearly invariant in lab frame | Current x / y variation | Bilinear-intensity x / y variation |
|---|---:|---:|
| stored lab centroid range | 0.862 / 0.897 px | 0.043 / 0.035 px |
| captured-power relative range | 0.804% / 1.170% | 0.0134% / 0.0110% |

For a finite sensor containing the beam away from an edge, translating the
camera should move the beam in sensor coordinates but should not move the
beam's lab-frame centroid or materially change captured power. The comparison
therefore isolates the current sampler as the dominant source of these
variations.

## Plateaus, boundaries, and derivatives

- Production full-image exact plateaus: 0 of 120 adjacent transitions for
  both camera axes. Different sensor columns cross simulation-grid boundaries
  at staggered positions, so at least one pixel changes at every 1.5 µm step.
- Production centre-sensor sample: only 3 index changes over the 121 points;
  its longest constant-index run is 39 points, consistent with the 58.651 µm
  simulation pitch.
- Production current peak: constant for all 121 points in both main camera
  sweeps, despite centroid/width jumps.
- Smoke full-image exact plateaus: 67/120 transitions for camera x and 62/120
  for camera y; longest whole-image plateau is four points.
- Smoke camera-x peak: longest constant run is 91 points, followed by the
  6.938-tolerance jump visible in the plot.
- The separate v12 position-limit sweep has 21 requests projected to exactly
  +3.0 mm and 20 exact repeated-image transitions. That plateau is expected
  from the explicit bound and is not attributed to sensor sampling.

Finite-difference diagnostics use `(metric[i]-metric[i-1])/0.0015 mm`.
Ignoring zero derivatives, production current camera x has 64 centroid-x and
87 sigma-x derivative sign changes; camera y has 87 centroid-y and 94 sigma-y
sign changes. A spike was defined as more than 10 times the median non-zero
absolute derivative. No spikes met that threshold because the artifact is a
regular, dense jagged pattern rather than isolated outliers. Active lens-axis
derivatives had only six sign changes and no spikes.

## Lens sweeps

| Production sweep / sampler | active centroid jump [tol] | active sigma jump [tol] | peak jump [tol] |
|---|---:|---:|---:|
| lens x, current | 0.4210 | 0.1058 | 0.5405 |
| lens x, bilinear intensity | 0.4021 | 0.1075 | 0.5346 |
| lens y, current | 0.4202 | 0.1072 | 0.5263 |
| lens y, bilinear intensity | 0.4080 | 0.1072 | 0.4871 |

Lens movement changes the thin-lens phase continuously and can change hard
aperture membership on the simulation grid before propagation. The current
and bilinear sensor samplers give similar lens-sweep magnitudes, no output
plateaus on the active quantities, and no derivative spikes. These lens
changes are therefore not explained by the camera lookup. This audit does not
claim that bilinear sampling proves their physical correctness.

## Interpolation semantics and conservation

At the identical production no-op state:

| Audit method | sensor sum | captured-power value | relative to current |
|---|---:|---:|---:|
| current left-`searchsorted` | 51,854.872 | 1.568610e-6 | 1 |
| bilinear intensity | 51,808.964 | 1.567221e-6 | 0.999115 |
| bilinear complex field then square | 23,309.296 | 7.051062e-7 | 0.4495 |

Bilinear intensity approximately preserves the integrated sampled intensity
at this state (0.0885% difference) and greatly reduces centroid/width
jaggedness, but it gives a piecewise-linear peak that changes as the maximum
moves between cells. It is an irradiance-resampling approximation; it does
not model finite sensor-pixel area integration.

Bilinear complex-field interpolation loses about 55% of sampled power in this
spot check because real/imaginary interpolation averages a phase-varying,
under-resolved field before squaring. It is not an acceptable drop-in based on
this evidence.

Fourier shifting the complex field would represent a translation of the
optical field relative to a fixed grid, which is not identical to moving a
finite camera window. Final-image interpolation is a display operation and
cannot recover missing physical samples. A future fix should first define
pixel-area integration and ensure the propagated field/intensity grid is
sufficiently resolved; the smoke 128 grid remains inadequate even with
bilinear intensity.

## Audit 1 conclusion

**Classification: confirmed material discontinuity.**

The materiality is established by 0.838 tolerance jumps away from a boundary,
1.695 at finite-sensor clipping, and 6.938 peak tolerances at smoke
resolution. Production main-sweep jumps are smaller than one tolerance per
step but consume most of a centroid tolerance at a displacement only 0.273
sensor pixel and 0.026 simulation cell. The artifact is therefore not
negligible for a continuous controller.

Recommendation: **fix before medium/full v12 generation, but current v11 work
may continue**. V11's discrete benchmark remains interpretable as behavior of
the legacy simulator, but not as evidence of smooth subpixel camera physics.

# Audit 2: lab-frame versus sensor-frame representation

## Frame table

| Quantity | Current frame | Image-derived frame | Unit | Transform available | Consistent? |
|---|---|---|---|---|---|
| centroid x | fixed lab-frame pseudo-pixel | sensor-array pixel centres | px | subtract `camera_x_m/pitch_m` | yes after transform; residual endpoint scale |
| centroid y | fixed lab-frame pseudo-pixel | sensor-array pixel centres | px | subtract `camera_y_m/pitch_m` | yes after transform; residual endpoint scale |
| sigma x | lab-coordinate width divided by declared pitch | sensor-array pixel centres | px | scale by `(W-1)/W` | yes |
| sigma y | lab-coordinate width divided by declared pitch | sensor-array pixel centres | px | scale by `(H-1)/H` | yes |
| peak intensity | sampled sensor image | same sampled sensor image | simulator intensity unit | identity | exact |

The exact image-derived recomputation used documented physical pixel centres:

```text
x_j = (j-(W-1)/2) * pixel_pitch
y_i = (i-(H-1)/2) * pixel_pitch
```

After the camera translation transform:

| Quantity | Maximum discrepancy across no-op + four interventions | In task tolerances |
|---|---:|---:|
| centroid x | 0.02230 px = 0.123 µm | 0.02230 |
| centroid y | 0.00634 px = 0.0349 µm | 0.00634 |
| sigma x | 0.11961 px = 0.658 µm | 0.05981 |
| sigma y | 0.11880 px = 0.653 µm | 0.05940 |
| peak | 0 | 0 |

The width residual is the exact `W/(W-1)` endpoint-spacing effect, not a
separate metric source. Peak and clipping are from the same sampled image, so
finite-sensor cropping is reflected in both stored and image-derived values.

## Intervention signs and magnitudes

For a pure camera `+x` translation, a stationary lab beam should:

- remain fixed in lab-frame centroid;
- move by `-delta_x/pitch` in sensor coordinates;
- therefore move by `-3.63636 px` for `+0.02 mm` at 5.5 µm pitch.

The same applies to `+y`. Observed production values:

| Intervention | Current stored lab centroid delta | Current sensor/image delta | Bilinear stored lab delta | Bilinear sensor/image delta |
|---|---:|---:|---:|---:|
| camera x +0.02 mm | -0.5763 px | -4.2127 / -4.2086 px | +0.0098 px | -3.6266 / -3.6230 px |
| camera y +0.02 mm | -0.4242 px | -4.0606 / -4.0566 px | +0.0127 px | -3.6236 / -3.6201 px |

The signs are correct in sensor coordinates: moving the camera `+x` or `+y`
makes the beam move toward negative sensor pixels. The current sampler also
introduces a spurious lab-centroid motion. The bilinear comparison nearly
recovers the expected lab invariance and `-3.636 px` sensor response.

Lens interventions change the optical field. Observed current sensor/image
centroid deltas are positive for positive lens motion:

| Intervention | current sensor / image delta | bilinear sensor / image delta |
|---|---:|---:|
| lens x +0.02 mm | +3.8373 / +3.8335 px | +3.6845 / +3.6809 px |
| lens y +0.02 mm | +1.4138 / +1.4124 px | +1.2004 / +1.1993 px |

No-op repeatability is bitwise exact.

## Is the dual representation compatible with the task?

It is internally transformable because camera x/y and pitch are present in
the 17-value setup/position/current-metric context. The numerical forward
model can therefore learn either frame.

It is nevertheless semantically consequential:

- A lab-frame target treats camera movement as changing the measurement
  window, not the beam. Away from clipping, an ideal camera move should not
  change lab centroid.
- A target stated as "centre the beam in the camera image" is sensor-frame and
  should move by the opposite of camera translation.
- The JSON schemas call both values `centroid_*_px` without a frame field.
  The config/manifest text mentions the legacy convention, but the structured
  target schema and image-conditioning interface do not make the transform a
  first-class contract.
- An image-conditioned model can receive a sensor-frame image paired with a
  lab-frame label. Because camera position is also supplied this is learnable,
  but it adds an avoidable coordinate transform and makes direct visual-label
  checks misleading.

Effects by use:

| Use | Effect |
|---|---|
| Forward numerical training | deterministic and learnable with camera position, but camera effects mix real clipping with sampler artifacts |
| Inverse/control training | task meaning depends on frame; camera actions are nearly inert for a contained beam under lab targets but directly useful under sensor-centering targets |
| Target definition | `centroid_*_px` is ambiguous without a declared frame |
| Image-conditioned models | image and label are in different frames unless transformed; position context makes this learnable, not automatically correct |
| Evaluation | comparisons remain valid within the legacy contract; image-derived evaluation needs the explicit transform |

## Audit 2 conclusion

**Classification: deliberate dual representation but insufficiently
documented at the schema/task boundary.** Numerically it is fully consistent
after the camera translation and endpoint-spacing transform; it is not a
mixture of unrelated metric sources.

Recommendation: **documentation/schema clarification only** for the existing
legacy numerical benchmark. Before claiming image-conditioned sensor-centering
performance, explicitly choose a task frame and either expose both
representations or transform labels into the sensor frame.

# Audit 3: whether `power_w` is ignored

## Static and causal result

`power_w` is created, serialized, hashed, loaded into
`GaussianSource.power`, and included in model features. It is never read by
`gaussian_source_field()`, which always creates the same peak-amplitude
Gaussian. The parameter is dead before propagation, not canceled later by
normalization.

For 0.25x, 0.5x, 1x, 2x, and 4x:

| Observable | Expected ratio at 0.25x / 0.5x / 1x / 2x / 4x if active | Observed ratios |
|---|---|---|
| source field amplitude | 0.5 / 0.7071 / 1 / 1.4142 / 2 | all exactly 1 |
| source integrated intensity | 0.25 / 0.5 / 1 / 2 / 4 | all exactly 1 |
| propagated grid intensity/power | 0.25 / 0.5 / 1 / 2 / 4 | all exactly 1 |
| captured power | 0.25 / 0.5 / 1 / 2 / 4 | all exactly 1 |
| unnormalized image sum and max | 0.25 / 0.5 / 1 / 2 / 4 | all exactly 1 |
| normalized image | 1 / 1 / 1 / 1 / 1 | all exactly 1 |
| centroid x/y and sigma x/y | 1 / 1 / 1 / 1 / 1 | all exactly 1 |
| peak intensity | 0.25 / 0.5 / 1 / 2 / 4 | all exactly 1 |
| serialized five-metric target | geometry invariant; peak should scale | all five exactly invariant |

The exact array hashes are identical at every power. At 1x the current source
integrated intensity is `1.5707963e-6` in simulator integral units and sampled
captured power is `1.5686099e-6`; these are not 1 W. Therefore
`captured_power` is not currently an absolute watt measurement.

Under a passive linear optical model, if `power_w` means total source power,
the source field should be normalized so
`sum(|E|^2)*dx^2 = power_w`. Ratios of field amplitude should then scale as
the square root of the power ratio, while raw intensity, raw peak, and
captured power scale linearly. Centroid and normalized widths should remain
unchanged absent detector nonlinearity.

`peak_intensity` is the maximum of the raw simulator sensor intensity. It is
not a normalized-image maximum, but because the source amplitude is
peak-normalized it is also not a calibrated physical W/m2 intensity.

## Existing non-protected dataset/model impact

No existing v11 or v12 smoke dataset directory was present at audit start.
Three non-protected training grids that are configured as current/legacy
sources were checked only for `setup.power_w`:

| Dataset | Groups with power | Distinct values | Range [W] |
|---|---:|---:|---:|
| `specialist_rebuild_v2` train | 2,500 | 2,499 | 0.800080-1.199998 |
| `control_rebuild_v4_quickcheck` train | 2,000 | 2,000 | 0.581072-1.419703 |
| `control_rebuild_v5_numerical` train | 6,000 | 6,000 | 0.580176-1.419725 |

`power_w` is one of the eight v12 setup fields and therefore one of the 17
setup/position/current-metric inputs before action features. Older v11/v9-style
features also include it among setup fields. The model is asked to fit an
irrelevant, continuously varying feature. A flexible model may learn to ignore
it, but it wastes capacity and can create spurious correlations with geometry.
Any target expectation that raw peak or captured power changes with
`power_w` is false in these simulator-generated datasets.

## Audit 3 conclusion

**Classification: dead/ignored input.**

Recommendation: **fix before medium/full v12 generation, but current v11 work
may continue**. The design decision must be explicit:

- For absolute-intensity/captured-power tasks, normalize source-field
  amplitude to total `power_w` and retain raw peak/captured power.
- For purely normalized shape/control tasks, remove or freeze `power_w` and
  define peak as normalized/relative.
- If both uses matter, expose both normalized image/peak and calibrated
  absolute intensity/power.

# Cross-problem analysis

1. Nearest/index sensor sampling does explain discontinuous centroid, width,
   peak, and image labels because all five metrics are downstream of the same
   sampled image.
2. The metrics are not upstream of the artifact. They differ from image
   coordinates by frame and endpoint scale, not by source array.
3. `power_w` is not ignored because of image or metric normalization. Raw
   source field, raw propagated field, raw image, peak, and captured power are
   already invariant.
4. An image-conditioned model can receive a sensor-frame image with lab-frame
   labels. The supplied camera position makes the transform learnable, but the
   schema does not state it directly.
5. The 128-grid smoke optical results are not physically representative:
   each sensor axis uses only 2-3 unique simulation samples and camera-x peak
   can jump 6.94 tolerances. Pipeline conclusions such as schema validation,
   serialization, no-op anchoring, group separation, and deterministic
   execution remain valid.
6. V11/v12 learned and oracle scores remain interpretable only as results
   inside this current discrete simulator. They should not be presented as
   evidence of smooth continuous-camera control or absolute-power behavior.
7. No evidence here establishes that the low learned-MPC smoke score was
   caused by these issues. Its tiny training set remains a sufficient stated
   limitation.

Existing non-protected simulator datasets inherit the camera sampler and
legacy frame convention; they also contain varying dead `power_w`. Frozen or
protected datasets were not inspected, so this report does not claim a
file-level audit of them.

# Minimum changes for a future implementation

No production change is made in this audit.

1. Define sensor physics first: pixel-centre convention, finite pixel-area
   integration, field/intensity resolution, clipping, and whether camera
   translation changes only sampling.
2. Increase or redesign simulation/sensor resolution; do not treat bilinear
   interpolation as sufficient for the 128-grid smoke path.
3. Choose and schema-version the target frame. Prefer explicit
   `lab_metrics` and `sensor_metrics`, or one declared frame plus a tested
   transform.
4. Decide whether intensity is absolute or normalized, then either activate
   `power_w` by source-energy normalization or remove/freeze it.
5. Generate medium/full v12 data only after these contracts and regression
   tests are approved.

# Recommended future regression tests

- Exact no-op repeatability and no-op action.
- Monotone subpixel camera sweep with a bound on adjacent derivative
  variation.
- Pixel-area/power conservation away from finite-sensor edges.
- Expected clipping response at each sensor edge.
- Lab centroid invariance and sensor centroid `-camera_delta/pitch`.
- Lens x/y sign and smoothness at multiple grid resolutions.
- Exact lab-to-sensor transform and width endpoint convention.
- Peak identity with the declared raw or normalized image.
- Source integrated power equals `power_w`.
- Raw intensity/peak/captured-power ratios follow power ratios.
- Normalized images, centroids, and widths remain invariant under pure power
  scaling.
- Resolution-convergence test comparing 128, 256, 512, and 1024+ grids.
- Identical seeds/configurations reproduce bitwise-identical audit outputs.

# Deliverables and verification

Created under this versioned directory:

- `simulator_three_issue_audit.md` — this report.
- `source_data_flow_trace.md` — exact source/function trace.
- `scripts/audit_core.py`, `scripts/run_audit.py` — isolated audit
  implementations.
- `tests/test_three_issue_audit.py` — 12 current-behavior tests plus three
  strict expected failures for desired semantics.
- `data/continuity_sweeps.csv`, `data/continuity_summary.csv` — raw production
  sweeps and summaries.
- `data/sensor_metric_comparison.csv`,
  `data/coordinate_interventions.csv`,
  `data/coordinate_frame_table.csv` — frame audit.
- `data/power_causal_scaling.csv`,
  `data/dataset_power_variation.csv` — power audit.
- `data/interpolation_semantics.csv` — current/intensity/complex-field spot
  comparison.
- `data/actuator_limit_boundary.csv` — expected v12 projection boundary.
- `data/smoke_128/` — equivalent smoke-resolution CSVs.
- `plots/` and `plots/smoke_128/` — metric and adjacent-change PNGs for all
  four main sweeps.
- `commands_used.txt` — exact execution commands.
- `files_created.txt` — exact audit artifact inventory, including package
  markers.

Final diagnostic test result:

```text
12 passed, 3 xfailed
```

The strict expected failures are intentionally retained for:

1. smooth camera response without plateaus;
2. direct equality of legacy stored centroid and sensor-image centroid;
3. physically active raw-intensity scaling with `power_w`.

No production file was modified by the audit.
