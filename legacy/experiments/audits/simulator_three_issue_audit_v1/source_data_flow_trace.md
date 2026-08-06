# Source and data-flow trace

Date: 2026-07-30

Scope: current `optical_sim`, v11 system-aligned generation, and v12
continuous control. No raw v10 locked-test data, frozen-v9 checkpoint, or
preserved v9/v10 artifact was opened or modified. There is no repository-local
`AGENTS.md`; this is also recorded in `V11_V12_AUDIT_AND_DESIGN.md:9-17`.

## Coordinate and propagation path

| Stage | Current source | Function / fields | Convention and unit |
|---|---|---|---|
| Simulation grid | `optical_sim/src/simulator.py:22-27` | `_make_grid()` | `N` inclusive samples over `[-extent,+extent]`; metres; `dx=2*extent/(N-1)` |
| Source schema | `optical_sim/src/optical_elements.py:13-28` | `GaussianSource` | wavelength and waist in m; `power` is declared as normalized power `[W]` |
| Source field | `optical_sim/src/simulator.py:34-49` | `gaussian_source_field()` | lab-frame complex field on the simulation grid; peak-amplitude Gaussian; `source.power` is not read |
| Source-to-lens propagation | `optical_sim/src/simulator.py:56-73` | `_fresnel_numpy()` | FFT transfer-function propagation; complex field; lab grid |
| Lens position | `optical_sim/src/simulator.py:120-139` | `apply_thin_lens()` | `X_lens=X-lens.x_offset`, `Y_lens=Y-lens.y_offset`; metres; offset affects continuous phase and a hard aperture mask |
| Lens-to-camera propagation | `optical_sim/src/simulator.py:192-209` | `run_simulation()` | complex lab-frame camera-plane field; camera x/y does not affect propagation |
| Sensor coordinates | `optical_sim/src/simulator.py:146-175` | `_extract_sensor_region()` | lab coordinates centred on camera `(ox,oy)`; metres; endpoints are `ox +/- W*pitch/2`, giving step `W/(W-1)*pitch` |
| Sensor mapping | `optical_sim/src/simulator.py:164-172` | `np.searchsorted()` and `np.ix_()` | left insertion index, clipped; no nearest-distance comparison or interpolation; quantized to simulation-grid indices |
| Saved v12 image | `continuous_control_v12/generate_dataset.py:87-105` | `save_capture()` | sampled, unnormalized intensity array saved in compressed NPZ |
| Beam measurement | `optical_sim/src/metrics.py:35-95` | `compute_metrics()` | computed after sensor sampling, using the lab-coordinate `sensor_X/Y`; centroid/sigma in m; peak is `intensity.max()` |
| Legacy five metrics | `optics_sft/physics/sim_adapter.py:50-80,114-120` | `metrics_to_state_m()`, `state_m_to_state_px()` | centroid is lab-frame metres divided by declared pitch plus array centre; widths divided by pitch; peak unchanged |
| Sensor-frame conversion | `optics_sft/physics/sim_adapter.py:82-111` | `state_m_to_sensor_frame_px()` | subtract camera x/y before division by pitch; translation does not affect width |
| V12 capture | `continuous_control_v12/simulator.py:212-291` | `simulate_state()` | metrics, unnormalized `float32` intensity, sampled captured power, boundary auxiliaries |
| V11 fixed grid | `control_rebuild_v5/simulator_grid.py:23-36,39-122` | `_state()`, `simulate_fixed_action_grid()` | same sampler and post-sampling metrics; camera fields are cached but semantics are unchanged |

The current sampler is more precisely described as a clipped
`searchsorted(..., side="left")` lookup than as nearest-neighbour. Except at
an exact grid coordinate it selects the grid sample immediately to the
positive side. The implementation report's "nearest-neighbour" statement is
directionally correct about integer-index sampling and discontinuity but not
exact about the selection rule.

## Actuator path and boundaries

| Item | Source | Meaning |
|---|---|---|
| mm-to-m setup application | `continuous_control_v12/simulator.py:167-209` | setup context and four absolute positions are copied into `OpticalSetup`; all position fields use explicit `*1e-3` |
| Delta application in legacy adapter | `optics_sft/physics/sim_adapter.py:37-47` | lens and camera action deltas in mm are added to metre-valued setup fields |
| V12 position update | `continuous_control_v12/contracts.py:209-235` | action is clipped to per-step bounds and remaining absolute-position bounds before addition |
| Per-step limits | `continuous_control_v12/config_v12.json:16-20` | lens x/y `+/-0.05 mm`; camera x/y `+/-0.02 mm` |
| Absolute domain | `continuous_control_v12/config_v12.json:22-28` | all four positions `+/-3 mm`; explicitly a repository sampling domain, not a hardware limit |

Camera x/y is applied only when sensor coordinates are built, after both
propagations. Lens x/y is applied at the thin-lens phase/aperture before the
second propagation. V12 positions are ordinary floating-point values; there
is no actuator quantization before simulation. The hidden quantization is in
the sensor lookup and, for the hard aperture, in membership of simulation-grid
samples.

## Metric frames, axes, and formulae

Array column index increases with physical `+x`; array row index increases
with physical `+y`. The numerical arrays are not flipped before NPZ storage.
The optical simulator's optional PNG helper displays with `origin="lower"`
(`optical_sim/src/io_utils.py:53-64`), which is a display choice rather than a
change to stored array order.

For width `W`, height `H`, declared pitch `p`, camera translation `(c_x,c_y)`,
and zero-based array pixel `(j,i)`:

```text
production sensor x(j) = c_x + (j - (W-1)/2) * p * W/(W-1)
production sensor y(i) = c_y + (i - (H-1)/2) * p * H/(H-1)

stored legacy centroid x_px = centroid_lab_x/p + (W-1)/2
stored legacy centroid y_px = centroid_lab_y/p + (H-1)/2

sensor-frame centroid x_px = stored_x_px - c_x/p
sensor-frame centroid y_px = stored_y_px - c_y/p
```

An image-derived calculation using documented pixel centres uses
`(j-(W-1)/2)*p`, not `W/(W-1)*p`. Consequently, after the camera translation
is removed, production widths are larger than image-index widths by exactly
`W/(W-1)` (and `H/(H-1)`), subject only to floating-point roundoff.

## Image construction and normalization

- V12 stores raw sampled intensity without display normalization
  (`continuous_control_v12/generate_dataset.py:87-105`).
- `peak_intensity` is the maximum of that same raw sampled array
  (`optical_sim/src/metrics.py:83-95`).
- The optional v12 image embedding divides 4x4 pooled means by the current
  peak but also appends `log1p(image.sum())` and `log(peak)`;
  it is not wholly scale invariant
  (`continuous_control_v12/world_model.py:93-114`).
- The general RGB rendering helper defaults to per-image min/max or
  percentile normalization (`optics_sft/physics/rendering.py:33-46,63-111`).
  It is separate from the v12 NPZ image path.

## Tolerances and units

The current five output fields are declared in
`continuous_control_v12/config_v12.json:29-41`:

| Field | Unit / current convention | Tolerance |
|---|---|---:|
| `centroid_x_px` | lab-frame pseudo-pixel | 1 px |
| `centroid_y_px` | lab-frame pseudo-pixel | 1 px |
| `sigma_x_px` | declared-pitch pseudo-pixel | 2 px |
| `sigma_y_px` | declared-pitch pseudo-pixel | 2 px |
| `peak_intensity` | raw simulator intensity unit | 5% of current peak, floor `1e-6` |

The executable tolerance definition is
`continuous_control_v12/contracts.py:238-251`.

## `power_w` complete trace

| Stage | Source | Read/write and effect |
|---|---|---|
| V12 creation | `continuous_control_v12/simulator.py:71-97` | sampled uniformly over 0.75-1.25 W |
| V11 creation | `physics_structured_rebuild_v11/generate_system_aligned.py:80-120` | uses the same sampled setup context and serializes it in each group |
| Older non-protected creation | `specialist_rebuild_v2/build_dataset.py:79-130` | varies `power_w` over training/OOD ranges |
| Field contract | `continuous_control_v12/contracts.py:15-24` | third of eight setup-context fields |
| Setup hash | `continuous_control_v12/contracts.py:64-76` | included in the setup identity |
| Transition schema | `continuous_control_v12/schemas/transition_v12.schema.json:91-113` | required numeric setup-context field |
| Setup construction | `continuous_control_v12/simulator.py:173-177` | copied into `cfg["source"]["power"]` |
| Dataclass construction | `optical_sim/src/optical_elements.py:95-129` | copied into `GaussianSource.power` |
| Initial field | `optical_sim/src/simulator.py:34-49` | **not read**; field peak is determined only by the Gaussian shape |
| Propagated field/intensity | `optical_sim/src/simulator.py:56-95,120-139` | linear field operations receive the already power-invariant source |
| Metrics | `optical_sim/src/metrics.py:35-95` | centroid/width/peak are computed from the invariant sampled intensity |
| Captured power | `continuous_control_v12/simulator.py:276-284` | `intensity.sum()*pixel_pitch**2`; invariant because input intensity is invariant |
| Transition serialization | `continuous_control_v12/generate_dataset.py:130-162` | setup context, current/next metrics, and captured-power auxiliary are serialized |
| Manifest disclosure | `continuous_control_v12/generate_dataset.py:415-470` | explicitly records that power is retained although ignored, and sets `power_affects_current_simulator=false` |
| Model feature | `continuous_control_v12/world_model.py:26-90` | included in the eight setup values within the 17-value setup/position/current-metric base context |
| Model target | `continuous_control_v12/world_model.py:171-249` | five tolerance-normalized metric deltas plus log captured-power auxiliary; all simulator targets are power invariant under causal perturbation |

There is no later cancellation: `power_w` is dead before the initial complex
field is created. Normalized display images would cancel a physically correct
global intensity scale, but the raw source field, raw propagated intensity,
raw NPZ, raw peak, and captured-power output are already identical here.
