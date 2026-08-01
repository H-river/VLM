# V12 source and data-flow trace

Version: `v12_sensor_power_semantics_v1` / schema `v12.1.0`

## Corrected optical path

| Stage | Source / function | Contract |
|---|---|---|
| Version selection | `continuous_control_v12/simulator.py::default_simulator_fixed` | The corrected path exists only when `simulator_semantics_version` is explicitly supplied. Absence of the field selects legacy behavior. |
| Setup construction | `continuous_control_v12/simulator.py::build_optical_setup` | Dataset nm/mm/um values are converted explicitly to SI and copied into `OpticalSetup`. |
| Simulation grid | `optical_sim/src/simulator.py::_make_grid` | Uniform inclusive samples on the configured `[-extent,+extent]` square. Corrected data config: 1536 samples and +/-6.25 mm, giving 8.14 um pitch. |
| Source shape | `optical_sim/src/simulator.py::gaussian_source_field` | Complex TEM00 amplitude on the simulation grid. This function itself remains legacy-compatible and peak-normalized. |
| Source power | `optical_sim/src/simulator.py::normalize_field_to_power`; called only by `continuous_control_v12/simulator.py::simulate_state` | Multiplies the corrected source by `sqrt(power_w / (sum(abs(E)^2)*dx^2))`. Legacy calls do not normalize. |
| Source-to-lens propagation | `optical_sim/src/simulator.py::_fresnel_numpy` through `_BACKENDS` | FFT transfer-function propagation of the complex field. |
| Lens | `optical_sim/src/simulator.py::apply_thin_lens` | Continuous decentered quadratic phase plus grid-sampled hard circular aperture. |
| Lens-to-camera propagation | `optical_sim/src/simulator.py::_fresnel_numpy` through `_BACKENDS` | Complex lab-frame camera-plane field. Camera x/y has not yet been applied. |
| Sensor pixel centres | `optical_sim/src/simulator.py::sensor_pixel_center_coordinates` | Declared pitch, explicit centre convention, axis 0 = +y lab, axis 1 = +x lab. |
| Continuous sensor measurement | `optical_sim/src/simulator.py::_extract_sensor_region_continuous` | Order-3 tensor Gauss-Legendre finite-pixel average of bilinearly interpolated irradiance. Outside-grid quadrature samples are zero. A full-pixel valid mask is returned. |
| Bilinear kernel | `optical_sim/src/simulator.py::_bilinear_sample_uniform_zero` | Uses continuous fractional indices on a verified uniform grid and explicit zero padding. No left-searchsorted selection is used. |
| Metrics | `optical_sim/src/metrics.py::compute_metrics` | Centroid, widths, and peak are computed from the same float32 `image_raw` that is serialized. |
| Frame conversion | `optics_sft/physics/sim_adapter.py::metrics_to_state` and `metrics_to_sensor_frame_state` | Both lab-frame compatibility metrics and sensor-frame image metrics are exposed. |
| Capture auxiliaries | `continuous_control_v12/simulator.py::simulate_state` | Includes captured power in W, aperture clipping, boundary flags, valid-region fraction, source integrated power, source amplitude scale, and absolute peak. |

## Legacy separation

`optical_sim/src/simulator.py::run_simulation` still calls
`_extract_sensor_region`, whose clipped left-searchsorted behavior is
unchanged. It still calls `gaussian_source_field` without power
normalization. `continuous_control_v12/config_v12.json` does not request
corrected semantics and therefore retains its v12.0 legacy behavior.

The regression suite fixes hashes for the deterministic 128-grid legacy
source, lens-plane field, post-lens field, camera-plane field, and sensor
image.

## Transition and model flow

| Stage | Source / function | Deployable fields |
|---|---|---|
| Setup signal gate | `continuous_control_v12/generate_dataset.py::generate_group` | Data-quality v2 deterministically resamples the full setup when initial captured/source power is below 0.01. Accepted retry index and fraction are auditable sampling metadata; v1 does not enable the gate. |
| Action sampling | `continuous_control_v12/sampling.py::sample_continuous_actions` | Deterministic no-op, paired, fine, axis, multi-axis, legacy-grid, and Sobol actions. |
| Bound projection | `continuous_control_v12/contracts.py::project_action` | Per-step and remaining absolute-position bounds; requested and effective actions/positions are serialized separately in v12.1. |
| Transition assembly | `continuous_control_v12/generate_dataset.py::transition_row` | Setup context, simulator semantics, positions, current observation, physical action, next supervision, frames, hashes, and non-privileged auxiliaries. |
| Schema validation | `continuous_control_v12/schema.py::validate_transition` | Checks action/position arithmetic, setup/context hashes, active power normalization, explicit lab-metric aliases, capture schema, and q-star leakage. |
| Model features | `continuous_control_v12/world_model.py::structured_features` | Eight setup values including now-causal `power_w`, four absolute positions including camera pose, five current lab metrics, physical actions, and engineered action interactions. |
| Model target | `continuous_control_v12/world_model.py::transition_arrays` | Five tolerance-normalized next-minus-current metric deltas. Statistics are derived from train rows only. |
| Oracle-only fields | `continuous_control_v12/generate_dataset.py::generate_group` | `q_star_mm` appears only inside `oracle_metadata`; `deployed_transition_input` and `assert_no_q_star` exclude/reject it. |
| Reachability grid | `continuous_control_v12/reachability.py::discrete_81_oracle` | Corrected targets evaluate all 81 canonical actions with corrected sensor/power semantics. The cached legacy-grid simulator is selected only when corrected semantics are absent. |

Camera pose is therefore present for both numerical and optional
image-conditioned models. A sensor-frame image paired with a lab-frame target
is identifiable, though it still asks an image-conditioned model to learn the
documented frame transform.
