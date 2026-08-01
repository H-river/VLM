# V12 sensor and power semantics

Version: `v12_sensor_power_semantics_v1`  
Dataset schema: `v12.1.0`

This path is opt-in through `config_v12_semantics_v1.json`. The legacy
`config_v12.json` and `optical_sim.run_simulation()` path remain unchanged.

## Propagation and measurement

The source field is a sampled complex TEM00 Gaussian on the inclusive
simulation grid. For v12 semantics only, it is scaled so

```text
sum(abs(E_source)**2) * simulation_grid_pitch**2 == power_w
```

The FFT propagation and thin-lens/aperture operations are unchanged.
The corrected data configuration uses a `1536 x 1536` grid over
`[-6.25,+6.25] mm` (8.14 um pitch). The legacy `[-30,+30] mm` extent is not changed on the
legacy path. The narrower corrected extent is required because the thin-lens
phase is badly under-resolved at the legacy 58.65 um pitch: at the
deterministic audit state, legacy-extent 1024 versus 2048 grids differed by
20.26 task tolerances. Across twelve deterministic ordinary, focusing,
clipping, camera-boundary, tolerance-boundary, and high-offset setups,
`+/-6.25 mm` at 1536 versus 2048 differed by at most 0.068 tolerance and
0.0061% captured power. The extent also fully contains the worst legal
1024-pixel, 6 um-pitch sensor footprint at the configured +/-3 mm camera
domain.

The camera measures a finite-pixel average of irradiance. Pixel centres are

```text
x[j] = camera_x + (j - (W - 1)/2) * sensor_pixel_pitch
y[i] = camera_y + (i - (H - 1)/2) * sensor_pixel_pitch
```

Array axis 1 increases with lab `+x`; array axis 0 increases with lab `+y`.
The camera pose is the centre of the sensor pixel-edge rectangle. Each pixel
uses tensor-product Gauss-Legendre quadrature over its physical area.
Irradiance `abs(E)**2`, not wrapped phase, is bilinearly interpolated.
Quadrature samples outside the propagated field are zero padded. A mask marks
pixels whose complete area lies inside the propagated grid.

This irradiance-area model was selected instead of complex-field point
sampling because the audit showed that real/imaginary bilinear interpolation
on the original 1024 / +/-30 mm candidate lost roughly 55% of sampled power.
Even after the final propagation grid was refined, that point-complex
candidate lost 5.35% of sampled power and differed by 0.594 task tolerance
from the finite-pixel irradiance reference. A camera is an irradiance
integrator, and an oversampled quadrature comparison is part of the v12
acceptance audit.

## Intensity and power

- `power_w`: integrated source optical power in watts.
- `image_raw`: pixel-area-average irradiance in `W/m^2`.
- `image_normalized`: `image_raw / max(image_raw)` for shape-only inspection.
- `peak_intensity` and `peak_intensity_abs`: maximum raw irradiance in `W/m^2`.
- `captured_power_w`: `sum(image_raw) * sensor_pixel_pitch**2`.
- `intensity_normalization`: `none_raw_irradiance_w_per_m2`.

In the passive linear model, field amplitude scales as `sqrt(power_w)`;
raw intensity, absolute peak, and captured power scale as `power_w`;
centroids, widths, and normalized images are invariant.

## Frames

The five deployable numerical targets retain the historical lab-frame
pseudo-pixel convention for compatibility. Images are sensor-frame arrays.
Every v12.1 transition serializes both lab- and sensor-frame metric forms,
camera pose, pitch, origin, axis convention, and the transform:

```text
centroid_sensor_px = centroid_lab_px - camera_pose_m / pixel_pitch_m
sigma_sensor_px = sigma_lab_px
```

The target control frame is
`lab_frame_legacy_pseudo_pixels`. Camera pose is included in the numerical
model input, so the sensor-image/lab-target pairing is identifiable.

## Dataset setup signal floor

`config_v12_semantics_v2.json` preserves the simulator and sensor semantics
above but versions the data-generation policy. A sampled group is accepted
only when its initial captured/source power fraction is at least 0.01.
Otherwise the complete optical setup and initial pose are resampled from a
deterministic retry seed, for at most 16 attempts. The accepted retry index
and initial captured-power fraction are serialized in every transition.

This is a dataset learnability policy, not a change to propagation or sensor
validity. It prevents nearly dark, numerically ill-conditioned starting
states from dominating tolerance-normalized regression. The v1 configuration
does not contain these keys and retains byte-identical generation behavior.
