# Width-relative secondary-reflection generator specification

Version: `secondary_reflection_primary_sigma_direction_v1`  
Separation mode: `primary_sigma_direction`

## Geometry

For a nonnegative clean primary image `I`, the generator computes its intensity-weighted centroid and full 2x2 covariance `Sigma_primary` before creating any secondary component. For a preregistered direction angle `theta`,

`u = [cos(theta), sin(theta)]`

`sigma_direction = sqrt(u^T Sigma_primary u)`

`d_reflection = k * sigma_direction`.

The recorded covariance source is always `clean_primary_before_reflection_injection`. No function recomputes the separation from the combined image.

## Component construction and power

The secondary begins as the clean primary resampled about its clean centroid by a scalar width ratio. It is peak-aligned to the primary, translated by `d_reflection * u` with bilinear interpolation, and multiplied by the relative amplitude. The resulting sensor image is

`I_anomaly = I_primary + amplitude * I_secondary_shifted`.

Power is added, not taken from the primary and not redistributed. Diagnostic PNGs are peak-normalized only after addition, matching the previous preprocessing contract. Full-sensor control observations retain physical intensity rather than peak normalization.

## Boundary handling

Translation uses zero fill with no periodic wrap. Any component outside the finite image is clipped. The generator records the retained component-power fraction. It never reflects, wraps, or recenters a truncated component.

## Camera, preprocessing and matching

The clean primary comes from the unchanged corrected v12 simulator and the previous overlay-free 128x128 canonical preprocessing. The generator changes reflection formulation only; the small-model architecture, rotation augmentation, optimizer, training budget, image size and grayscale serialization remain unchanged.

For every anomaly, the old clean-counterfactual fitter produces a single translated/anisotropically scaled clean beam with the same five moments. The metrics remain centroid x/y, width x/y and peak. Frozen tolerances are `[1, 1, 2, 2, 0.05]`; acceptance requires every normalized absolute difference <= 0.25 and total normalized L2 <= 0.40. Both float arrays and reloaded 8-bit-equivalent arrays are measured and recorded.

## Model-input boundary

Models may receive only the grayscale image, five metrics, one short-history metric vector, target, metric uncertainties and randomized candidate decisions. Generator parameters, direction, width quartile, boundary status, setup identity/hash, severity, labels and oracle/control outcomes are evaluator-only.

The small CNN used in this experiment is a visual diagnostic, not a VLM.
