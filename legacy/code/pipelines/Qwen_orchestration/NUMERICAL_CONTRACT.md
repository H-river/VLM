# Numerical contract

This file fixes the numerical meaning and ordering of every value passed from
the orchestrator to a frozen specialist. Qwen emits named JSON fields. A
deterministic wrapper converts those fields to arrays in the order below.

## 1. Shared quantities

### Setup: 12 values

| Index | Field | Unit |
|---:|---|---|
| 0 | `wavelength_nm` | nm |
| 1 | `beam_waist_mm` | mm |
| 2 | `power_w` | W |
| 3 | `lens_focal_length_mm` | mm |
| 4 | `lens_aperture_mm` | mm |
| 5 | `source_to_lens_mm` | mm |
| 6 | `lens_to_camera_mm` | mm |
| 7 | `lens_x_offset_mm` | mm |
| 8 | `lens_y_offset_mm` | mm |
| 9 | `camera_x_offset_mm` | mm |
| 10 | `camera_y_offset_mm` | mm |
| 11 | `pixel_size_um` | um |

### Beam state: 5 values

| Index | Field | Unit |
|---:|---|---|
| 0 | `centroid_x_px` | px |
| 1 | `centroid_y_px` | px |
| 2 | `sigma_x_px` | px |
| 3 | `sigma_y_px` | px |
| 4 | `peak_intensity` | sensor intensity units |

### Proposed action: 4 values

| Index | Field | Unit |
|---:|---|---|
| 0 | `lens_x_delta_mm` | mm |
| 1 | `lens_y_delta_mm` | mm |
| 2 | `camera_x_delta_mm` | mm |
| 3 | `camera_y_delta_mm` | mm |

## 2. Forward and direction input

The public input has exactly:

```text
12 setup values + 5 current-beam values + 4 action values = 21 values
```

The order is the concatenation of the three tables above. Before the array
enters a specialist, the wrapper:

1. rejects missing, extra, Boolean, NaN, or infinite values;
2. converts stated equivalent units to the canonical units above;
3. applies `log(1 + max(peak_intensity, 0))` to the peak input, matching the
   frozen training code;
4. applies the normalization stored in the frozen model bundle.

Qwen does not perform steps 3 or 4.

## 3. Direction specialist

```text
21 normalized inputs
-> 128 hidden values
-> 64 hidden values
-> five independent outputs, each containing 3 class scores
```

The five outputs correspond, in order, to centroid x, centroid y, sigma x,
sigma y, and peak intensity. Each output selects one class:

```text
0 = decrease
1 = no_change
2 = increase
```

Thus the raw output shape for one example is `5 x 3 = 15` scores. The direction
model does not produce five numerical changes.

## 4. Forward specialist

The neural component has:

```text
21 normalized inputs
-> 128 hidden values
-> 64 hidden values
|-> regression branch: 64 -> 64 -> 5 numerical changes
`-> direction branch: five independent 64 -> 3 class-score heads
```

The five regression outputs are changes, not final beam values:

```text
delta centroid_x_px
delta centroid_y_px
delta sigma_x_px
delta sigma_y_px
delta peak_intensity
```

The stored hybrid forward predictor blends those five neural regression
outputs with a tree regressor. The tree receives the same 21 public values plus
25 deterministic engineered values, for 46 internal values. Its direction
classes still come from the neural branch. These 25 derived values are computed
inside the frozen wrapper and are never Qwen outputs.

## 5. Inverse specialist

The public inverse input contains:

```text
12 setup values + 5 current-beam values + 5 desired-beam values = 22 values
```

The direct inverse classifier internally adds:

```text
5 desired-minus-current values + 4 deterministic derived values
```

so its internal feature array has `22 + 5 + 4 = 31` values.

The action space is fixed:

```text
lens x:   {-0.05, 0.00, +0.05} mm
lens y:   {-0.05, 0.00, +0.05} mm
camera x: {-0.02, 0.00, +0.02} mm
camera y: {-0.02, 0.00, +0.02} mm
```

There are `3 x 3 x 3 x 3 = 81` possible actions. Qwen selects only the inverse
route; it does not invent an action. The frozen inverse system selects an action
from this 81-element set.

## 6. Image routes

An image is not inserted directly into a numerical specialist. The dataflow is:

```text
beam image + calibration
-> frozen deterministic image meter
-> 5-value beam state
-> numerical specialist
```

Qwen sees the image to determine its role and whether the request is visually
grounded. The meter, not Qwen, calculates centroid, width, and peak intensity.

