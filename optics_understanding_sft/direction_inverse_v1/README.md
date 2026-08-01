# Direction and inverse optics v1

This package separates three capabilities that were previously mixed together:

1. predicting qualitative beam-change directions from a visible setup and action;
2. predicting the numerical forward response of the setup; and
3. selecting an actuator action that transforms an observed beam image A into a target image B.

The optical simulator is permitted only during dataset construction, private replay auditing, and offline scoring. Model prompts contain the complete setup, calibrated observations, and declared action space. They never contain a state handle, simulator result, hidden response curve, or tool output.

Direction training uses two complementary views:

- exactly balanced single-field records for each of five outputs and three direction classes;
- natural-distribution five-field records for joint prediction and evaluation.

Inverse labels enumerate the full declared action grid. Every matching action is retained, and the minimum-motion matching action is selected deterministically. This permits unique, ambiguous, and infeasible targets without pretending that an arbitrary generating action is the only correct answer.

Inverse matching uses a stricter 0.5 px centroid-vector, 1 px per-width, and 2% peak-intensity tolerance. These values are separate from the direction deadbands: they define whether two complete beam outcomes are interchangeable for control, not whether a single change is qualitatively noticeable.

The quantitative forward stage uses `train_forward_small.py`. It jointly predicts
five numerical changes and five direction labels and reports strict all-five
success, per-field error, and skill over the zero-change baseline.

## Model roles

- `train_direction_small.py`: shared numerical encoder with five direction heads.
- `train_forward_small.py` and `train_forward_hybrid.py`: numerical change model
  and validation-tuned neural/gradient-tree blend.
- `train_inverse_direct.py`: direct four-axis action classifier trained from
  exhaustive group-safe grid replay.
- `evaluate_inverse_controller.py` and `evaluate_inverse_ensemble.py`: exhaustive
  learned-forward search and a validation-routed direct/forward ensemble.
- `evaluate_visual_pipeline.py`: calibrated image metrology followed by the same
  learned inverse controller. The deterministic meter is the preferred clean-image
  baseline; a CNN is unnecessary unless future images include nuisance variation
  that the declared calibration cannot invert.

## Coordinate frames

Numeric forward and inverse records retain the legacy lab-coordinate convention.
PNG pixels live in the camera sensor-array frame. `repair_visual_sensor_frame.py`
therefore derives separate visual matching sets and private replay states after
subtracting the final camera offset for each candidate action. Numeric and visual
records describe the same physical A-to-B pair but can correctly have different
`unique`/`ambiguous` labels.

The selected visual images use one shared absolute intensity maximum per A/B pair
and an invertible gamma of 0.5. This preserves peak and Gaussian-tail information
in 8-bit PNGs; the calibration object is mandatory input to quantitative image
measurement.

## Promotion policy

`certify_system.py` promotes components separately. A schema-valid LLM response is
not a correctness result, a forward record passes only when all five numerical
outputs pass together, and every forward score is shown beside the strict
zero-change baseline. Full autonomous promotion requires every component gate;
otherwise the output is explicitly `component_only`.
