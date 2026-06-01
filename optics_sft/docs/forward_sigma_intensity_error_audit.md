# Forward Sigma/Intensity Error Audit

## Summary Verdict

The large `sigma_x/y` and `peak_intensity` errors are most likely a model-learning and observability issue, not a dataset/evaluator field mismatch.

Main findings:

- `target.predicted_after_state` and `private_eval.after_state` match exactly for `sigma_x_px`, `sigma_y_px`, and `peak_intensity`.
- The evaluator compares the expected fields: model `predicted_after_state.{sigma_x_px,sigma_y_px,peak_intensity}` against `private_eval.after_state`.
- Sigma definitions are consistent: the simulator computes second-moment 1-sigma widths, and the SFT state converts those meter values to pixels.
- Peak intensity is raw simulator intensity, not rendered image intensity.
- Rendered PNGs are normalized independently with percentile clipping, gamma, background/noise, blur, and optional colorization; therefore they do not preserve absolute peak intensity and can distort image-derived moment widths.
- The model predictions are compressed toward a narrow numeric range, especially for peak intensity.

This means the model is being asked to predict raw simulator sigma/peak values from a before image whose rendering does not preserve those quantities faithfully, plus metadata/action that mostly explains geometry rather than raw image brightness.

## Artifacts

Diagnostic script:

- `optics_sft/scripts/diagnose_forward_sigma_intensity_errors.py`

Diagnostic outputs:

- `../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/diagnostics_sigma_intensity/diagnostics_summary.json`
- `../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/diagnostics_sigma_intensity/diagnostics_report.md`
- `../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/diagnostics_sigma_intensity/worst_sigma_examples.csv`
- `../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/diagnostics_sigma_intensity/worst_peak_examples.csv`

Inputs checked:

- `../VLM_data/physics_sft_forward_transition_v1/val.jsonl`
- `../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/val_predictions.jsonl`
- `../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/val_forward_eval.json`
- `../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/val_forward_eval.csv`

## Value Distributions

Validation rows checked: `100`.

| Source | Field | Min | Max | Mean | Std |
| --- | --- | ---: | ---: | ---: | ---: |
| private_eval | sigma_x_px | 64.9365 | 186.294 | 136.255 | 29.8565 |
| private_eval | sigma_y_px | 64.6015 | 185.366 | 136.679 | 29.7167 |
| private_eval | peak_intensity | 0.892695 | 236.496 | 36.2723 | 56.7485 |
| target | sigma_x_px | 64.9365 | 186.294 | 136.255 | 29.8565 |
| target | sigma_y_px | 64.6015 | 185.366 | 136.679 | 29.7167 |
| target | peak_intensity | 0.892695 | 236.496 | 36.2723 | 56.7485 |
| predicted | sigma_x_px | 111.9 | 169.001 | 149.604 | 13.0777 |
| predicted | sigma_y_px | 110.99 | 169.001 | 149.557 | 13.1158 |
| predicted | peak_intensity | 1.01011 | 10.148 | 5.3274 | 4.30416 |

The predicted range is strongly compressed:

- `sigma_x_px` predicted/true std ratio: `0.438`; range ratio: `0.471`
- `sigma_y_px` predicted/true std ratio: `0.441`; range ratio: `0.480`
- `peak_intensity` predicted/true std ratio: `0.076`; range ratio: `0.039`

## Target vs Private Eval Consistency

The training target and evaluation target are consistent.

Max absolute differences:

- `target.predicted_after_state.sigma_x_px` vs `private_eval.after_state.sigma_x_px`: `0.0`
- `target.predicted_after_state.sigma_y_px` vs `private_eval.after_state.sigma_y_px`: `0.0`
- `target.predicted_after_state.peak_intensity` vs `private_eval.after_state.peak_intensity`: `0.0`

Therefore, the large metric errors are not caused by training against one target and evaluating against another.

## Evaluator Field Mapping

`eval_forward_physics.py` requires these prediction fields:

- `centroid_x_px`
- `centroid_y_px`
- `sigma_x_px`
- `sigma_y_px`
- `peak_intensity`

It extracts model `predicted_after_state` and compares each field directly to `private_eval.after_state`. No alias or fallback was found for sigma/peak that would silently map to a wrong field.

The evaluator only accepts numeric Python `int`/`float` for `predicted_after_state` values. The current saved predictions are valid JSON with numeric values, so missing-field defaults are not causing these metrics.

## Sigma Definition

No sigma definition mismatch was found.

`optical_sim/src/metrics.py` computes:

- `sigma_x` and `sigma_y` as second-moment 1-sigma standard deviations in meters.
- `width_4sigma_x` and `width_4sigma_y` separately as `4 * sigma`.
- `peak_intensity` as `I.max()` on the raw simulator intensity array.

`optics_sft/physics/sim_adapter.py` maps:

- `sigma_x_m <- metrics["sigma_x"]`
- `sigma_y_m <- metrics["sigma_y"]`
- `sigma_x_px <- sigma_x_m / pixel_pitch`
- `sigma_y_px <- sigma_y_m / pixel_pitch`

So the dataset/eval `sigma_x_px` field is 1-sigma in pixels, not D4sigma, FWHM, radius, or diameter.

## Peak Intensity Definition

`peak_intensity` is raw simulator peak intensity from `I.max()`. It is not the rendered image max, not normalized `0-1`, and not uint8 `0-255`.

This matters because the rendered images do not preserve absolute simulator intensity. The renderer:

- adds optional background and read noise,
- normalizes each image using min/max or percentile clipping,
- clips to `0-1`,
- applies gamma,
- optionally colorizes,
- converts to uint8,
- optionally blurs.

For the current `medium` rendered dataset, peak values in checked PNGs were all close to saturated:

- after PNG peak mean, `0-1` scale: `0.9754`
- after PNG peak mean, uint8 scale: `248.73`
- corresponding true raw simulator peak mean in checked examples: `15.04`

This makes raw `peak_intensity` mostly unobservable from the image.

## Baselines And Collapse Evidence

The model is not literally constant, but it is close to a compressed/default predictor for sigma.

Sigma baselines:

- `sigma_x_px` model MAE: `25.55`
- `sigma_x_px` validation-mean baseline MAE: `25.04`
- `sigma_x_px` copy-before-state MAE: `1.42`
- `sigma_y_px` model MAE: `25.22`
- `sigma_y_px` validation-mean baseline MAE: `24.84`
- `sigma_y_px` copy-before-state MAE: `1.29`

Correlations:

- predicted vs true `sigma_x_px`: `0.294`
- before vs true `sigma_x_px`: `0.9977`
- predicted vs true `sigma_y_px`: `0.284`
- before vs true `sigma_y_px`: `0.9985`

This is a strong clue: the action mostly shifts beam position and only weakly changes sigma. If the model had reliable numeric before-state sigma, copying it would be very strong. Instead, the prompt only provides the rendered before image, and the rendered image moment width is not comparable to simulator sigma.

Peak intensity:

- model MAE: `31.34`
- validation-mean baseline MAE: `42.89`
- copy-before-state MAE: `12.29`
- predicted vs true correlation: `0.569`
- before vs true correlation: `0.885`

Again, after-state peak is much closer to before-state peak than to the model's compressed outputs, but raw before-state peak is not available to the model as a numeric prompt field.

## Rendered PNG Analysis

For five worst sigma examples, simple image-derived moment widths from rendered PNGs were not comparable to simulator sigma:

- after PNG sigma_x MAE vs private simulator sigma_x: `171.56 px`
- after PNG sigma_y MAE vs private simulator sigma_y: `173.24 px`

Example:

- `phys_fwd_000713`
- true after sigma: `(64.94, 64.60) px`
- rendered after PNG moment sigma: `(274.36, 274.28) px`
- rendered after PNG peak: `248/255`
- true raw peak: `15.79`

These simple PNG moment metrics are affected by normalization, background, noise, gamma, and clipping. They should not be treated as exact beam metrics, but they do show that the rendered image is not a faithful carrier of raw simulator sigma/peak.

## Code Issues Found

No direct field-mapping bug was found in:

- `optics_sft/scripts/eval_forward_physics.py`
- `optics_sft/scripts/generate_physics_transition_dataset.py`
- `optics_sft/physics/sim_adapter.py`
- `optical_sim/src/metrics.py`

The main design issue is in the data/rendering contract:

- The target asks for raw simulator `peak_intensity`.
- The image input is independently normalized and transformed, which destroys absolute intensity scale.
- The target asks for exact second-moment sigma, while the image transform can make simple apparent width very different from simulator sigma.
- The prompt does not include numeric `before_state`, even though before-state sigma/peak are extremely predictive of after-state sigma/peak for these small transition actions.

## Recommended Fixes

Priority 1: remove or change `peak_intensity` in the forward target.

- Raw simulator peak is not visually identifiable after per-image normalization.
- Prefer a normalized relative target such as `peak_intensity_ratio = after_peak / before_peak`, or a categorical direction `higher|same|lower`.
- If raw peak is required, include numeric before-state peak and render parameters, or use images whose intensity scale is globally calibrated.

Priority 2: expose numeric before-state metrics for forward transition, or change the task to predict deltas only.

- `copy before_state` is far better than the current model for sigma/peak.
- Include prompt-visible `before_state` fields if the scientific task permits it.
- Alternatively train only `predicted_change` for sigma/peak, where expected change is near zero, instead of absolute after-state values.

Priority 3: simplify the forward target while debugging.

- Keep centroid fields first because centroid and delta-centroid are already reasonable.
- Temporarily remove `sigma_x_px`, `sigma_y_px`, and `peak_intensity` from `predicted_after_state`, or split them into a separate task.

Priority 4: make rendering physically metric-preserving if sigma/peak must be inferred from images.

- Use clean/global normalization instead of per-image percentile normalization.
- Disable background/noise/blur/gamma for metric-prediction training.
- Store and expose normalization metadata if absolute intensity is part of the target.

Priority 5: add a cheap baseline evaluator.

- Report mean baseline and copy-before baseline for forward transitions.
- This will reveal whether the VLM is adding value over simple physics priors for sigma/peak.

## Reproduction

Validation commands run:

```bash
python -m compileall optics_sft optical_sim
python -m compileall -q optics_sft optical_sim
python optics_sft/scripts/diagnose_forward_sigma_intensity_errors.py \
  --val-jsonl ../VLM_data/physics_sft_forward_transition_v1/val.jsonl \
  --predictions-jsonl ../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/val_predictions.jsonl \
  --eval-csv ../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/val_forward_eval.csv \
  --output-dir ../VLM_runs/qwen25vl_3b_qlora_forward_only_v1/diagnostics_sigma_intensity
```
