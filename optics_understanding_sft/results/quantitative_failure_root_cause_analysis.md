# Root-cause analysis of poor quantitative optics performance

**Checkpoint analyzed:** corrective-v2 seed 314  
**Evaluation set:** independent `dev_v2`, 600 records  
**Supporting data:** `corrective_v2_seed314_quantitative_failure_analysis.json`

## Executive conclusion

The weak quantitative performance is not one isolated visual failure. It is a combination of an objective mismatch, default-value shortcuts, insufficiently discriminative metrics, hidden or tool-like physics, fragile numerical serialization, and limited visual coverage. The strongest direct finding is that the checkpoint does not merely have high forward error: it is worse than predicting zero change on every forward output field.

## Confirmed causes

### 1. Zero-change shortcut

Of 58 parseable forward outputs:

- 30 keep every predicted change within 0.05 of zero;
- peak change is exactly zero in 48;
- sigma-x change is exactly zero in 48;
- sigma-y change is exactly zero in 51.

Baseline-relative skill, defined as `1 - model absolute error / zero-baseline absolute error`, is negative for all five fields:

| Field | Model MAE | Zero-baseline MAE | Skill |
|---|---:|---:|---:|
| Centroid x | 2.2082 px | 2.0848 px | -0.059 |
| Centroid y | 2.2381 px | 2.0961 px | -0.068 |
| Peak intensity | 4.8149 | 4.8110 | -0.001 |
| Sigma x | 1.1139 px | 1.1123 px | -0.001 |
| Sigma y | 0.9485 px | 0.9461 px | -0.003 |

The model therefore adds noise without improving on a zero prediction.

### 2. Existing tolerances reward false positives

Rubric v2 can award full forward credit when every prediction is near zero, provided the true response lies inside 2 px centroid/width and 5% intensity tolerances. Five forward records received 1.0, but manual review found no credible complete prediction.

The metric currently measures sensor-level acceptability, not whether the model learned the response. Both quantities are useful, but they must be reported separately.

### 3. Performance collapses as response magnitude grows

Mean forward task score is 0.575 on the lower half of target response magnitudes and 0.218 on the upper half. The model succeeds primarily where default predictions are hardest to distinguish from the target.

### 4. Direction is not learned reliably

For target components with magnitude above 0.05, sign accuracy is:

- centroid x: 15.8%;
- centroid y: 17.2%;
- peak intensity: 0%;
- sigma x: 0%;
- sigma y: 0%.

This rules out the interpretation that the model understands direction but lacks only decimal precision.

### 5. Strong status priors replace case discrimination

The development labels are balanced, but predictions are not:

- control: 112 infeasible and 8 feasible, versus 60/60 targets;
- sufficiency: 77 insufficient and 43 answerable, versus 60/60 targets.

Status accuracy is only 0.533 for control and 0.542 for sufficiency, barely above the 0.50 constant-class baseline. Diagnosis, which has clearer discrete evidence, reaches 0.867 status accuracy.

### 6. Literal numerical tool calls are fragile

V6 removed numerical thresholding from the LLM, but asked it to serialize ordered arrays. Step 120 produced 0/41 exact control calls. The model invented compressed arguments such as `action_trial_index` rather than preserving residual, motion, tolerance, and order arrays.

This shows that ordinary SFT does not reliably teach long, exact numeric copying either.

### 7. Visual performance is uneven and sparsely measured

Only 30/600 development records are visual. Visual forward prediction scores 0.167 versus 0.407 for text, and visual counterfactual reasoning scores 0.356 versus 0.820 for text. However, visual causal effects score 0.950 and visual diagnosis 0.879.

The evidence supports a weakness in **visual quantitative tasks**, not a universal inability to process images. The sample is too small and task-confounded for a general visual conclusion.

### 8. The public forward prompt is not a complete simulator state

The prompt-visible `setup` contains safe physical metadata, but deliberately omits the baseline lens and camera lateral offsets stored in the private simulator configuration. Those offsets affect the response to the requested action. Replaying the simulator from the visible setup therefore fails before numerical prediction: the public fields are not a complete `OpticalSetup` and cannot reproduce the exact after-state.

The current observation partially summarizes the hidden state, so approximate prediction may still be learnable. It is not enough to certify that the exact Fresnel transition is identifiable. A deterministic forward tool needs either a trusted opaque experiment-state handle or an independently calibrated state estimator; exposing private offsets directly would leak simulator state into the benchmark.

## Likely contributing causes

These explanations are technically plausible and consistent with the results, but are not individually isolated by the current experiments.

### 9. Token-level SFT is misaligned with numerical error

Cross-entropy treats number strings as token sequences. It does not directly penalize physical distance, wrong sign, simulator failure, or failure to beat a zero baseline. Teacher-forced loss improved while generated physical decisions remained poor in several rounds.

### 10. Exact Fresnel propagation is a tool problem

The simulator response is high-precision and can be non-monotonic. A 3B model should not be expected to emulate it from a small synthetic corpus. Exact propagation belongs behind a deterministic simulator or measurement tool.

### 11. Mixed output contracts consume limited model capacity

Seven tasks use different schemas and numerical conventions. A small model must learn optics, routing, arithmetic, JSON structure, units, thresholds, and task-specific serialization simultaneously. Larger-data rounds improved schemas and categorical tasks first, which is consistent with capacity being spent on predictable formatting.

### 12. Training coverage is too small for continuous numerical generalization

The simulator defines a continuous, multi-parameter response surface. Thousands of records remain sparse relative to the combinations of setup parameters, actions, image conditions, and response scales. Adding similar examples alone is unlikely to solve this; the representation and objective must change.

## Corrective strategy

1. Separate `sensor_tolerance_pass` from `quantitative_skill` in evaluation.
2. Require improvement over zero and fixed-gain baselines.
3. Score per-component sign for materially changing outputs.
4. Move exact propagation, image measurement, thresholding, and exhaustive search into deterministic tools.
5. Have the LLM select tools and compact JSON paths/semantic roles; a deterministic adapter resolves paths into arrays.
6. Supervise tool-result interpretation separately from tool-call construction.
7. Preserve setup, causal, diagnosis, and counterfactual anchors during focused training.
8. Increase visual evaluation only after deterministic image measurements are available, so image perception and numerical physics can be scored separately.

## Experiment update

V7 replaced literal arrays with compact source-path mappings. On its first balanced 60-record generation panel it achieved 1.000 tool choice, 1.000 exact source mapping, and 1.000 mapped execution while preserving a 0.693 seven-task macro on a balanced public panel. Its long interpretation contract still failed through schema hallucination and candidate-table copying.

V7.1 therefore moved final numerical materialization into the deterministic adapter and reduced the LLM interpretation target to the required status and intermediate evidence indices/direction sets. The first balanced 60-record panel reached 1.000 on every exact routing, mapping, interpretation, pair, and end-to-end metric. Full public-development evaluation is in progress; confirmation remains sealed.
