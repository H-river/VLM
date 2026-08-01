# Seven-task success and failure casebook

**Checkpoint:** `/home/jiamo/VLM_runs/qwen25vl_3b_qlora_corrective_v2_seed314`  
**Evaluation data:** independent `dev_v2`, 600 records  
**Rubric:** v2  
**Overall score:** 0.694 equal-task macro

This is the highest-scoring checkpoint evaluated across all seven original task families. It is the current best checkpoint by aggregate score, but it was **not promoted**: feasible-control recall was 0.10, feasible simulator success was 0.083, control status F1 was 0.425, and sufficiency status F1 was 0.532.

“Success” below normally means a recorded task score of 1.0 under rubric v2. “Failure” is a zero-score example when available, otherwise the lowest-scoring clear error for that task. Forward prediction is an explicit exception: manual inspection found no genuine success, even though five records received a score of 1.0. Those records are treated as rubric false positives rather than model successes.

### Visual coverage in this evaluation

Four of the seven task families can use images. In `dev_v2`, the 30 visual records are distributed as follows:

- causal effects: 4 visual and 56 text;
- forward prediction: 6 visual and 54 text;
- diagnosis: 11 visual and 109 text;
- counterfactual reasoning: 9 visual and 51 text.

Setup interpretation, information sufficiency, and constrained intervention are text-only in this evaluation. The later v5A/v6 evidence and tool-use experiments are also text-only.

## Summary

| Task | Success | Score | Failure | Score | Main lesson |
|---|---|---:|---|---:|---|
| Setup interpretation | `dev_v2_030004_setup_001` | 1.000 | `dev_v2_030046_setup_011` | 0.333 | Structure is strong; arithmetic can still drift. |
| Information sufficiency | `dev_v2_030001_suff_001` | 1.000 | `dev_v2_030000_suff_000` | The model often treats any hidden field as automatically insufficient. |
| Causal effects | `dev_v2_030051_causal_012` | 1.000 | `dev_v2_030035_causal_008` | It can read clear changes but misses thresholded directions. |
| Forward prediction | **None observed** | — | `dev_v2_030147_forward_058` and `dev_v2_030012_forward_003` | All five nominal score-1.0 cases are rubric false positives. |
| Diagnosis | `dev_v2_030000_diagnosis_000` | 1.000 | `dev_v2_030013_diagnosis_013` | Unique cases work, but the model can overpredict ambiguity. |
| Constrained intervention | `dev_v2_030004_control_004` | 1.000 | `dev_v2_030000_control_000` | Infeasibility certification is much stronger than finding feasible actions. |
| Counterfactual reasoning | `dev_v2_030009_cf_002` | 1.000 | `dev_v2_030000_cf_000` | Direction/difference scoring can hide bad absolute response predictions. |

## 1. Setup interpretation and units

### Success — exact setup extraction

**Example:** `dev_v2_030004_setup_001` — score 1.000

Visible facts:

- focal length: 103.062458 mm;
- source-to-lens distance: 188.090459 mm;
- lens-to-camera distance: 177.337178 mm;
- adjustable axes: lens x/y and camera x/y.

Ground truth and model prediction were identical:

```json
{
  "status": "answerable",
  "answer": {
    "component_order": ["gaussian_source", "thin_lens", "camera_sensor"],
    "lens_focal_length_m": 0.103062,
    "total_source_to_sensor_mm": 365.4276,
    "adjustable_parameters": ["lens_x", "lens_y", "camera_x", "camera_y"]
  }
}
```

**Why it succeeded:** The model identified the component order and actuator roles, converted millimetres to metres correctly, and added the two propagation distances without error.

### Failure — incorrect distance arithmetic

**Example:** `dev_v2_030046_setup_011` — score 0.333

Relevant visible values were 227.413196 mm from source to lens and 175.836323 mm from lens to camera.

| Field | Ground truth | Model |
|---|---:|---:|
| Focal length | 0.098361 m | 0.098360 m |
| Total distance | 403.2495 mm | 393.2500 mm |

The component order and adjustable axes were correct, but the total distance was wrong by 9.9995 mm and the rounded focal length missed the evaluator’s numerical tolerance.

**Why it failed:** This is not an optics-concept failure; it is a precision and arithmetic failure inside an otherwise correct structural interpretation.

## 2. Information sufficiency

### Success — correctly recognizes non-identifiability

**Example:** `dev_v2_030001_suff_001` — score 1.000

The candidate action omitted `lens_x_delta_mm`. Five compatible values were allowed: -0.04, -0.06, -0.05, -0.03, and -0.02 mm. Simulator replay showed that the requested centroid direction was not invariant across compatible completions.

Ground truth and model prediction were identical:

```json
{
  "status": "insufficient_information",
  "answer": {
    "missing_fields": ["lens_x_delta_mm"],
    "nonidentifiable_output": "centroid_x_direction"
  }
}
```

**Why it succeeded:** It linked the missing actuator value to a genuinely non-identifiable output and returned the correct missing field.

### Failure — assumes a hidden field always makes the answer impossible

**Example:** `dev_v2_030000_suff_000` — score 0.000

The same actuator field was hidden, but all compatible completions produced the same `increase` direction. The correct result was therefore:

```json
{"status":"answerable","answer":{"centroid_x_direction":"increase","missing_fields":[]}}
```

The model instead returned:

```json
{"status":"insufficient_information","answer":{"missing_fields":["lens_x_delta_mm"],"nonidentifiable_output":"centroid_x_direction"}}
```

**Why it failed:** The model used a surface rule—“a field is missing, therefore insufficient”—instead of checking whether every compatible completion agrees on the requested output.

## 3. Causal effects

### Success — reads multiple thresholded effects correctly

**Example:** `dev_v2_030051_causal_012` — score 1.000

A -0.035 mm lens-x intervention changed the observation as follows:

| Measurement | Before | After | Correct class |
|---|---:|---:|---|
| Centroid x | 508.7132 px | 505.0341 px | decrease |
| Centroid y | 504.4345 px | 504.4345 px | no change |
| Peak intensity | 10.0150 | 12.0286 | increase |
| Sigma x | 119.6197 px | 120.5835 px | no change |
| Sigma y | 119.3753 px | 119.3753 px | no change |

The model returned all five labels exactly.

**Why it succeeded:** The centroid and intensity changes were large enough to cross their respective thresholds, while the width changes stayed below the 2 px threshold.

### Failure — confuses a clear centroid shift with a negligible intensity change

**Example:** `dev_v2_030035_causal_008` — score 0.600

For a -0.035 mm lens-x intervention, centroid x increased from 505.8008 to 508.5611 px. Peak intensity changed only from 8.5286 to 8.4992, which is below the 5% threshold.

| Measurement | Ground truth | Model |
|---|---|---|
| Centroid x | increase | no change |
| Centroid y | no change | no change |
| Peak intensity | no change | decrease |
| Sigma x | no change | no change |
| Sigma y | no change | no change |

**Why it failed:** It missed the large 2.76 px centroid motion and overreacted to a very small intensity decrease. Three of five effects remained correct, hence the 0.600 score.

## 4. Forward prediction

### No genuine success was observed

Rubric v2 assigns a score of 1.0 when the centroid vector is within 2 px, both widths are within 2 px, and peak intensity is within 5%. Those tolerances are appropriate for sensor relevance, but they allow a near-zero prediction to pass whenever the true change is small.

The checkpoint has five nominal score-1.0 forward records. Manual inspection found that none predicts the complete change credibly. Across all 58 parseable forward outputs, peak change is exactly zero in 48 cases, sigma-x change is zero in 48, and sigma-y change is zero in 51. Thirty outputs keep every predicted change within 0.05 of zero.

Therefore, the casebook records **no forward-prediction success** for this checkpoint.

### Rubric false positive — only one component is close

**Example:** `dev_v2_030147_forward_058` — score 1.000

Ground truth change:

```json
{"centroid_x_px":0.3846,"centroid_y_px":-0.1364,"peak_intensity":0.2172,"sigma_x_px":-0.6,"sigma_y_px":-0.1298}
```

Model-predicted change:

```json
{"centroid_x_px":0.4537,"centroid_y_px":0.4562,"peak_intensity":0.0009,"sigma_x_px":0.0,"sigma_y_px":0.0}
```

Only centroid x is close: +0.4537 px predicted versus +0.3846 px ground truth. Centroid y has the wrong sign, peak intensity is nearly zero instead of +0.2172, and both width changes are returned as zero.

**Why the rubric gave it 1.0:** The combined centroid error is only 0.597 px, width errors are below 0.6 px, and the peak error is below 5%. The rubric does not require correct per-component signs when changes are small.

**Why this report classifies it as a failure:** Four of the five predicted changes are qualitatively wrong or collapsed to zero. Passing aggregate tolerances is not enough to claim that the model understood the forward response.

### Failure — wrong direction and magnitude for a large transition

**Example:** `dev_v2_030012_forward_003` — score 0.000

| Change | Ground truth | Model |
|---|---:|---:|
| Centroid x | -5.5267 px | +0.0270 px |
| Centroid y | +1.0348 px | -1.8204 px |
| Peak intensity | -0.7788 | 0.0000 |
| Sigma x | +0.4075 px | 0.0000 px |
| Sigma y | -2.0858 px | 0.0000 px |

The predicted after-centroid was 6.247 px from the ground truth, and the peak and width checks also failed.

**Why it failed:** Once the true response moved outside the small-change tolerance region, the checkpoint’s near-copy/default prediction no longer worked.

## 5. Diagnosis and ambiguity

### Success — identifies a unique cause

**Example:** `dev_v2_030000_diagnosis_000` — score 1.000

Four candidate lens-x interventions were supplied. Only `primary_lens_intervention` reproduced the observed transition within the declared centroid and width tolerances.

Ground truth and model prediction:

```json
{"status":"unique","answer":{"plausible_causes":["primary_lens_intervention"]}}
```

**Why it succeeded:** It selected the only matching candidate and correctly classified the diagnosis as unique.

### Failure — overpredicts ambiguity

**Example:** `dev_v2_030013_diagnosis_013` — score 0.200

Ground truth:

```json
{"status":"unique","answer":{"plausible_causes":["primary_lens_intervention"]}}
```

Model prediction:

```json
{
  "status": "ambiguous",
  "answer": {
    "plausible_causes": [
      "primary_lens_intervention",
      "alternative_intervention_1",
      "alternative_intervention_2",
      "alternative_intervention_3"
    ]
  }
}
```

**Why it failed:** It retained every candidate instead of applying the matching tolerances. Status was wrong and plausible-cause F1 fell to 0.4, producing a task score of 0.2.

## 6. Constrained intervention

### Success — correctly certifies infeasibility

**Example:** `dev_v2_030004_control_004` — score 1.000

The lens-x grid ranged from -0.05556 to +0.05556 mm with a 2 px success tolerance. Exhaustive replay found no successful action; the best achievable residual was 2.5096 px.

| Result | Ground truth | Model |
|---|---|---|
| Status | infeasible within limits | infeasible within limits |
| Control plan | null | null |
| Best residual | 2.5096 px | 2.5101 px |

**Why it succeeded:** It selected the correct infeasible branch and estimated the best residual within tolerance.

### Failure — rejects an exactly feasible action

**Example:** `dev_v2_030000_control_000` — score 0.000

The correct plan was a +0.008262 mm lens-x move, which achieved a cached residual of 0.0 px.

Ground truth:

```json
{
  "status": "feasible",
  "answer": {
    "control_plan": {"lens_x_delta_mm":0.008262,"lens_y_delta_mm":0.0,"camera_x_delta_mm":0.0,"camera_y_delta_mm":0.0},
    "expected_residual_px": 0.0
  }
}
```

Model prediction:

```json
{"status":"infeasible_within_limits","answer":{"control_plan":null,"best_achievable_residual_px":2.3212}}
```

**Why it failed:** The checkpoint’s strong infeasibility bias prevented it from selecting a valid grid action. This example directly reflects its 0.10 feasible recall.

## 7. Counterfactual reasoning

### Scored success — correct direction and tiny between-scenario difference

**Example:** `dev_v2_030009_cf_002` — score 1.000

The two scenarios differed only in `source_to_lens_mm`. Under the same +0.04 mm lens-x action, both centroid responses moved in the same direction. Their ground-truth centroid-x response difference was only 0.0055 px.

The model correctly returned:

- changed parameter: `source_to_lens_mm`;
- centroid direction preserved: `true`;
- a near-zero response difference.

However, the model predicted individual centroid responses near zero, whereas both ground-truth responses were approximately +11.42 px.

**Why it scored as a success:** Rubric v2 scores the changed parameter, preserved direction, and numerical difference between responses. The near-zero predicted difference was within tolerance, even though the two absolute response magnitudes were poor. This is another metric success that should not be treated as complete physical understanding.

### Failure — wrong direction relation and invalid response schema

**Example:** `dev_v2_030000_cf_000` — score 0.200, visual

Scenario A and B differed in lens focal length. The ground truth said the shared action did **not** preserve centroid direction:

```json
{
  "centroid_direction_preserved": false,
  "changed_parameter": "lens_focal_length_mm",
  "response_difference": {
    "centroid_x_px": 1.7415,
    "peak_intensity": -0.6123,
    "sigma_x_px": -3.6001
  }
}
```

The model said direction was preserved and used unsupported `x_shift`/`y_shift` fields instead of the required response schema. Only the changed-parameter identification was correct.

Visual inputs:

| Scenario A after | Scenario B after |
|---|---|
| ![Scenario A after](/home/jiamo/VLM/optics_understanding_sft/data/dev_v2/images/val/dev_v2_030000_cf_000_scenario_a_after.png) | ![Scenario B after](/home/jiamo/VLM/optics_understanding_sft/data/dev_v2/images/val/dev_v2_030000_cf_000_scenario_b_after.png) |

**Why it failed:** It missed the sign change between scenario responses and violated the output schema, so all numerical counterfactual checks failed.

## Overall interpretation

The checkpoint is strongest at structured setup extraction, causal classification, and diagnosis. Its main weaknesses are visible in the paired examples:

1. it substitutes simple priors for completion-set reasoning in sufficiency;
2. it strongly favors infeasible control decisions;
3. it often copies the current state for forward prediction;
4. it can receive full credit on small-difference counterfactuals without predicting either absolute response correctly;
5. numerical extraction and arithmetic remain less reliable than categorical structure.

Therefore, the 0.694 macro score represents real improvement but should not be reported as 69.4% laboratory competence. The strict promotion gates correctly prevented deployment.
