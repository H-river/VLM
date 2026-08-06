# Visual Counterfactual Feasibility Pilot

## Result: RED

This is candidate-only feasibility evidence. Frozen evaluation remains disabled; no controller/checkpoint was modified, no CEM rollout was started, and no result establishes closed-loop improvement.

## Model and runtime

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct` revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Adapter: `supervisor_v1_1_candidate/artifacts/training/full_seed_2026080101/final_adapter` SHA-256 `acf6ba79d75d933586e8b23ee3884450fd59d6ed629619953499eae5487e8e3d`.
- Selection rationale: No single deployed adapter is declared in the candidate protocol; fixed full-modality seed 2026080101 (the first predeclared full-model seed) is used without result-based selection.
- Decoding: greedy (`do_sample=false`, `temperature=0`, `num_beams=1`, `max_new_tokens=256`); 36 requests were deterministically shuffled with seed `2026080301`.
- Smoke interface checks: two non-formal requests. The first exposed nested-JSON/type errors and is retained unmodified as `smoke_outputs_prompt_v0.jsonl`; the prompt was strengthened before the second smoke, which strictly parsed. The final prompt was then frozen before all 36 formal requests.
- Git commit: `e6f18d4d5f513f912e73317aaaa66fdc3c479edf`. Working tree at reporting time:

```text
?? artifacts/qwen_controller_overnight_v1/
?? artifacts/qwen_h1_selector_v1/
?? artifacts/reduced_h1_rescue_v3/
?? artifacts/visual_counterfactual_pilot/
?? qwen_h1_meta_v0_candidate/
?? qwen_vl_supervisor_v1/artifacts/pilot/
?? qwen_vl_supervisor_v1/artifacts/training/pilot_qwen25vl_3b_full96_dev36_seed_2026080101_200step/
?? qwen_vl_supervisor_v1/configs/training_pilot_local_200step.yaml
?? supervisor_v1_1_candidate/
```

## Data, frozen inputs, and image construction

- Source: `qwen_h1_meta_v0_candidate/data/generated_v1/manifest_dev.jsonl` (SHA-256 `10e89b4fa545612d4b7f1a19723aac1b74498a2809deca3470f9567f829f06ef`), candidate-dev only; 12/24 records were admitted after deterministic normal/single-peak checks. Seven clear setups are represented, with alternate target records used to reach twelve.
- The source retained only 8-bit `L` PNGs, not raw linear intensity arrays. Images were therefore interpreted as linear grayscale `PNG/255`; this is a material limitation. All output images are 1024×1024 `L` PNG at the unchanged 0–255 visualization scale.
- Each record froze its five-dimensional state, target, actuator state/bounds, text context and **16** probes from the unchanged numeric forward ensemble (`runs/overnight_v12_semantics_20260731_002709/models/lc_128g_v2/continuous_forward_v12_128g.pt`, SHA-256 `d9b30627c80817f6ecade1959d8cc9e91e7a9de9cc51485153fdbfaa173aca2e`). No simulator or CEM was used for these probes.
- Exact cross-condition equality is checked via the per-record `numeric_context_sha256` recorded in `manifest.json`; only `image_sha256` differs. The shuffled condition uses a category derangement.

## Frozen prompt

```text
You are a visual reliability specialist immediately before a frozen one-step H1 controller. Inspect the beam image and the frozen numerical context. The numerical context is fixed for this request and may not describe image-only artifacts; do not infer record identity, source paths, experiment labels, future outcomes, continuous actions, or hidden setup variables. Return exactly one JSON object and no other text. Do not provide chain-of-thought. The JSON keys must appear exactly in this order: visual_abnormality, visual_evidence, measurement_reliability, forward_model_reliability, controller_plan, reason_code.

Use only these values:
- visual_abnormality: normal, edge_clipping, saturation, double_peak_or_ghost, asymmetry_or_tail, uncertain
- visual_evidence: a nonempty JSON array of short observable image features
- measurement_reliability.centroid, .width, .intensity: reliable, uncertain, unreliable
- forward_model_reliability: reliable, uncertain, unreliable
- controller_plan.mode: normal_control, conservative_control, reject_or_remeasure
- controller_plan.bound_scale: exactly 1.0, 0.35, or 0.0
- controller_plan.objective_preset: standard, measurement_uncertain, hold
- reason_code: normal_visual_state, edge_clipping_bias, sensor_saturation, multimodal_beam, shape_ood, insufficient_visual_evidence

Interpret normal visual state as reliable measurement and forward model with normal_control, 1.0, standard. For visible edge clipping or saturation, prefer reject_or_remeasure with 0.0, hold. For visible double peaks/ghosts or one-sided tails, make the forward model uncertain or unreliable and use conservative_control with 0.35, measurement_uncertain, unless the image warrants reject_or_remeasure.

Structural template only; replace all values based on the image. `measurement_reliability` MUST be an object, never a string. `controller_plan.bound_scale` MUST be an unquoted JSON number, never a string:
{"visual_abnormality":"normal","visual_evidence":["single centered peak"],"measurement_reliability":{"centroid":"reliable","width":"reliable","intensity":"reliable"},"forward_model_reliability":"reliable","controller_plan":{"mode":"normal_control","bound_scale":1.0,"objective_preset":"standard"},"reason_code":"normal_visual_state"}

```

## Pre-frozen expected policy

| Image state | Reliability expectation | Forward-model expectation | Allowed recommendation |
|---|---|---|---|
| normal | centroid/width/intensity reliable | reliable | normal_control, 1.0, standard |
| edge_clipping | centroid and width uncertain or unreliable | uncertain or unreliable | reject_or_remeasure, 0.0, hold |
| saturation | intensity unreliable | uncertain or unreliable | reject_or_remeasure, 0.0, hold |
| double_peak_or_ghost | all three measurement fields uncertain or unreliable | uncertain or unreliable | conservative_control, 0.35, measurement_uncertain; or reject_or_remeasure |
| asymmetry_or_tail | centroid and width uncertain or unreliable | uncertain or unreliable | conservative_control, 0.35, measurement_uncertain |

These rules were written before formal inference and are encoded in `manifest.json`; they were not retuned from outputs.

## Metrics

- JSON validity: **36/36 (100.0%)**.
- Clean normal accuracy: **9/12 (75.0%)**; false positives: **3/12**.
- Matched abnormal classification: **6/12 (50.0%)**; macro-F1 **0.375**.
- Measurement reliability (clean + matched images, field-level): **60/72 (83.3%)**.
- Controller-plan correctness (clean + matched images): **19/24 (79.2%)**.
- Matched-pair expected decision flip: **7/12 (58.3%)**.
- Shuffled image-follow accuracy: **6/12 (50.0%)**; original-record-reference accuracy **3/12 (25.0%)**; decline **25.0pp**.
- Shuffled controller plan differs from the matched-abnormal plan: **7/12**.

Wilson 95% intervals are included in `metrics.json`; n=12 is small, so they are feasibility context only.

## Did the image change the recommendation?

Partially. With identical numerical context, **7/12** clean→matched pairs made the expected plan change, and **6/12** shuffled cases named the actually injected image category versus **3/12** matching the receiver record's own category. The plan changed relative to its matched-abnormal counterpart in **7/12** shuffled cases. This is evidence that image content affected some Qwen recommendations, but it is below the preregistered reliability threshold and is not robust enough for a positive feasibility conclusion.

## Per-pair outcomes

| Sample | Matched image | Shuffled image | Clean output | Matched output | Shuffled output | Expected flip | Image-follow |
|---|---|---|---|---|---|---:|---:|
| s00 | edge_clipping | saturation | normal | edge_clipping | edge_clipping | True | False |
| s01 | saturation | double_peak_or_ghost | normal | edge_clipping | double_peak_or_ghost | True | True |
| s02 | double_peak_or_ghost | asymmetry_or_tail | edge_clipping | double_peak_or_ghost | edge_clipping | False | False |
| s03 | asymmetry_or_tail | edge_clipping | normal | edge_clipping | edge_clipping | False | True |
| s04 | edge_clipping | saturation | normal | edge_clipping | edge_clipping | True | False |
| s05 | saturation | double_peak_or_ghost | edge_clipping | edge_clipping | double_peak_or_ghost | False | True |
| s06 | double_peak_or_ghost | asymmetry_or_tail | normal | double_peak_or_ghost | edge_clipping | True | False |
| s07 | asymmetry_or_tail | edge_clipping | normal | edge_clipping | edge_clipping | False | True |
| s08 | edge_clipping | saturation | normal | edge_clipping | edge_clipping | True | False |
| s09 | saturation | double_peak_or_ghost | edge_clipping | edge_clipping | double_peak_or_ghost | False | True |
| s10 | double_peak_or_ghost | asymmetry_or_tail | normal | double_peak_or_ghost | edge_clipping | True | False |
| s11 | asymmetry_or_tail | edge_clipping | normal | edge_clipping | edge_clipping | True | True |

Full plan fields and raw/parsed output are in `pairwise_results.csv`, `raw_outputs.jsonl`, and `parsed_results.jsonl`.

## Category examples

- `edge_clipping`: success `s00`
- `saturation`: no successful matched example
  - failure `s05` predicted `edge_clipping`; raw output is retained in `raw_outputs.jsonl`.
- `double_peak_or_ghost`: success `s06`
- `asymmetry_or_tail`: no successful matched example
  - failure `s07` predicted `edge_clipping`; raw output is retained in `raw_outputs.jsonl`.

## Gate interpretation and limitations

Gate checks that failed: **abnormal_at_least_9_of_12, clean_fp_at_most_2, pair_flip_at_least_8_of_12, shuffle_follow_at_least_8_of_12**.

A GREEN result would mean only that this Qwen candidate can use image-visible phenomena absent from the five-dimensional numeric state to alter a **meta-controller recommendation**. The existing controller does not execute `reject_or_remeasure`, `conservative_control`, `bound_scale`, or `objective_preset` from this pilot; they are recommendation labels only. Synthetic image interventions are not real optical faults, and this pilot cannot demonstrate optical physics reasoning or strict-all-five closed-loop success.

Next step: if visual and shuffle gates are promising, preregister a separate candidate-only study with physically simulated/real anomalies and an audited integration path before any controller experiment. If they are not, keep the visual recommendation path off and investigate prompt/schema or data coverage without using frozen evaluation.
