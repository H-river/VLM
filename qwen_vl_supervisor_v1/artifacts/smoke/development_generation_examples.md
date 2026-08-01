# Development-only generation examples

These are complete raw continuations from the authoritative deterministic
engineering smoke. The adapter is
`smoke_qwen25vl_3b_midresume_reference_deterministic_pinned/final_adapter`
(`adapter_model.safetensors` SHA-256
`743789b04ad289bdbf554be14e80dbc6385339e45d3c8d4248930735133f2e54`).
No JSON substring extraction, repair, or normalization was applied. These are
development examples only, not frozen-test evidence or a scientific result.
The source prediction file is
`predictions_authoritative_d139_pinned_dev12_seed_2026080101.jsonl` (SHA-256
`1c519846bb26aa49e300e58d5c6122774e060055b39ef5ea2416608b1a7a4519`).

## Correct nominal example

Sample: `qvlsup1_c6dce3116581cf380750c810`

Target and raw continuation:

```json
{"diagnosis":"nominal","measurement_policy":"standard","supervisor_action":"execute"}
```

## Correct saturation example

Sample: `qvlsup1_1e572b62b9b798952fad503a`

Target and raw continuation:

```json
{"diagnosis":"sensor_saturation","measurement_policy":"lower_exposure_reacquire","supervisor_action":"reacquire"}
```

## Correct width-relative reflection example

Sample: `qvlsup1_b8639d866d3dd6c10139a3ed`

Target and raw continuation:

```json
{"diagnosis":"secondary_reflection","measurement_policy":"primary_spot","supervisor_action":"switch_measurement"}
```

## Reflection error retained for audit

Sample: `qvlsup1_5f26b4f4c2d93922825af0a7`

Target:

```json
{"diagnosis":"secondary_reflection","measurement_policy":"primary_spot","supervisor_action":"switch_measurement"}
```

Raw continuation:

```json
{"diagnosis":"sensor_saturation","measurement_policy":"lower_exposure_reacquire","supervisor_action":"reacquire"}
```

## Nominal error retained for audit

Sample: `qvlsup1_823e848780bb7939db764a01`

Target:

```json
{"diagnosis":"nominal","measurement_policy":"standard","supervisor_action":"execute"}
```

Raw continuation:

```json
{"diagnosis":"sensor_saturation","measurement_policy":"lower_exposure_reacquire","supervisor_action":"reacquire"}
```

Across all 12 balanced development examples, strict whole-string parsing
accepted 12/12 outputs. The predicted diagnosis distribution was 2 nominal,
8 sensor saturation, and 2 secondary reflection. Recall was 0.50 nominal,
1.00 saturation, and 0.50 reflection. No accuracy threshold applies to this
approximately 20-step engineering smoke.
