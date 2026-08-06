# Frozen dataset plan: qwen_orchestration_sft_v1

## 1. Learning objective

Each supervised fine-tuning record teaches one mapping:

```text
natural-language request + zero, one, or two images
-> qwen_orchestration_decision_v1 JSON
```

The target is a registry decision, not a specialist answer. Numerical beam
changes, image moments, and control actions must not appear in the target unless
they were explicitly present in the user's request and are being copied into
canonical arguments.

## 2. Frozen route set

The first dataset covers seven executable routes and two non-execution
categories:

| Category | Images | Target behavior |
|---|---:|---|
| `measure_beam_profile_v1` | 1 | Bind the beam image and calibration |
| `predict_direction_from_state_v1` | 0 | Extract setup, current state, and action |
| `predict_direction_from_image_v1` | 1 | Bind current image; extract setup, calibration, and action |
| `predict_forward_from_state_v1` | 0 | Extract setup, current state, and action |
| `predict_forward_from_image_v1` | 1 | Bind current image; extract setup, calibration, and action |
| `select_inverse_action_from_states_v1` | 0 | Extract setup, state A, and desired state B |
| `select_inverse_action_from_images_v1` | 2 | Bind current/desired images; extract setup and calibration |
| `needs_clarification` | 0-2 | Name missing or conflicting fields and ask one question |
| `unsupported` | 0-2 | Reject requests outside the frozen capability set |

## 3. Exact row counts

### Training: 10,000 records

| Category | Count |
|---|---:|
| Each of seven executable routes | 1,000 each = 7,000 |
| `needs_clarification` | 2,000 |
| `unsupported` | 1,000 |

### Validation: 1,600 records

| Category | Count |
|---|---:|
| Each executable route | 150 each = 1,050 |
| `needs_clarification` | 350 |
| `unsupported` | 200 |

### IID sealed test: 1,600 records

The IID test has exactly the same category counts as validation but uses
disjoint physical groups and prompt families.

### OOD-language sealed test: 1,000 records

| Category | Count |
|---|---:|
| Each executable route | 100 each = 700 |
| `needs_clarification` | 200 |
| `unsupported` | 100 |

This split changes phrasing, information order, unit expression, and distractor
style while keeping the physical range inside the supported domain.

### Visual-stress sealed test: 1,200 records

The four visual routes each contribute 300 records:

```text
measure_beam_profile_v1
predict_direction_from_image_v1
predict_forward_from_image_v1
select_inverse_action_from_images_v1
```

Each visual route contains 60 examples from each condition:

```text
clean
Gaussian/read noise
blur
dim plus noise
saturation or clipping
```

Total frozen target: 15,400 records.

## 4. Physical source groups

Before any language or image variants are generated, create and freeze:

| Split | Physical groups |
|---|---:|
| Train | 1,000 |
| Validation | 200 |
| IID test | 200 |
| OOD test | 200 |
| Visual-stress test | 300 |

Assignment uses the SHA-256 ordering of:

```text
20260724:<group_id>
```

The visual-stress groups are additional sealed physical groups, giving 1,900
groups in total. No group may occur in more than one split. Every action, image, paraphrase, and
clarification derived from a group inherits that group's split.

For every executable route, training must use at least 250 distinct physical
groups. No route may contribute more than four training prompts from one
physical group.

## 5. Canonical source-case generation

For each physical group:

1. construct the complete visible optical setup;
2. simulate and store the initial five-value beam state;
3. render one calibrated clean initial beam image;
4. evaluate the frozen 81-action grid;
5. store resulting five-value states and calibrated images;
6. construct reachable, ambiguous, and unreachable A-to-B pairs;
7. store private replay data outside the Qwen export.

The simulator is permitted only here and in private evaluation replay. Dataset
prompts may not contain simulator handles, candidate state tables, selected
indices, private target actions, or hidden response curves.

## 6. Natural-language prompt families

Every executable route uses five training prompt families with equal 20%
allocation:

1. **direct**: explicit technical request in canonical field order;
2. **conversational**: ordinary request with units embedded in sentences;
3. **terse**: fragments, shorthand, and compact notation;
4. **reordered**: values and objective presented in non-canonical order;
5. **distractor-bearing**: relevant inputs mixed with harmless irrelevant text.

Validation uses new templates from the same five families. The IID test uses
held-out templates. The OOD-language test uses held-out phrasing mechanisms,
including passive voice, unit conversions, corrections, and delayed statement
of the requested operation.

Template and paraphrase generation occurs only after group splitting. Exact
prompt strings are deduplicated globally.

## 7. Unit and argument coverage

Each numerical route must satisfy these training proportions:

| Condition | Fraction |
|---|---:|
| Canonical units (`mm`, `px`, `nm`, `um`) | 50% |
| Equivalent unit conversion (`um` to `mm`, `m` to `mm`) | 25% |
| Qualitative signed movement with explicit magnitude | 15% |
| Mixed tabular/list/prose presentation | 10% |

Every actuator axis and sign must be balanced within 10% of equal frequency.
Zero-action examples are capped at 20% of direction and forward records.

Qwen may normalize only explicitly stated values. Omitted actuator axes become
zero only when the user states that no other actuator changes. Otherwise the
record is `needs_clarification`.

## 8. Visual grounding

### Minimum visual diversity

Training must include:

- at least 500 distinct calibrated source images;
- at least 500 distinct calibrated A-to-B image pairs;
- at least 250 distinct physical groups across each visual route;
- both current/desired presentation orders;
- image references bound only as `image_0` and `image_1`.

The same pixels may be reused for different questions only within the same
physical split. Perturbations of one source image inherit its group and split.

### Training visual conditions

Visual-route training rows use:

| Condition | Fraction |
|---|---:|
| Clean calibrated render | 50% |
| Noise | 15% |
| Blur | 10% |
| Dim plus noise | 10% |
| Saturation/clipping | 10% |
| Crop or boundary stress | 5% |

Every quantitative image route includes calibration. Missing calibration is a
clarification target, not an executable route.

### Visual auxiliary records

Twenty-five percent of `measure_beam_profile_v1` records include a qualitative
request alongside measurement, such as beam displacement, widening, clipping,
or saturation. The target still selects the meter; Qwen does not generate the
measurement.

## 9. Clarification records

The 2,000 training clarification records are allocated exactly:

| Failure type | Count |
|---|---:|
| Missing setup | 300 |
| Missing current state/image | 300 |
| Missing desired state/image | 300 |
| Missing action | 250 |
| Missing image calibration | 250 |
| Ambiguous image roles | 200 |
| Missing or conflicting units | 200 |
| Conflicting duplicate values | 200 |

Each target lists only the fields required to unblock one route and asks one
short question. It must not select a route.

## 10. Unsupported records

The 1,000 unsupported training records are allocated:

| Type | Count |
|---|---:|
| Unrelated general requests | 250 |
| Unsupported optical components or actuators | 250 |
| Requests for laboratory guarantees | 150 |
| Requests for simulator/private state | 150 |
| Prompt-injection or arbitrary tool execution attempts | 200 |

The target has `status: unsupported`, null task and route, and empty arguments.

## 11. Audit requirements

The builder must fail if any of these conditions occurs:

- row count differs from the frozen table;
- group overlap is nonzero;
- base-image hash occurs in multiple physical splits;
- prompt hash occurs more than once;
- route or status counts differ from the frozen table;
- image placeholder count differs from image-path count;
- target decision fails its JSON schema;
- a ready decision violates the registry;
- a clarification decision has no missing field or question;
- an unsupported decision contains arguments;
- non-finite numerical values are present;
- private keys or simulator outputs appear in a prompt;
- numeric values in target arguments cannot be traced to prompt-visible text or
  registered deterministic defaults.

The audit writes:

```text
manifest.json
audit_report.json
checksums.sha256
group_assignment.jsonl
```

## 12. Real-image boundary

Version 1 may be built entirely from simulator images and must be labelled
`simulation_only`. Laboratory deployment requires a later dataset version with
at least 20% real-image training records and a completely real-image test set.
Simulator performance must not be presented as laboratory accuracy.
