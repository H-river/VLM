# Qwen Orchestration Evaluation Report

Date: 2026-07-25

## Outcome

Training and all evaluation allowed before opening the sealed test sets are
complete. No Stage-2 checkpoint is promoted.

The two full-validation candidates both missed the same three frozen gates:
JSON-schema validity, registry-valid ready calls, and image-role accuracy.
Because full validation did not produce a promotable checkpoint, no metrics
were computed on the 3,800 sealed records and they were not used for selection.

Checkpoint 1000 passed the separate end-to-end task-quality gate, but that does
not override the failed orchestration gates.

## Evaluated system

Input:

```text
natural-language request + zero, one, or two beam images
```

Qwen output:

```text
status + task type + registered route + canonical numerical arguments
+ image-role bindings + missing-input information
```

Evaluation dataflow:

```text
validation request
-> Qwen checkpoint
-> optional normalization of non-executing decisions only
-> JSON-schema validator
-> registered-route validator
-> frozen specialist, only for valid ready decisions
-> deterministic structured answer
-> comparison with the canonical decision and frozen oracle execution
```

The normalizer cannot alter any `ready` decision. It only canonicalizes
`needs_clarification` and `unsupported` decisions, which do not execute a
specialist.

## Data and denominators

The complete dataset contains 15,400 records in 1,900 disjoint physical groups.

| Split | Records | Evaluation use |
|---|---:|---|
| Training | 10,000 | Adapter training |
| Validation | 1,600 | Checkpoint selection and end-to-end validation |
| IID sealed test | 1,600 | Not evaluated |
| OOD-language sealed test | 1,000 | Not evaluated |
| Visual-stress sealed test | 1,200 | Not evaluated |

The 1,600 validation records contain:

- 1,050 executable requests: 150 for each of seven routes;
- 350 requests that require clarification;
- 200 unsupported requests;
- 600 executable requests that contain images;
- 2,700 required argument groups;
- 19,200 numerical values whose field paths also specify their units.

## Full-validation orchestration results

An exact rate is the number of correct records divided by the stated
denominator. A gate passes only when its full-validation rate reaches the
frozen minimum.

| Metric | Minimum | Checkpoint 600 | Checkpoint 1000 |
|---|---:|---:|---:|
| JSON-schema-valid decisions | 99.00% | 98.50% = 1,576/1,600 | 98.8125% = 1,581/1,600 |
| Registry-valid ready calls | 99.00% | 98.4762% = 1,034/1,050 | 98.5714% = 1,035/1,050 |
| Correct route on ready requests | 95.00% | 98.6667% = 1,036/1,050 | 98.7619% = 1,037/1,050 |
| Exact required argument groups | 97.00% | 98.00% = 2,646/2,700 | 98.1481% = 2,650/2,700 |
| Exact numerical value and unit path | 95.00% | 98.2396% = 18,862/19,200 | 98.3021% = 18,874/19,200 |
| Exact image-role binding | 98.00% | 97.8333% = 587/600 | 97.8333% = 587/600 |
| Correct clarification status | 95.00% | 97.7143% = 342/350 | 98.8571% = 346/350 |
| Correct unsupported status | 95.00% | 100.00% = 200/200 | 100.00% = 200/200 |
| Unsupported request incorrectly marked ready | At most 1.00% | 0/200 | 0/200 |

Checkpoint 1000 was closest to the frozen thresholds:

- schema validity was 3 records below the required 1,584/1,600;
- registry-valid ready calls were 5 records below the required 1,040/1,050;
- image-role accuracy was 1 record below the required 588/600.

The formal full-selection result is `selected_checkpoint: null`.

The safe non-executing normalizer was applied to 68 checkpoint-600 outputs and
73 checkpoint-1000 outputs. It changed zero executable `ready` decisions.

## Clarification-content audit

The promotion gate measures whether Qwen correctly refuses execution and asks
for clarification. Checkpoint 1000 did this for 346/350 records.

A stricter, non-gating audit compared the content with the single canonical
training target:

- exact missing-field list: 11/350;
- exact clarification-question wording: 0/350;
- both exact: 0/350.

This does not mean all 346 questions were unusable; exact wording rejects every
paraphrase. It does show that clarification content is not reliably
canonicalized and should be improved before deployment.

## End-to-end results for checkpoint 1000

The end-to-end evaluator uses the 1,600 validation records. An oracle means the
same frozen specialist is called with the canonical route and canonical
arguments. Loss is `oracle score - orchestrated score`, with negative values
clamped to zero.

| Metric | Orchestrated system | Oracle | Absolute loss |
|---|---:|---:|---:|
| Valid execution of ready requests | 98.5714% = 1,035/1,050 | Not applicable | Pass, minimum 95% |
| Direction macro-F1 | 100.00% | 100.00% | 0 percentage points |
| Forward strict-all-five success | 98.00% = 294/300 | 100.00% = 300/300 | 2 percentage points |
| Inverse target reached | 35.3333% = 106/300 | 37.3333% = 112/300 | 2 percentage points |
| Visual measurement strict-all-five success | 76.6667% = 115/150 | 76.6667% = 115/150 | 0 percentage points |

Definitions:

- Direction macro-F1 evaluates `decrease`, `no_change`, and `increase` for each
  of five beam quantities, then averages the 15 class-and-quantity scores.
- Forward strict-all-five success requires all five predicted beam changes to
  be inside their tolerances in the same record.
- Inverse target reached requires centroid distance at most 0.5 pixels,
  both width errors at most 1 pixel, and peak-intensity relative error at most
  2%.
- Visual measurement strict-all-five success requires both centroid errors at
  most 1 pixel, both width errors at most 2 pixels, and peak-intensity relative
  error at most 5%.

The maximum task loss was exactly 2 percentage points, which passes the
`<= 2`-percentage-point gate. The mean loss across the four task metrics was
1 percentage point. There were zero silent executions on missing-input
requests and zero simulator calls during inference. The 600 simulator calls in
the report are private evaluation-only replays: two calls for each of 300
inverse-control records.

## Integrity and resource audit

- Frozen baseline: 31 files and 6 directories verified by SHA-256 and size.
- Registry/scaffold: 7 routes and 4 task types verified.
- Dataset: 15,400 records, 1,900 groups, and 3,822 checksummed files verified.
  This integrity check did not generate predictions or metrics on sealed data.
- Checkpoints 600 and 1000: 696 adapter tensors each, consisting of 504
  language tensors and 192 visual tensors, with no unexpected non-adapter
  tensors.
- Regression tests: 24/24 passed.
- Python compilation audit: passed.
- Evaluation used conservative CPU settings, at most eight CPU cores, two
  numerical-library threads, and one compilation worker. No second system
  freeze occurred after these limits were applied.

## Final decision

The evaluation process reached a valid no-promotion result. Qwen demonstrates
strong in-domain routing and numerical copying, and the combined system passes
the end-to-end task-quality gate on validation. It is not yet eligible for
production or sealed-test evaluation because its full-validation orchestration
contract is still below the frozen thresholds.
