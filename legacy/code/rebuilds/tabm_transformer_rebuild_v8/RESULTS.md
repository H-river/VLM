# TabM and Transformer controlled comparison

## Fixed experiment contract

- One seed: `20260801`.
- Forward and direction training: 10,500 complete optical-setup groups and
  850,500 transitions from the completed old, difficult, and v5-additional
  datasets.
- Forward and direction validation: 300 old groups and 300 difficult groups,
  each expanded to 24,300 transitions.
- Numerical inverse training: 108,000 requests, each with 81 candidate control
  actions. Of these requests, 80,000 have at least one ground-truth-valid
  action.
- Numerical inverse validation: 900 old clean, 900 old
  measurement-augmented, 1,800 difficult clean, and 1,800 difficult
  measurement-augmented requests.
- The partial v7 targeted-generation shards were not used.
- No held-out test file was opened. No deployed manifest or checkpoint was
  changed.

The forward/direction input is a 46-value engineered numerical vector. The
forward output is five predicted beam-state changes. The direction output is
five independent three-class predictions: decrease, no change, or increase.

The inverse input contains a 31-value request context and, for each of 81
actions, a 23-value candidate description derived from the frozen v5 forward
model. Its outputs are 81 ranking scores and one three-class reachability
status.

## Metric definitions

- **Forward all-five success:** all five numerical changes are within their
  respective error tolerances on the same transition.
- **Direction all-five correct:** all five direction classes are correct on the
  same transition.
- **Inverse feasible-target success:** among requests that have at least one
  ground-truth-valid action, the selected action is one of those valid actions.
- **Old:** the original in-distribution validation setup distribution.
- **Difficult:** the intentionally harder validation setup distribution.
- **Noisy:** a validation request whose measured current state has realistic
  measurement error added.

## Forward and direction results

| Model | Parameters | Train time | Forward old | Forward difficult | Direction old | Direction difficult |
|---|---:|---:|---:|---:|---:|---:|
| Existing v7 specialist | 825,236 | 65.03 s | **59.29%** | **50.21%** | **52.05%** | **47.58%** |
| TabM v8 | 258,592 | 81.17 s | 26.42% | 29.09% | 28.68% | 29.83% |
| Feature-token Transformer v8 | 516,756 | 390.74 s | 45.54% | 37.03% | 42.68% | 37.47% |

For transitions that change three or four controls, the same comparison is:

| Model | Forward old | Forward difficult | Direction old | Direction difficult |
|---|---:|---:|---:|---:|
| Existing v7 specialist | **53.54%** | **42.40%** | **45.94%** | **39.85%** |
| TabM v8 | 16.24% | 19.22% | 19.62% | 20.14% |
| Feature-token Transformer v8 | 37.51% | 27.40% | 35.30% | 28.92% |

Neither new direct forward/direction model is a replacement candidate.

## Numerical inverse results

| Model | Parameters | Train time | Old clean | Difficult clean | Old noisy | Difficult noisy |
|---|---:|---:|---:|---:|---:|---:|
| Existing v5 inverse | n/a | n/a | 45.67% | **67.08%** | 30.50% | 51.33% |
| TabM v8 | 120,528 | 57.24 s | 38.17% | 60.83% | 26.17% | 46.08% |
| Candidate Set Transformer v8 | 198,564 | 38.90 s | **48.33%** | 65.67% | **33.67%** | **53.42%** |

The Set Transformer changes relative to v5 are +2.67, -1.42, +3.17, and
+2.08 percentage points respectively. Its unweighted average across the four
blocks is 50.27%, versus 48.65% for v5: +1.63 percentage points. It is a
promising inverse candidate, but not a universal replacement because difficult
clean performance decreased.

## Safety and completion state

- The v7 targeted-data generator stopped at exactly 2,205/3,000 groups.
- Both generator and post-processing services are inactive and unlinked.
- All 2,205 completed shard files remain intact.
- All training jobs have exited.
- Six implementation tests pass, including exact agreement between the
  registered NumPy inverse features and the new Torch implementation.
