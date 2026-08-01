# Control rebuild v5

This directory contains a numerical-only rebuild of the two weakest control
specialists:

1. five-value forward prediction;
2. numerical inverse target reaching.

All v1-v4 data, checkpoints, validation sets, Qwen artifacts, image models,
and visual-inverse artifacts remain unchanged.

## Forward v5

Forward v5 uses five independent histogram-gradient-boosting regressors. Each
head receives the existing 46-value engineered feature vector and predicts one
tolerance-normalized beam change. The five outputs are:

- horizontal centroid change;
- vertical centroid change;
- horizontal width change;
- vertical width change;
- peak-intensity change.

The feature vector contains the 12-value optical setup, five-value current
beam state, four-value action, and 25 explicit physical/action interaction
features. Training weights make one-, two-, three-, and four-component actions
contribute equally. The zero action is forced to return exactly five zeros.

## Inverse v5

Inverse v5 evaluates the fixed 81-action grid produced by forward v5. It uses:

- one binary tree model that estimates whether each candidate action truly
  reaches the target;
- one regression tree that corrects the residual cost computed from imperfect
  forward-v5 predictions;
- one three-class tree model that predicts `unique`, `ambiguous`, or
  `infeasible_within_limits`;
- the deterministic physical residual and minimum-movement tie-break.

Candidate training uses all positive actions and the closest incorrect
actions under forward-v5 predictions. Labels always come from simulator truth.
Half of training requests use paired empirical measurement errors, matching
the v4 numerical-inverse evaluation contract.

## Isolation

Default output locations are:

```text
/home/jiamo/VLM_data/control_rebuild_v5_numerical
/home/jiamo/VLM_runs/control_rebuild_v5_one_seed
```

No held-out test file is opened for training or checkpoint selection.

## Completed one-seed validation

The completed run uses seed `20260728`. The added dataset contains 6,000
independent setup groups and 486,000 transitions, split evenly across
`iid_expanded`, `ood_boundary`, `high_nonlinearity`, and `hard_interaction`.
Dataset integrity checks passed before training.

Forward v5 was trained on 850,500 transitions in total. Exact success means
all five predicted changes are simultaneously within their field-specific
tolerances.

| Forward validation split | v4 | v5 |
| --- | ---: | ---: |
| Old IID, exact all five | 33.36% | 57.95% |
| Difficult, exact all five | 27.06% | 48.92% |
| Difficult, four-component actions | 12.69% | 36.58% |

Inverse v5 was trained from 108,000 target requests and 2,283,363 selected
candidate-action examples. Target reached means that simulator replay of the
chosen action places all five physical beam values within their tolerances.

| Inverse validation split | Target reached | Status accuracy |
| --- | ---: | ---: |
| IID clean | 45.67% | 90.00% |
| IID measurement-augmented | 30.50% | 85.22% |
| Difficult clean | 67.08% | 94.94% |
| Difficult measurement-augmented | 51.33% | 83.00% |

The frozen 1,600-request system validation produced 49.67% end-to-end
forward exact-all-five accuracy and 47.33% numerical state-inverse target
reaching. These results are a validation candidate, not production approval.
Thirteen of seventeen declared engineering gates passed. The remaining misses
are difficult peak-intensity accuracy, IID inverse target reaching, and both
60% numerical-inverse system gates.

Primary outputs:

```text
/home/jiamo/VLM_runs/control_rebuild_v5_one_seed/forward_tree_v5_summary.json
/home/jiamo/VLM_runs/control_rebuild_v5_one_seed/inverse_tree_v5_summary.json
/home/jiamo/VLM_runs/control_rebuild_v5_one_seed/orchestrated_system_numerical_v5_validation.json
/home/jiamo/VLM_runs/control_rebuild_v5_one_seed/numerical_v5_audit.json
```
