# Protected specialist-improvement cycle — 2026-07-28

## Decision

The deployed specialist baseline remains frozen at:

`/home/jiamo/VLM_runs/physics_structured_rebuild_v9_one_seed/orchestrated_forward_selector_ensemble_v9_validation.json`

The final combined candidate passed its non-regression checks but did not
increase any strict whole-task count. Its artifacts remain experimental and
were not promoted.

## Frozen-data contract

- No new training setup, transition, image, or simulator-generated dataset was
  created.
- Training used the existing 2,000-request, 162,000-action inverse cache and
  the existing 3,000-request, 243,000-transition forward/direction cache.
- Protected validation and system validation were not used to fit model
  parameters.
- No held-out test file was opened.

## Quantitative results

### Inverse feasibility audit

- Existing state-inverse system requests: 150.
- Requests with at least one physically successful action among all 81 legal
  actions: 150/150 = 100%.
- Successful actions per request: median 4, mean 4.68, minimum 1, maximum 16.
- Requests with exactly one successful action: 19/150.
- Conclusion: the 58% state-inverse result is an action-ranking failure, not a
  physical-feasibility ceiling.

### Strict five-value forward correction

- Untouched internal split: 17,231/24,867 = 69.29% to
  17,637/24,867 = 70.93%.
- Protected calibrator retained only small output blends that improved both
  protected contracts.
- State-forward system replay: 93/150 = 62.00% before and after.
- Mean absolute error changed from 0.44461 to 0.44436.
- Decision: not promoted because strict all-five system accuracy did not
  increase.

### Boundary-aware direction correction

- Older protected grid: 12,862/24,300 = 52.93% to
  12,902/24,300 = 53.09%.
- Difficult protected grid: 11,730/24,300 = 48.27% to
  11,791/24,300 = 48.52%.
- State-direction system replay: 81/150 = 54.00% before and after.
- State-direction macro-F1: 0.73186 to 0.73326.
- Final 300-request direction macro-F1: 0.73115 to 0.73185.
- Decision: retained as an experimental protected overlay; not promoted
  because all-five accuracy did not increase.

### Group-balanced inverse success ranker

- Internal 400-request split, primary forward enumeration:
  268/400 to 271/400.
- Internal 400-request split, secondary forward enumeration:
  270/400 to 271/400.
- IID protected block: 282/900 to 284/900.
- Difficult protected block: 810/1,800 to 811/1,800.
- State-inverse system replay: 87/150 = 58.00% before and after.
- Three actions changed; no request changed its final success/failure status.
- Decision: retained as an experimental protected overlay; not promoted
  because physical target-reaching accuracy did not increase.

## Final 1,600-request system comparison

| Metric | Frozen baseline | Combined candidate | Change |
|---|---:|---:|---:|
| Direction, all five exact | 54.00% | 54.00% | 0 |
| Direction, equal-field macro-F1 | 0.73115 | 0.73185 | +0.00070 |
| Forward, end-to-end all five within tolerance | 60.00% | 60.00% | 0 |
| Forward, correctly routed all five within tolerance | 61.67% | 61.67% | 0 |
| Inverse, end-to-end physical target reached | 59.33% | 59.33% | 0 |
| Inverse, correctly routed physical target reached | 61.00% | 61.00% | 0 |
| Measurement, all five within tolerance | 76.67% | 76.67% | 0 |

Final candidate report:

`/home/jiamo/VLM_runs/physics_structured_rebuild_v9_one_seed/orchestrated_gated_cycle_v9_validation.json`

## Safety

All long or model-loading commands ran through the laptop safety wrapper on
CPU cores 0 and 8. The final full-system run took 319 seconds and observed:

- Maximum GPU memory: 1,050 MiB.
- Maximum GPU temperature: 60 C.
- Minimum available RAM: 6,729 MiB.
- Maximum swap use: 0.33 MiB.
- Safety stop reason: none.
