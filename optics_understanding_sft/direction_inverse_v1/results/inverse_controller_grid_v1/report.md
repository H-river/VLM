# Numeric inverse controller

The controller evaluates all 81 declared actions with the learned forward model. Simulator-cached states are used only after prediction for private scoring.

| Split | Status macro-F1 | Exact selected action | Any acceptable action | Target reached | Zero-action baseline |
|---|---:|---:|---:|---:|---:|
| val | 0.380 | 0.083 | 0.250 | 0.250 | 0.000 |
| eval_iid | 0.390 | 0.000 | 0.133 | 0.133 | 0.000 |
| eval_ood | 0.355 | 0.096 | 0.250 | 0.250 | 0.000 |
