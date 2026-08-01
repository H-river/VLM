# Numeric inverse controller

The controller evaluates all 81 declared actions with the learned forward model. Simulator-cached states are used only after prediction for private scoring.

| Split | Status macro-F1 | Exact selected action | Any acceptable action | Target reached | Zero-action baseline |
|---|---:|---:|---:|---:|---:|
| val | 0.390 | 0.000 | 0.208 | 0.208 | 0.000 |
| eval_iid | 0.316 | 0.067 | 0.300 | 0.300 | 0.000 |
| eval_ood | 0.274 | 0.058 | 0.192 | 0.192 | 0.000 |
