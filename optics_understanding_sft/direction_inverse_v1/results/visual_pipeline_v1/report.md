# Visual measurement and inverse-control pipeline

A deterministic moment-based image meter extracts calibrated beam states. The learned forward model then searches the 81-action grid; cached simulator states are used only for private scoring.

| Split | Image state all-five | Centroid-x MAE | Width-x MAE | Peak MAE | Inverse target reached |
|---|---:|---:|---:|---:|---:|
| val | 0.000 | 10.719 | 2.305 | 5.705 | 0.000 |
| eval_iid | 0.011 | 10.818 | 3.817 | 11.199 | 0.000 |
| eval_ood | 0.000 | 11.382 | 7.558 | 35.161 | 0.000 |
