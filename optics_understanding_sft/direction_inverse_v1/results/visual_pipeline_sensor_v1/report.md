# Visual measurement and inverse-control pipeline

A deterministic moment-based image meter extracts calibrated beam states. The learned forward model then searches the 81-action grid; cached simulator states are used only for private scoring.

| Split | Image state all-five | Centroid-x MAE | Width-x MAE | Peak MAE | Inverse target reached |
|---|---:|---:|---:|---:|---:|
| val | 0.847 | 0.059 | 0.163 | 0.250 | 0.250 |
| eval_iid | 0.933 | 0.074 | 0.189 | 0.433 | 0.167 |
| eval_ood | 0.814 | 0.136 | 0.338 | 1.791 | 0.151 |
