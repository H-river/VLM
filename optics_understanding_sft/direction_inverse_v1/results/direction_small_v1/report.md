# Dedicated small direction classifier

A shared numerical encoder predicts five independent three-class direction heads. Training combines exactly balanced single-field records with natural-distribution all-field records. The simulator is unavailable at inference.

| Split | Model | Equal-field macro-F1 | All-five exact |
|---|---|---:|---:|
| val | majority | 0.247 | 0.189 |
| val | shared_mlp | 0.570 | 0.239 |
| eval_iid | majority | 0.215 | 0.056 |
| eval_iid | shared_mlp | 0.504 | 0.037 |
| eval_ood | majority | 0.240 | 0.167 |
| eval_ood | shared_mlp | 0.583 | 0.233 |
