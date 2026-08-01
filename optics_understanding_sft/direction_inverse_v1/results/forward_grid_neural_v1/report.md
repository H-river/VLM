# Joint quantitative forward model

A compact local network predicts five numerical beam changes and five direction classes without simulator access at inference. A record succeeds only when all five numerical errors are inside their declared sensor tolerances.

| Split | Strict all-five | Tolerance MAE | Skill over zero | Direction macro-F1 |
|---|---:|---:|---:|---:|
| val | 0.205 | 0.742 | 0.120 | 0.570 |
| eval_iid | 0.171 | 0.987 | 0.071 | 0.537 |
| eval_ood | 0.138 | 1.058 | 0.096 | 0.551 |
