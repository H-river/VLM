# Joint quantitative forward model

A compact local network predicts five numerical beam changes and five direction classes without simulator access at inference. A record succeeds only when all five numerical errors are inside their declared sensor tolerances.

| Split | Strict all-five | Tolerance MAE | Skill over zero | Direction macro-F1 |
|---|---:|---:|---:|---:|
| val | 0.228 | 0.989 | 0.198 | 0.598 |
| eval_iid | 0.037 | 1.698 | 0.098 | 0.536 |
| eval_ood | 0.200 | 1.175 | 0.154 | 0.586 |
