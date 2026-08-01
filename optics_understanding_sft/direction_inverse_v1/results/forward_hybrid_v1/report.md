# Joint quantitative forward model

A compact local network predicts five numerical beam changes and five direction classes without simulator access at inference. A record succeeds only when all five numerical errors are inside their declared sensor tolerances.

| Split | Strict all-five | Zero strict | Tolerance MAE | Skill over zero | Direction macro-F1 |
|---|---:|---:|---:|---:|---:|
| val | 0.267 | 0.189 | 0.847 | 0.313 | 0.598 |
| eval_iid | 0.093 | 0.056 | 1.495 | 0.206 | 0.536 |
| eval_ood | 0.167 | 0.167 | 1.071 | 0.228 | 0.586 |
