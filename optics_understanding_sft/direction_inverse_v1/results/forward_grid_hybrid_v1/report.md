# Joint quantitative forward model

A compact local network predicts five numerical beam changes and five direction classes without simulator access at inference. A record succeeds only when all five numerical errors are inside their declared sensor tolerances.

| Split | Strict all-five | Zero strict | Tolerance MAE | Skill over zero | Direction macro-F1 |
|---|---:|---:|---:|---:|---:|
| val | 0.283 | 0.157 | 0.640 | 0.240 | 0.570 |
| eval_iid | 0.265 | 0.137 | 0.822 | 0.227 | 0.537 |
| eval_ood | 0.202 | 0.123 | 0.950 | 0.188 | 0.551 |
