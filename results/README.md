# Result policy

Only compact, traceable summaries belong in Git. Raw predictions, rendered
media, tensorboard logs, and full episode trees stay in local `runs/` or
`artifacts/` roots and must be represented by manifests and hashes.

`published_tables/confirmed_results.csv` contains only `CONFIRMED` rows and is
checked against tracked source reports by:

```bash
python scripts/reproduce/verify_published_table.py
```

Candidate-only transition comparisons and counterfactual plan-selection
numbers are intentionally excluded from this confirmed table. They remain
documented as candidate-only in `docs/EXPERIMENT_STATUS.md`.

