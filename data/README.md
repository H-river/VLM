# Data policy

This repository is simulation-only unless an individual manifest proves a
different origin. Full generated datasets are local artifacts and are ignored
by Git. Commit only schemas, manifests, licenses/redistribution decisions, and
tiny synthetic fixtures whose provenance is explicit.

## Retained local data families

- Continuous v12 transition data and controller suites under `runs/`.
- Specialist measurement/direction/forward/inverse datasets in local package
  data roots and historical archives.
- Saturation and reflection counterfactual datasets used by the visual
  benchmark.
- Qwen supervisor train/development/frozen manifests, with generated JSONL and
  images retained locally.
- Candidate transition-comparison data under `artifacts/transition_pilot_*`.

Use `manifests/dataset_manifest.template.json` for external storage. Hash the
exact file or deterministic tree representation; do not put a placeholder hash
into a scientific report. `examples/synthetic_transition_v1.json` is a tiny
contract fixture, not a benchmark sample.

