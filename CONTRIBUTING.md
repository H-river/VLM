# Contributing

The project is paused, but reviewable fixes and documentation improvements can
be prepared privately.

## Scientific rules

- Preserve the distinction between `CONFIRMED`, `PROVISIONAL`,
  `CANDIDATE_ONLY`, `EXCLUDED`, `NOT_RUN`, and `UNKNOWN`.
- Never promote a candidate result because its metric is favorable.
- Do not describe Qwen as a validated numerical physics model.
- Keep simulation and real-world evidence separate.
- Do not change stored numerical results, labels, hashes, split assignments, or
  frozen/protected status without tracing the exact source evidence.
- Keep the continuous controller and supervisor contracts separate; supervisor
  outputs may not contain actuator values.

## Code and data rules

- Put new shared code under `src/optics_vla/` and keep thin adapters around
  hash-pinned historical modules.
- Use repository-relative paths, CLI options, config fields, or environment
  variables. Do not add personal absolute paths.
- Use explicit deterministic seeds and SHA-256 identities.
- Keep generated data/checkpoints outside Git and complete the manifest
  templates before sharing.
- Do not edit or delete historical reports merely to make the tree look clean.

## Validation

```bash
python -m pip install -e '.[test]'
optics-vla all-smoke
python scripts/reproduce/verify_published_table.py
python -m pytest
```

Report missing checkpoint, CUDA, dependency, or dataset prerequisites as
`BLOCKED` with the exact reason. Do not convert them to passes or silently use a
toy substitute.

