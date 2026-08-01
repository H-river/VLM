# INVALID FORMAL RUN

This run must not be used as a scientific accuracy result.

- Invalidated after completion during read-only postflight inspection.
- Root cause: the frozen Qwen execution environment did not contain SciPy.
- Exact error on every correct-route image/measurement baseline: `SpecialistError: ... ModuleNotFoundError: No module named 'scipy'`.
- Affected routes: measurement plus image direction, image forward, and image inverse (40 ready cases total).
- The run's files are retained unchanged as diagnostic evidence. No metric from this directory supersedes the replacement run.
- Recovery: install and pin SciPy in the Qwen environment, use a new run ID, rebuild the frozen manifest/config, run image-route smoke in that same environment, and rerun all cases from the beginning.

