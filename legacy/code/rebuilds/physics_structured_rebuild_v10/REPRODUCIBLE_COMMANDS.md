# V10 reproducible commands

Run from `/home/jiamo/VLM` with the `optical_sim` Conda Python. Long commands
must be wrapped by `physics_structured_rebuild_v9/safe_run.py` and pinned with
`taskset -c 0,8`; the commands below show the inner command for readability.

```bash
PY=/home/jiamo/miniconda3/envs/optical_sim/bin/python
DATA=/home/jiamo/VLM_data/physics_structured_rebuild_v10
RUN=/home/jiamo/VLM_runs/physics_structured_rebuild_v10_full
```

Smoke tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q physics_structured_rebuild_v10/tests
$PY physics_structured_rebuild_v10/generate_dataset.py --scale smoke --output-dir /home/jiamo/VLM_data/physics_structured_rebuild_v10_smoke --workers 2
$PY physics_structured_rebuild_v10/validate_dataset.py --data-dir /home/jiamo/VLM_data/physics_structured_rebuild_v10_smoke
for EXPERIMENT in A B C D; do
  $PY physics_structured_rebuild_v10/train_forward_ablation.py --experiment "$EXPERIMENT" --smoke --data-dir /home/jiamo/VLM_data/physics_structured_rebuild_v10_smoke --run-dir /home/jiamo/VLM_runs/physics_structured_rebuild_v10_smoke --device cuda
done
$PY physics_structured_rebuild_v10/evaluate_forward_ablation.py --smoke --data-dir /home/jiamo/VLM_data/physics_structured_rebuild_v10_smoke --run-dir /home/jiamo/VLM_runs/physics_structured_rebuild_v10_smoke --device cuda
$PY physics_structured_rebuild_v10/train_inverse_ranker.py --smoke --data-dir /home/jiamo/VLM_data/physics_structured_rebuild_v10_smoke --run-dir /home/jiamo/VLM_runs/physics_structured_rebuild_v10_smoke --device cuda
$PY physics_structured_rebuild_v10/train_visual_inverse.py --smoke --data-dir /home/jiamo/VLM_data/physics_structured_rebuild_v10_smoke --run-dir /home/jiamo/VLM_runs/physics_structured_rebuild_v10_smoke --device cuda
$PY physics_structured_rebuild_v10/train_direction_calibrator.py --smoke --data-dir /home/jiamo/VLM_data/physics_structured_rebuild_v10_smoke --run-dir /home/jiamo/VLM_runs/physics_structured_rebuild_v10_smoke --device cuda
```

Pilot generation and pre-freeze validation:

```bash
$PY physics_structured_rebuild_v10/generate_dataset.py --scale pilot --output-dir "$DATA" --workers 2
$PY physics_structured_rebuild_v10/validate_dataset.py --data-dir "$DATA"
```

The recorded cycle first attempted `--scale full`. When its sustained compute
estimate exceeded the current window, the deterministic training prefix was
preserved and the following explicit fallback finalized the pilot:

```bash
$PY physics_structured_rebuild_v10/generate_dataset.py --scale pilot --finalize-pilot-from-full-attempt --output-dir "$DATA" --workers 2
```

Controlled forward and specialist development runs:

```bash
for EXPERIMENT in A B C D; do
  $PY physics_structured_rebuild_v10/train_forward_ablation.py --experiment "$EXPERIMENT" --data-dir "$DATA" --run-dir "$RUN" --device cuda
done
$PY physics_structured_rebuild_v10/evaluate_forward_ablation.py --data-dir "$DATA" --run-dir "$RUN" --device cuda
$PY physics_structured_rebuild_v10/train_inverse_ranker.py --data-dir "$DATA" --run-dir "$RUN" --device cuda
$PY physics_structured_rebuild_v10/train_visual_inverse.py --data-dir "$DATA" --run-dir "$RUN" --device cuda
$PY physics_structured_rebuild_v10/train_direction_calibrator.py --data-dir "$DATA" --run-dir "$RUN" --device cuda
```

Freeze and one-time locked evaluation:

```bash
$PY physics_structured_rebuild_v10/freeze_and_evaluate.py freeze --data-dir "$DATA" --run-dir "$RUN" --device cuda
$PY physics_structured_rebuild_v10/validate_dataset.py --data-dir "$DATA" --include-locked-after-freeze
$PY physics_structured_rebuild_v10/freeze_and_evaluate.py evaluate-locked --data-dir "$DATA" --run-dir "$RUN" --device cuda
$PY physics_structured_rebuild_v9/evaluate_dual_direction_extra_forward.py --forward-selector-ensemble --output "$RUN/postfreeze_system_comparability_v10.json"
$PY physics_structured_rebuild_v10/build_results_report.py --data-dir "$DATA" --run-dir "$RUN"
```

The system-comparability replay is deliberately last and its output path
refuses overwrite. It is a post-freeze confirmation only, never a selection
input.

Post-cycle diagnostic audit:

```bash
$PY physics_structured_rebuild_v10/audit_forward_identical_natural.py --data-dir "$DATA" --run-dir "$RUN" --device cuda
```

This audit reopens development checkpoints and saved locked summaries only. It
does not open the raw locked-test JSONL, retrain, or modify model selection.
