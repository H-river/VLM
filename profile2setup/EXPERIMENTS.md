# profile2setup v2 Experiments

## Baseline Experiment

Name:

```text
profile2setup_v2_all_modes_baseline
```

Model:

- profile CNN encoder
- simple text encoder
- setup encoder
- fusion MLP
- delta / absolute / change heads

Dataset:

- `all_modes` train/val/test

Metrics:

- offline normalized MAE
- offline physical MAE
- routed setup MAE
- closed-loop profile MSE
- closed-loop centroid/sigma error

Training history is saved in each run directory as `history.csv` and
`history.json`, with `loss_curve.png`, `component_loss_curves.png`, and
`mae_curves.png` when the corresponding metrics are available.

Qualitative prediction examples can be generated with:

```bash
python -m profile2setup.scripts.visualize_predictions_cli \
  --checkpoint profile2setup/checkpoints/profile2setup_v2_all_modes_baseline/best.pt \
  --data profile2setup/data/all_modes/test.jsonl \
  --variables-config profile2setup/configs/variables.yaml \
  --out-dir profile2setup/results/prediction_examples \
  --num-examples 10
```

## Recommended Result Table

| Run | Dataset | Model | Train loss | Val loss | Routed setup MAE | Closed-loop NMSE | Notes |
|---|---|---|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |  |

## Ablation Ideas

- no prompt
- no setup input
- target profile only
- current + target profile
- small CNN vs optional ResNet18 later
- delta-only vs absolute-only vs routed
- `target_base` vs `current_base` closed-loop policy

## Notes About Ambiguity

Many optical setups can produce similar profiles. Track parameter accuracy and profile success together, because exact absolute setup recovery may be ambiguous even when the predicted profile is useful.
