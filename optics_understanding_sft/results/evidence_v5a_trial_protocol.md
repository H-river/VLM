# Evidence-grounded v5A 50-step protocol

This protocol was frozen before generation or training. It permits one local seed-42 QLoRA run capped at 50 optimizer steps and no paid API calls. It does not open the v4 holdout, unchanged legacy development sets, or sealed pilot test.

## Data

- Generate 300 new IID physical scenarios with seed 15052 and new group/record namespaces.
- Split complete physical scenarios with seed 15053 into 200 train, 50 development, and 50 confirmation scenarios.
- Each scenario yields one feasible/infeasible control pair and one answerable/insufficient sufficiency pair: 1,200 records total, with exact status balance inside every split.
- Prompts expose raw measured centroids plus adapter-derived control residuals, sufficiency deltas, and sufficiency directions. They never expose success/agreement flags, ranks, selected actions, witnesses, statuses, or labels.
- Confirmation prompts are target-free. Confirmation labels remain private unless a development checkpoint passes every gate.
- The 50-step curriculum uses 50 training scenarios (200 unique records). With batch size 1 and four-way gradient accumulation, every record is consumed exactly once.

## Training

- Base model: local Qwen2.5-VL-3B-Instruct, with a fresh rank-8 QLoRA adapter.
- Seed and data seed: 42.
- Ordinary completion loss; no decision-token weighting.
- Learning rate 1e-5, 10% warmup, checkpoints at steps 25 and 50.
- Maximum 50 optimizer steps. No later seed is allowed in this round.

## Development promotion gates

All gates must pass on all 200 development records:

- schema validity at least 0.98;
- status macro-F1 at least 0.75 for both tasks;
- recall at least 0.70 for all four statuses;
- pair-joint status accuracy at least 0.70 for both tasks;
- feasible control action exact-match at least 0.60;
- feasible control simulator success at least 0.70;
- feasible control minimum-motion correctness at least 0.60;
- answerable sufficiency direction accuracy at least 0.75;
- insufficient sufficiency visible-witness validity at least 0.70.

If neither checkpoint passes, stop without opening confirmation. If one or both pass, select the checkpoint by: number of passed gates, then minimum per-task status F1, then mean pair-joint accuracy, then lower step. Open the 200-record confirmation split once for the selected checkpoint only. Confirmation uses the same gates and cannot be used for model selection.

Passing v5A establishes prompt-visible evidence aggregation only. It does not establish raw-centroid arithmetic, internal Fresnel simulation, topology generalization, or laboratory validity.
