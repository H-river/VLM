# Direction specialist rebuild v4

This folder trains and serves a new five-output direction specialist while
leaving Qwen and the other specialists frozen.

## Numerical contract

Each transition contains:

- optical setup values;
- the current five-value beam state;
- one action from the registered 81-action grid;
- the simulator-produced next beam state, used only to construct labels.

`data.py` converts setup, current state, and action into 46 numerical features.
For each of the five beam fields, it divides the physical change by that
field's tolerance. A normalized change below `-1` is `decrease`, a change
between `-1` and `1` inclusive is `no_change`, and a change above `1` is
`increase`.

The selected model consists of five independent
`HistGradientBoostingClassifier` heads. Every head produces three class
probabilities. The final output therefore contains five direction labels and
five three-class probability vectors.

## Training data and balance

The one-seed candidate uses:

- 202,500 transitions from 2,500 existing-distribution training groups;
- 162,000 transitions from 2,000 difficult-distribution training groups;
- 364,500 training transitions total.

The two validation files are disjoint from training and each contain 24,300
transitions. No held-out test file is opened.

For every output field, samples are stratified by:

1. direction class: `decrease`, `no_change`, or `increase`;
2. distance from the nearest direction threshold:
   `abs(abs(normalized_change) - 1)`;
3. distance bin: near (`<=0.25`), middle (`<=0.75`), or far (`>0.75`).

`balance_table` assigns inverse-square-root weights to those 45
field/class/distance strata. The tree run mixes 20% of this stratum weight with
80% uniform weight.

## Reproduce the selected run

```bash
/home/jiamo/miniconda3/envs/optical_sim/bin/python \
  /home/jiamo/VLM/direction_rebuild_v4/train_tree_direction.py \
  --output-dir /home/jiamo/VLM_runs/direction_rebuild_v4_tree_one_seed \
  --max-iter 240 \
  --max-leaf-nodes 63 \
  --min-samples-leaf 40 \
  --learning-rate 0.08 \
  --l2-regularization 0.10 \
  --balance-strength 0.20 \
  --device cuda
```

The selected artifact is
`/home/jiamo/VLM_runs/direction_rebuild_v4_tree_one_seed/direction_tree_v4.pkl`.
It is enabled only through a hash-pinned direction overlay manifest; the
default candidate manifest continues to use the frozen v1 direction model.
