# profile2setup data layout

Derived dataset artifacts live here. Raw simulator outputs stay under
`optical_sim/outputs/...`.

Each input-mode folder contains:

- `all.jsonl`: all records for that mode
- `train.jsonl`: training split
- `val.jsonl`: validation split
- `test.jsonl`: test split

Folders:

- `absolute/`: `target_profile + prompt -> target_setup`
- `edit/`: `current_profile + target_profile + current_setup + prompt -> target_setup/target_delta`
- `paired_no_setup/`: `current_profile + target_profile + prompt -> target_setup`
- `all_modes/`: combined dataset across all three modes

`current_only` was merged into `absolute` because both are profile-only to
setup-prediction tasks. The active dataset structure no longer keeps a separate
`current_only/` folder or task type.

Use `all_modes/train.jsonl` and `all_modes/val.jsonl` for mixed-mode training.
