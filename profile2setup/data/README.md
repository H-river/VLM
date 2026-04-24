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
- `current_only/`: `current_profile + prompt -> target_setup`
- `paired_no_setup/`: `current_profile + target_profile + prompt -> target_setup`
- `all_modes/`: combined dataset across all four modes

Use `all_modes/train.jsonl` and `all_modes/val.jsonl` for mixed-mode training.
