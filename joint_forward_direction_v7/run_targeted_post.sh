#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/jiamo/VLM
run_root=/home/jiamo/VLM_runs/joint_forward_direction_v7_targeted_one_seed
data_root=/home/jiamo/VLM_data/joint_forward_direction_targeted_v7
python_bin=/home/jiamo/miniconda3/envs/optical_sim/bin/python

cd "$repo_root"
mkdir -p "$run_root"

if [[ ! -f "$run_root/shared_forward_direction_v7_summary.json" ]]; then
  "$python_bin" joint_forward_direction_v7/train.py \
    --targeted-data "$data_root" \
    --output-dir "$run_root" \
    --version joint_forward_direction_v7_targeted_one_seed
fi

if [[ ! -f "$run_root/orchestrated_system_shared_v7_validation.json" ]]; then
  "$python_bin" joint_forward_direction_v7/evaluate_system.py \
    --v7-run "$run_root"
fi

"$python_bin" joint_forward_direction_v7/audit_targeted_results.py
