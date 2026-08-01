#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jiamo/VLM"
PYTHON_BIN="/home/jiamo/miniconda3/envs/optical_sim/bin/python"
DATA_DIR="/home/jiamo/VLM_data/control_rebuild_v5_numerical"
RUN_DIR="/home/jiamo/VLM_runs/control_rebuild_v5_one_seed"
STAGE_DIR="${RUN_DIR}/stages"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${STAGE_DIR}"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" control_rebuild_v5/build_numerical_dataset.py \
    --output-dir "${DATA_DIR}" \
    --workers 4 \
    --skip-checksums
touch "${STAGE_DIR}/generation.complete"
systemctl --user start --no-block vlm-v5-final.service
