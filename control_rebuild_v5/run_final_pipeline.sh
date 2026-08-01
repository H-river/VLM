#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jiamo/VLM"
PYTHON_BIN="/home/jiamo/miniconda3/envs/optical_sim/bin/python"
DATA_DIR="/home/jiamo/VLM_data/control_rebuild_v5_numerical"
RUN_DIR="/home/jiamo/VLM_runs/control_rebuild_v5_one_seed"
STAGE_DIR="${RUN_DIR}/stages"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${RUN_DIR}" "${STAGE_DIR}"
cd "${REPO_ROOT}"

if [[ ! -f "${STAGE_DIR}/finalize_data.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v5/finalize_numerical_dataset.py \
        --data-dir "${DATA_DIR}" \
        > "${RUN_DIR}/finalize_numerical_dataset.log" 2>&1
    touch "${STAGE_DIR}/finalize_data.complete"
fi

if [[ ! -f "${STAGE_DIR}/train_forward.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v5/train_forward.py \
        --additional-data "${DATA_DIR}" \
        --include-additional-data \
        --output-dir "${RUN_DIR}" \
        > "${RUN_DIR}/train_forward_v5.log" 2>&1
    touch "${STAGE_DIR}/train_forward.complete"
fi

if [[ ! -f "${STAGE_DIR}/train_inverse.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v5/train_inverse.py \
        --additional-data "${DATA_DIR}" \
        --include-additional-data \
        --output-dir "${RUN_DIR}" \
        > "${RUN_DIR}/train_inverse_v5.log" 2>&1
    touch "${STAGE_DIR}/train_inverse.complete"
fi

if [[ ! -f "${STAGE_DIR}/tests.complete" ]]; then
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${PYTHON_BIN}" -m pytest -q \
        control_rebuild_v5/tests control_rebuild_v4/tests \
        > "${RUN_DIR}/tests.log" 2>&1
    touch "${STAGE_DIR}/tests.complete"
fi

if [[ ! -f "${STAGE_DIR}/system_validation.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v5/evaluate_orchestrated_system.py \
        --v5-run "${RUN_DIR}" \
        --output "${RUN_DIR}/orchestrated_system_numerical_v5_validation.json" \
        --device cuda \
        > "${RUN_DIR}/orchestrated_system_numerical_v5_validation.log" 2>&1
    touch "${STAGE_DIR}/system_validation.complete"
fi

if [[ ! -f "${STAGE_DIR}/manifest.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v5/write_candidate_manifest.py \
        --run-dir "${RUN_DIR}" \
        > "${RUN_DIR}/candidate_manifest.log" 2>&1
    touch "${STAGE_DIR}/manifest.complete"
fi

if [[ ! -f "${STAGE_DIR}/audit.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v5/audit_results.py \
        --data-dir "${DATA_DIR}" \
        --run-dir "${RUN_DIR}" \
        > "${RUN_DIR}/audit.log" 2>&1
    touch "${STAGE_DIR}/audit.complete"
fi

touch "${RUN_DIR}/pipeline.complete"
echo "control rebuild v5 numerical pipeline complete"

