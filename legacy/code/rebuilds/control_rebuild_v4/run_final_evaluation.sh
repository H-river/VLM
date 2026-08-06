#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jiamo/VLM"
PYTHON_BIN="/home/jiamo/miniconda3/envs/optical_sim/bin/python"
RUN_DIR="/home/jiamo/VLM_runs/control_rebuild_v4_one_seed"
STAGE_DIR="${RUN_DIR}/final_evaluation_stages"

if [[ ! -f "${RUN_DIR}/post_generation_pipeline.complete" ]]; then
    echo "validation pipeline is not complete" >&2
    exit 1
fi
if ! jq -e '.complete == true and .proceed_to_heldout == true' \
    "${RUN_DIR}/validation_decision.json" > /dev/null 2>&1; then
    echo "validation gates did not authorize the held-out evaluation" >&2
    exit 1
fi
if [[ -f "${RUN_DIR}/final_evaluation.complete" ]]; then
    echo "v4 final held-out evaluation is already complete"
    exit 0
fi
mkdir -p "${STAGE_DIR}"

cd "${REPO_ROOT}"
if [[ ! -f "${STAGE_DIR}/controlled_evaluation.complete" ]]; then
    if ! jq -e '.complete == true' "${RUN_DIR}/controlled_evaluation.json" \
        > /dev/null 2>&1; then
        "${PYTHON_BIN}" control_rebuild_v4/evaluate_controlled.py \
            --control-run "${RUN_DIR}" \
            --splits test_iid test_ood_physics test_visual_stress \
            --output "${RUN_DIR}/controlled_evaluation.json" \
            --device cuda \
            > "${RUN_DIR}/controlled_evaluation.log" 2>&1
    fi
    touch "${STAGE_DIR}/controlled_evaluation.complete"
fi

if [[ ! -f "${STAGE_DIR}/closed_loop_test_iid.complete" ]]; then
    if ! jq -e '.complete == true' "${RUN_DIR}/closed_loop_test_iid.json" \
        > /dev/null 2>&1; then
        "${PYTHON_BIN}" control_rebuild_v4/evaluate_closed_loop.py \
            --run-dir "${RUN_DIR}" \
            --split test_iid \
            --max-requests 50 \
            --max-steps 3 \
            --output "${RUN_DIR}/closed_loop_test_iid.json" \
            --device cuda \
            > "${RUN_DIR}/closed_loop_test_iid.log" 2>&1
    fi
    touch "${STAGE_DIR}/closed_loop_test_iid.complete"
fi

if [[ ! -f "${STAGE_DIR}/closed_loop_test_ood_physics.complete" ]]; then
    if ! jq -e '.complete == true' \
        "${RUN_DIR}/closed_loop_test_ood_physics.json" \
        > /dev/null 2>&1; then
        "${PYTHON_BIN}" control_rebuild_v4/evaluate_closed_loop.py \
            --run-dir "${RUN_DIR}" \
            --split test_ood_physics \
            --max-requests 50 \
            --max-steps 3 \
            --output "${RUN_DIR}/closed_loop_test_ood_physics.json" \
            --device cuda \
            > "${RUN_DIR}/closed_loop_test_ood_physics.log" 2>&1
    fi
    touch "${STAGE_DIR}/closed_loop_test_ood_physics.complete"
fi

if [[ ! -f "${STAGE_DIR}/summarize_final.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/summarize_results.py \
        --run-dir "${RUN_DIR}" \
        --phase final \
        > "${RUN_DIR}/summarize_final.log" 2>&1
    touch "${STAGE_DIR}/summarize_final.complete"
fi

if [[ ! -f "${STAGE_DIR}/completion_audit.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/audit_completion.py \
        --run-dir "${RUN_DIR}" \
        > "${RUN_DIR}/completion_audit.log" 2>&1
    touch "${STAGE_DIR}/completion_audit.complete"
fi

touch "${RUN_DIR}/final_evaluation.complete"
echo "v4 final held-out evaluation complete"
