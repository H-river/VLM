#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jiamo/VLM"
PYTHON_BIN="/home/jiamo/miniconda3/envs/optical_sim/bin/python"
DATA_DIR="/home/jiamo/VLM_data/control_rebuild_v4_numerical"
RUN_DIR="/home/jiamo/VLM_runs/control_rebuild_v4_one_seed"
MEASUREMENT_CALIBRATOR="/home/jiamo/VLM_runs/measurement_rebuild_v4_one_seed/measurement_calibrator_v4.pt"
STAGE_DIR="${RUN_DIR}/post_generation_stages"

mkdir -p "${RUN_DIR}" "${STAGE_DIR}"
# This service is triggered by a PathExists unit. Disable the path trigger
# after the first launch so a persistent manifest cannot re-run training.
systemctl --user stop vlm-v4-post-generation.path || true
systemctl --user stop vlm-v4-post-generation-ccd2.path || true

generator_is_busy() {
    local unit state
    for unit in vlm-numdata-v4-full.service vlm-numdata-v4-safe.service; do
        state="$(
            systemctl --user show "${unit}" \
                --property=ActiveState \
                --value 2> /dev/null || true
        )"
        case "${state}" in
            active|activating|reloading|deactivating)
                return 0
                ;;
        esac
    done
    return 1
}

while generator_is_busy; do
    sleep 10
done

cd "${REPO_ROOT}"
if [[ ! -f "${STAGE_DIR}/verify_frozen_qwen.complete" ]]; then
    "${PYTHON_BIN}" Qwen_orchestration/scripts/verify_frozen_baseline.py \
        > "${RUN_DIR}/verify_frozen_qwen.log" 2>&1
    touch "${STAGE_DIR}/verify_frozen_qwen.complete"
fi

if [[ ! -f "${STAGE_DIR}/finalize_numerical_dataset.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/finalize_numerical_dataset.py \
        --data-dir "${DATA_DIR}" \
        > "${RUN_DIR}/finalize_numerical_dataset.log" 2>&1
    touch "${STAGE_DIR}/finalize_numerical_dataset.complete"
fi

if [[ ! -f "${STAGE_DIR}/train_forward_v4.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/train_forward.py \
        --v4-data "${DATA_DIR}" \
        --output-dir "${RUN_DIR}" \
        --epochs 36 \
        --device cuda \
        > "${RUN_DIR}/train_forward_v4.log" 2>&1
    touch "${STAGE_DIR}/train_forward_v4.complete"
fi

if [[ ! -f "${STAGE_DIR}/train_inverse_v4.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/train_inverse.py \
        --v4-data "${DATA_DIR}" \
        --output-dir "${RUN_DIR}" \
        --forward-artifact "${RUN_DIR}/forward_physics_residual_v4.pt" \
        --error-bank "${RUN_DIR}/measurement_error_bank_v4.npz" \
        --epochs 20 \
        --device cuda \
        > "${RUN_DIR}/train_inverse_v4.log" 2>&1
    touch "${STAGE_DIR}/train_inverse_v4.complete"
fi

if [[ ! -f "${STAGE_DIR}/compare_v3_v4_selection_validation.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/compare_v3_v4_selection_validation.py \
        --data-dir "${DATA_DIR}" \
        --v4-run "${RUN_DIR}" \
        --device cuda \
        > "${RUN_DIR}/compare_v3_v4_selection_validation.log" 2>&1
    touch "${STAGE_DIR}/compare_v3_v4_selection_validation.complete"
fi

if [[ ! -f "${STAGE_DIR}/train_visual_integrated_v4.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/train_visual_scorer.py \
        --output-dir "${RUN_DIR}" \
        --device cuda \
        --measurement-calibrator "${MEASUREMENT_CALIBRATOR}" \
        --forward-artifact "${RUN_DIR}/forward_physics_residual_v4.pt" \
        --inverse-initialization "${RUN_DIR}/inverse_control_v4.pt" \
        --artifact-name visual_sensor_scorer_v4_integrated.pt \
        > "${RUN_DIR}/train_visual_integrated_v4.log" 2>&1
    touch "${STAGE_DIR}/train_visual_integrated_v4.complete"
fi

if [[ ! -f "${STAGE_DIR}/candidate_overlay_manifest.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/write_candidate_manifest.py \
        --control-run "${RUN_DIR}" \
        > "${RUN_DIR}/candidate_overlay_manifest.log" 2>&1
    touch "${STAGE_DIR}/candidate_overlay_manifest.complete"
fi

if [[ ! -f "${STAGE_DIR}/controlled_validation.complete" ]]; then
    if ! jq -e '.complete == true' "${RUN_DIR}/controlled_validation.json" \
        > /dev/null 2>&1; then
        "${PYTHON_BIN}" control_rebuild_v4/evaluate_controlled.py \
            --control-run "${RUN_DIR}" \
            --splits val \
            --output "${RUN_DIR}/controlled_validation.json" \
            --device cuda \
            > "${RUN_DIR}/controlled_validation.log" 2>&1
    fi
    touch "${STAGE_DIR}/controlled_validation.complete"
fi

if [[ ! -f "${STAGE_DIR}/closed_loop_val.complete" ]]; then
    if ! jq -e '.complete == true' "${RUN_DIR}/closed_loop_val.json" \
        > /dev/null 2>&1; then
        "${PYTHON_BIN}" control_rebuild_v4/evaluate_closed_loop.py \
            --run-dir "${RUN_DIR}" \
            --split val \
            --max-requests 50 \
            --max-steps 3 \
            --output "${RUN_DIR}/closed_loop_val.json" \
            --device cuda \
            > "${RUN_DIR}/closed_loop_val.log" 2>&1
    fi
    touch "${STAGE_DIR}/closed_loop_val.complete"
fi

if [[ ! -f "${STAGE_DIR}/orchestrated_runtime_validation.complete" ]]; then
    if ! jq -e '.complete == true' \
        "${RUN_DIR}/orchestrated_runtime_validation.json" \
        > /dev/null 2>&1; then
        "${PYTHON_BIN}" control_rebuild_v4/validate_orchestrated_runtime.py \
            --control-run "${RUN_DIR}" \
            --device cuda \
            > "${RUN_DIR}/orchestrated_runtime_validation.log" 2>&1
    fi
    touch "${STAGE_DIR}/orchestrated_runtime_validation.complete"
fi

if [[ ! -f "${STAGE_DIR}/orchestrated_system_validation.complete" ]]; then
    if ! jq -e '.complete == true' \
        "${RUN_DIR}/orchestrated_system_validation.json" \
        > /dev/null 2>&1; then
        "${PYTHON_BIN}" control_rebuild_v4/evaluate_orchestrated_system.py \
            --control-run "${RUN_DIR}" \
            --device cuda \
            > "${RUN_DIR}/orchestrated_system_validation.log" 2>&1
    fi
    touch "${STAGE_DIR}/orchestrated_system_validation.complete"
fi

if [[ ! -f "${STAGE_DIR}/assess_validation.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/assess_validation.py \
        --run-dir "${RUN_DIR}" \
        > "${RUN_DIR}/assess_validation.log" 2>&1
    touch "${STAGE_DIR}/assess_validation.complete"
fi

if [[ ! -f "${STAGE_DIR}/summarize_validation.complete" ]]; then
    "${PYTHON_BIN}" control_rebuild_v4/summarize_results.py \
        --run-dir "${RUN_DIR}" \
        --phase validation \
        > "${RUN_DIR}/summarize_validation.log" 2>&1
    touch "${STAGE_DIR}/summarize_validation.complete"
fi

touch "${RUN_DIR}/post_generation_pipeline.complete"
echo "v4 post-generation training and validation complete"
