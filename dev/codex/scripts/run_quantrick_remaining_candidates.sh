#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 LEGACY_PID" >&2
  exit 2
fi

LEGACY_PID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ID="20260722T053851Z"
RUN_ROOT="${REPO_ROOT}/snn/experiments/snn_v2_quantrick/${RUN_ID}"
CONFIG="${REPO_ROOT}/snn/train_v2_config_quantrick.yaml"

while kill -0 "${LEGACY_PID}" 2>/dev/null; do
  sleep 30
done

LEGACY_HISTORY="${RUN_ROOT}/candidates/legacy_8bit/legacy_8bit_training_history.json"
if [[ ! -s "${LEGACY_HISTORY}" ]]; then
  echo "Legacy candidate ended without a readable history: ${LEGACY_HISTORY}" >&2
  exit 1
fi

run_candidate() {
  local label="$1"
  local scope="$2"
  local output_dir="experiments/snn_v2_quantrick/${RUN_ID}/candidates/${label}"
  local log_path="${RUN_ROOT}/candidate_${label}.log"

  python "${REPO_ROOT}/snn/train_v2.py" \
    --config "${CONFIG}" \
    --set "paths.output_dir=${output_dir}" \
    --set "paths.checkpoint_name=${label}_network.pt" \
    --set "paths.plot_name=${label}_training_curves.png" \
    --set "paths.history_name=${label}_training_history.json" \
    --set model.weight_quantization.mode=decomposed \
    --set "model.weight_quantization.scope=${scope}" \
    --set training.epochs=40 \
    --set training.max_train_samples=50000 \
    --set training.max_val_samples=10000 \
    --set training.val_eval_samples=10000 \
    --set runtime.device=cuda \
    --set runtime.save_plot=true 2>&1 | tee "${log_path}"
}

cd "${REPO_ROOT}"
run_candidate decomposed_first first
run_candidate decomposed_all all


python "${REPO_ROOT}/dev/codex/scripts/analyze_quantrick_candidates.py" \
  "${RUN_ROOT}" 2>&1 | tee "${RUN_ROOT}/candidate_ranking.log"

