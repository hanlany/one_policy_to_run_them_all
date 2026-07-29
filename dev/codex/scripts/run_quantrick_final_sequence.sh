#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 MATCHED_CANDIDATE_ORCHESTRATOR_PID" >&2
  exit 2
fi

MATCHED_PID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ID="20260722T053851Z"
RUN_ROOT="${REPO_ROOT}/snn/experiments/snn_v2_quantrick/${RUN_ID}"
CONFIG="${REPO_ROOT}/snn/train_v2_config_quantrick.yaml"
BASELINE_CHECKPOINT="${REPO_ROOT}/snn/experiments/snn_v1_points_1_2/20260619T050231Z/long_best/long_e300_ts20_iw1_thr0p2_cd0p3_vd0p02/long_e300_ts20_iw1_thr0p2_cd0p3_vd0p02_network.pt"

while kill -0 "${MATCHED_PID}" 2>/dev/null; do
  sleep 30
done

RANKING="${RUN_ROOT}/candidate_ranking.json"
if [[ ! -s "${RANKING}" ]]; then
  echo "Matched candidate sequence ended without a ranking: ${RANKING}" >&2
  exit 1
fi

mapfile -t SCOPES < <(python - "${RANKING}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
for key in ("selected", "runner_up"):
    label = payload[key]
    if label not in {"decomposed_first", "decomposed_all"}:
        raise SystemExit(f"Unexpected candidate label: {label}")
    print(label.removeprefix("decomposed_"))
PY
)

SELECTED_SCOPE="${SCOPES[0]}"
RUNNER_UP_SCOPE="${SCOPES[1]}"

run_training() {
  local label="$1"
  local scope="$2"
  local output_dir="$3"
  local checkpoint_name="$4"
  local history_name="$5"
  local epochs="$6"
  local learning_rate="$7"
  local init_checkpoint="$8"
  local log_path="${RUN_ROOT}/${label}_training.log"

  python "${REPO_ROOT}/snn/train_v2.py" \
    --config "${CONFIG}" \
    --set "paths.output_dir=${output_dir}" \
    --set "paths.checkpoint_name=${checkpoint_name}" \
    --set "paths.plot_name=${label}_training_curves.png" \
    --set "paths.history_name=${history_name}" \
    --set "paths.ann_checkpoint=${init_checkpoint}" \
    --set "model.weight_quantization.scope=${scope}" \
    --set "training.epochs=${epochs}" \
    --set "training.learning_rate=${learning_rate}" \
    --set training.max_train_samples=null \
    --set training.max_val_samples=null \
    --set runtime.device=cuda \
    --set runtime.save_plot=true 2>&1 | tee "${log_path}"
}

check_gate() {
  local label="$1"
  local history_path="$2"
  local checkpoint_path="$3"
  python "${SCRIPT_DIR}/check_quantrick_acceptance.py" \
    "${history_path}" "${checkpoint_path}" \
    --output "${RUN_ROOT}/${label}_acceptance.json"
}

check_parity() {
  local label="$1"
  local checkpoint_path="$2"
  python "${SCRIPT_DIR}/verify_quantrick_rollout_parity.py" \
    "${checkpoint_path}" \
    --dataset "${REPO_ROOT}/snn/teacher_student_dagger_dataset.npz" \
    --samples 256 \
    --atol 1e-6 \
    --output "${RUN_ROOT}/${label}_rollout_parity.json"
}

promote_accepted() {
  local label="$1"
  local checkpoint_path="$2"
  local history_path="$3"
  local accepted_dir="${RUN_ROOT}/accepted"
  mkdir "${accepted_dir}"
  cp "${checkpoint_path}" "${accepted_dir}/quantrick_network.pt"
  cp "${history_path}" "${accepted_dir}/quantrick_training_history.json"
  cp "${RUN_ROOT}/${label}_acceptance.json" "${accepted_dir}/acceptance.json"
  cp "${RUN_ROOT}/${label}_rollout_parity.json" "${accepted_dir}/rollout_parity.json"
  printf '%s\n' "${label}" > "${accepted_dir}/source_label.txt"
  python "${SCRIPT_DIR}/verify_quantrick_accepted_bundle.py" \
    "${accepted_dir}" \
    --output "${accepted_dir}/bundle_verification.json"
}

cd "${REPO_ROOT}"
run_training \
  full_selected "${SELECTED_SCOPE}" \
  "experiments/snn_v2_quantrick/${RUN_ID}/final" \
  quantrick_network.pt quantrick_training_history.json \
  300 0.002 "${REPO_ROOT}/snn/student_model_latest.pth"
if check_gate full_selected \
  "${RUN_ROOT}/final/quantrick_training_history.json" \
  "${RUN_ROOT}/final/quantrick_network.pt"; then
  check_parity full_selected "${RUN_ROOT}/final/quantrick_network.pt"
  promote_accepted full_selected \
    "${RUN_ROOT}/final/quantrick_network.pt" \
    "${RUN_ROOT}/final/quantrick_training_history.json"
  exit 0
fi

run_training \
  fallback_selected "${SELECTED_SCOPE}" \
  "experiments/snn_v2_quantrick/${RUN_ID}/fallback_selected" \
  fallback_selected_network.pt fallback_selected_training_history.json \
  100 0.0002 "${BASELINE_CHECKPOINT}"
if check_gate fallback_selected \
  "${RUN_ROOT}/fallback_selected/fallback_selected_training_history.json" \
  "${RUN_ROOT}/fallback_selected/fallback_selected_network.pt"; then
  check_parity fallback_selected "${RUN_ROOT}/fallback_selected/fallback_selected_network.pt"
  promote_accepted fallback_selected \
    "${RUN_ROOT}/fallback_selected/fallback_selected_network.pt" \
    "${RUN_ROOT}/fallback_selected/fallback_selected_training_history.json"
  exit 0
fi

run_training \
  fallback_runner_up "${RUNNER_UP_SCOPE}" \
  "experiments/snn_v2_quantrick/${RUN_ID}/fallback_runner_up" \
  fallback_runner_up_network.pt fallback_runner_up_training_history.json \
  100 0.0002 "${BASELINE_CHECKPOINT}"
check_gate fallback_runner_up \
  "${RUN_ROOT}/fallback_runner_up/fallback_runner_up_training_history.json" \
  "${RUN_ROOT}/fallback_runner_up/fallback_runner_up_network.pt"
check_parity fallback_runner_up \
  "${RUN_ROOT}/fallback_runner_up/fallback_runner_up_network.pt"
promote_accepted fallback_runner_up \
  "${RUN_ROOT}/fallback_runner_up/fallback_runner_up_network.pt" \
  "${RUN_ROOT}/fallback_runner_up/fallback_runner_up_training_history.json"
