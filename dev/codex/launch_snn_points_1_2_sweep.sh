#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${SNN_SWEEP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${SNN_SWEEP_RUN_ROOT:-${REPO_ROOT}/snn/experiments/snn_v1_points_1_2/${RUN_ID}}"
LOG_DIR="${RUN_ROOT}/logs"
ORCH_LOG="${LOG_DIR}/orchestrator_${RUN_ID}.log"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

setsid nohup python3 dev/codex/run_snn_points_1_2_sweep.py \
  --run-root "${RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  "$@" \
  > "${ORCH_LOG}" 2>&1 < /dev/null &

PID="$!"
printf '%s\n' "${PID}" > "${RUN_ROOT}/orchestrator.pid"

echo "Started SNN points 1+2 sweep"
echo "  pid: ${PID}"
echo "  run root: ${RUN_ROOT}"
echo "  orchestrator log: ${ORCH_LOG}"
echo "  status: ${RUN_ROOT}/status.json"
