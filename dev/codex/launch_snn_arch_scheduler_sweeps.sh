#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${SNN_ARCH_SCHEDULER_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${SNN_ARCH_SCHEDULER_RUN_ROOT:-${REPO_ROOT}/snn/experiments/snn_v1_arch_scheduler/${RUN_ID}}"
LOG_DIR="${RUN_ROOT}/logs"
ORCH_LOG="${LOG_DIR}/orchestrator_${RUN_ID}.log"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

setsid nohup python3 dev/codex/run_snn_arch_scheduler_sweeps.py \
  --run-root "${RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  "$@" \
  > "${ORCH_LOG}" 2>&1 < /dev/null &

PID="$!"
printf '%s\n' "${PID}" > "${RUN_ROOT}/orchestrator.pid"

echo "Started SNN architecture/scheduler sweeps"
echo "  pid: ${PID}"
echo "  run root: ${RUN_ROOT}"
echo "  orchestrator log: ${ORCH_LOG}"
echo "  status: ${RUN_ROOT}/status.json"
