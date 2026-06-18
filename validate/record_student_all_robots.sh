#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

mapfile -t RECORD_ROBOTS < <(python3 -m experiments.run --preset record_student --list-record-robots)

for ROBOT_ENTRY in "${RECORD_ROBOTS[@]}"; do
  ROBOT_INDEX="${ROBOT_ENTRY%%:*}"
  ROBOT_NAME="${ROBOT_ENTRY#*: }"

  echo "Recording record_student for ${ROBOT_INDEX}: ${ROBOT_NAME}"
  python3 -m experiments.run --preset record_student --record-robot "${ROBOT_INDEX}" "$@"
done
