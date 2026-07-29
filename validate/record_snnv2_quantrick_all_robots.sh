#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="20260722T053851Z"
RUN_ROOT="${REPO_ROOT}/snn/experiments/snn_v2_quantrick/${RUN_ID}"
ACCEPTED_DIR="${RUN_ROOT}/accepted"
VIDEO_DIR="${REPO_ROOT}/experiments/videos"
RECORDING_DIR="${RUN_ROOT}/recording"
MAPPING_PATH="${RECORDING_DIR}/video_mapping.tsv"

cd "${REPO_ROOT}"
python3 dev/codex/scripts/verify_quantrick_accepted_bundle.py \
  "${ACCEPTED_DIR}" \
  --output "${ACCEPTED_DIR}/bundle_verification.json"
mkdir -p "${RECORDING_DIR}"
find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' \
  | sort > "${RECORDING_DIR}/before_all_videos.txt"
printf 'robot_index\trobot_name\tvideo_path\tlog_path\n' > "${MAPPING_PATH}"

mapfile -t RECORD_ROBOTS < <(python3 -m experiments.run --preset record_snnv2_quantrick --list-record-robots)

for ROBOT_ENTRY in "${RECORD_ROBOTS[@]}"; do
  ROBOT_INDEX="${ROBOT_ENTRY%%:*}"
  ROBOT_NAME="${ROBOT_ENTRY#*: }"
  BEFORE_PATH="$(mktemp)"
  AFTER_PATH="$(mktemp)"
  LOG_PATH="${RECORDING_DIR}/robot_${ROBOT_INDEX}_${ROBOT_NAME}.log"

  find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' \
    | sort > "${BEFORE_PATH}"

  echo "Recording record_snnv2_quantrick for ${ROBOT_INDEX}: ${ROBOT_NAME}"
  python3 -m experiments.run --preset record_snnv2_quantrick --record-robot "${ROBOT_INDEX}" "$@" 2>&1 | tee "${LOG_PATH}"

  find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' \
    | sort > "${AFTER_PATH}"
  mapfile -t NEW_VIDEOS < <(comm -13 "${BEFORE_PATH}" "${AFTER_PATH}")
  rm -f "${BEFORE_PATH}" "${AFTER_PATH}"
  if [[ "${#NEW_VIDEOS[@]}" -ne 1 ]]; then
    echo "Expected exactly one new MP4 for ${ROBOT_NAME}, found ${#NEW_VIDEOS[@]}" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "${ROBOT_INDEX}" "${ROBOT_NAME}" \
    "${VIDEO_DIR}/${NEW_VIDEOS[0]}" "${LOG_PATH}" >> "${MAPPING_PATH}"
done

echo "Saved authoritative robot-to-video mapping to ${MAPPING_PATH}"

python3 dev/codex/scripts/validate_quantrick_videos.py \
  --mapping "${MAPPING_PATH}" \
  --baseline-snapshot "${RECORDING_DIR}/before_all_videos.txt" \
  --video-dir "${VIDEO_DIR}" \
  --robots-config "${REPO_ROOT}/experiments/configs/presets.yaml" \
  --contact-dir "${RECORDING_DIR}/contact_sheets" \
  --output "${RECORDING_DIR}/video_validation.json" \
  --expected-duration 10 \
  --duration-tolerance 2
