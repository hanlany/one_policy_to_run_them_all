#!/usr/bin/env bash

set -euo pipefail

container_name="lavaurma_dev"
container_state="$(docker inspect --format '{{.State.Running}}' "${container_name}" 2>/dev/null || true)"

if [[ -z "${container_state}" ]]; then
    echo "Container '${container_name}' does not exist. Create it with ./spawn.sh." >&2
    exit 1
fi

if [[ "${container_state}" == "true" ]]; then
    echo "Container '${container_name}' is already running. Enter it with ./attach.sh."
    exit 0
fi

created_display="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${container_name}" | sed -n 's/^DISPLAY=//p')"
if [[ -n "${created_display}" && -n "${DISPLAY:-}" && "${created_display}" != "${DISPLAY}" ]]; then
    echo "Warning: current DISPLAY '${DISPLAY}' differs from creation-time DISPLAY '${created_display}'." >&2
    echo "Recreate the container if SSH-forwarded X11 does not work." >&2
fi

if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    if command -v xhost >/dev/null 2>&1; then
        xhost +local:docker >/dev/null
    else
        echo "DISPLAY is set, but xhost is unavailable; local X11 authorization was not changed." >&2
    fi
fi

docker start "${container_name}" >/dev/null
echo "Started container '${container_name}'. Enter it with ./attach.sh."
