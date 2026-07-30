#!/usr/bin/env bash

set -euo pipefail

container_name="lavaurma_dev"
container_state="$(docker inspect --format '{{.State.Running}}' "${container_name}" 2>/dev/null || true)"

if [[ -z "${container_state}" ]]; then
    echo "Container '${container_name}' does not exist. Create it with ./spawn.sh." >&2
    exit 1
fi

if [[ "${container_state}" != "true" ]]; then
    echo "Container '${container_name}' is stopped. Start it with ./run.sh." >&2
    exit 1
fi

# docker exec creates a separate shell. Exiting it leaves the container running.
exec docker exec -it "${container_name}" bash
