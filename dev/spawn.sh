#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"
container_name="lavaurma_dev"
image_name="lavaurma"
history_file="${script_dir}/bash_history"

if docker inspect "${container_name}" >/dev/null 2>&1; then
    echo "Container '${container_name}' already exists." >&2
    echo "Start it with ./run.sh, or remove it explicitly before running ./spawn.sh again." >&2
    exit 1
fi

touch "${history_file}"

docker_args=(
    run
    --detach
    --name "${container_name}"
    --network host
    --workdir /app/one_policy_to_run_them_all
    --volume "${workspace_root}:/app"
    --volume "${history_file}:/root/.bash_history"
    --env LIBGL_ALWAYS_SOFTWARE=1
    --env MESA_LOADER_DRIVER_OVERRIDE=swrast
    --env MUJOCO_GL=egl
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    --env PYOPENGL_PLATFORM=egl
)

docker_info="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
if [[ "${docker_info}" == *nvidia* ]]; then
    docker_args+=(--gpus all)
else
    echo "NVIDIA Docker runtime not detected; starting without GPU access." >&2
fi

if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
    docker_args+=(--env "DISPLAY=${DISPLAY}" --volume /tmp/.X11-unix:/tmp/.X11-unix)

    if [[ -f "${XAUTHORITY:-${HOME}/.Xauthority}" ]]; then
        xauthority_file="${XAUTHORITY:-${HOME}/.Xauthority}"
        docker_args+=(--env XAUTHORITY=/tmp/.Xauthority --volume "${xauthority_file}:/tmp/.Xauthority:ro")
    fi

    if command -v xhost >/dev/null 2>&1; then
        xhost +local:docker >/dev/null
    else
        echo "DISPLAY is set, but xhost is unavailable; local X11 authorization was not changed." >&2
    fi
elif [[ -n "${DISPLAY:-}" ]]; then
    echo "DISPLAY is set, but /tmp/.X11-unix is unavailable; X11 forwarding is disabled." >&2
fi

docker "${docker_args[@]}" "${image_name}" sleep infinity
