#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

docker build \
    --tag lavaurma \
    --file "${repo_root}/docker/Dockerfile" \
    "${repo_root}"
