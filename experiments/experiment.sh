#!/bin/bash
#SBATCH --job-name=one_policy_to_run_them_all
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=49
#SBATCH --mem-per-cpu=2000
#SBATCH --time=2-23:59:59

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

conda activate one_policy_to_run_them_all
exec python3 -m experiments.run --preset train_full "$@"
