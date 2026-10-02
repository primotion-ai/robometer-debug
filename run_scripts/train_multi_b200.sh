#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The platform executes this command once per node/container. Accelerate starts
# GPU workers locally. Reuse the mounted uv environment without installing or
# changing dependencies concurrently from multiple nodes.
printf '[robometer launcher] entered on host=%s\n' "${HOSTNAME:-unknown}"
# Log only the distributed configuration, never the full environment or keys.
for rbm_name in NNODES NODE_RANK GROUP_RANK RANK NPROC_PER_NODE GPUS_PER_NODE WORLD_SIZE MASTER_ADDR MASTER_PORT; do
  printf '[robometer launcher] %s=%s\n' "$rbm_name" "${!rbm_name-<unset>}"
done
rbm_venv="${ROBOMETER_VENV:-$PWD/.venv}"
if [[ ! -x "$rbm_venv/bin/python" ]]; then
  echo "Robometer Python not found: $rbm_venv/bin/python" >&2
  echo "Prepare the environment before submitting: uv sync --locked --extra b200 --extra b200-kernels" >&2
  echo "The mounted .venv and its base Python must be accessible inside every container." >&2
  exit 1
fi
export VIRTUAL_ENV="$(cd "$rbm_venv" && pwd)"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME

# Preserve the platform's CUDA visibility, library paths and NCCL/RDMA settings.
# NNODES/NODE_RANK/NPROC_PER_NODE/MASTER_ADDR/MASTER_PORT are read by the launcher.
# NNODES defaults to 5 when absent; the platform's injected value takes priority.
exec "$VIRTUAL_ENV/bin/python" scripts/launch_b200.py multi "$@"
