#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rbm_venv="${ROBOMETER_VENV:-$PWD/.venv}"
export VIRTUAL_ENV="$(cd "$rbm_venv" && pwd)"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
exec "$VIRTUAL_ENV/bin/python" scripts/launch_b200.py single "$@"
