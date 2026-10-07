#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rbm_venv="${ROBOMETER_VENV:-$PWD/.venv}"
export VIRTUAL_ENV="$(cd "$rbm_venv" && pwd)"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
# Keep the production workload. Debug mode forcibly disables uploads/eval/saves.
exec "$VIRTUAL_ENV/bin/python" scripts/launch_b200.py debug 4b \
  training.per_device_train_batch_size=32 data.max_frames=8 \
  data.resized_height=null data.resized_width=null \
  'data.train_datasets=[rbm-1m-id]' \
  data.dataloader_num_workers=8 data.dataloader_persistent_workers=true "$@"
