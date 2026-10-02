#!/usr/bin/env bash
# Same batch, normal GC (steps 1–2), disabled GC (3–5), normal GC (6–7).
# Small dataset avoids the full RBM-1M startup; model/input dimensions stay full size.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export DEBUG_STEPS="${DEBUG_STEPS:-7}"
export RUN_NAME="${RUN_NAME:-qwen27b_rank_wait_$(date +%Y%m%d_%H%M%S)}"
export ROBOMETER_DIAGNOSE_RANK_WAIT=1
export ROBOMETER_PROFILE_GC=1
export ROBOMETER_PROFILE_TIMING=1
exec bash run_scripts/debug_b200.sh 27b \
    training.per_device_train_batch_size=16 data.max_frames=8 \
    data.resized_height=null data.resized_width=null \
    data.dataloader_num_workers=8 data.dataloader_persistent_workers=true "$@"
