#!/usr/bin/env bash
set -euo pipefail
# Single node, eight GPUs by default. Optional Hydra overrides follow the script.
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_single_b200.sh" 4b "$@"
