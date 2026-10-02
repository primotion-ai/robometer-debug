#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run --no-sync python scripts/launch_b200.py single "$@"
