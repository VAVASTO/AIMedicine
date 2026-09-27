#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-offline}"
if [[ "$mode" != offline && "$mode" != live ]]; then
  echo "Usage: scripts/run_demo.sh [offline|live]" >&2
  exit 2
fi
.venv/bin/python -m aimedicine.cli demo --mode "$mode" --output "reports/$mode"
