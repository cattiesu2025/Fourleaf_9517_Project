#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "run_sift_sweep.sh is kept for compatibility; using the unified sweep runner." >&2
exec "$PROJECT_ROOT/scripts/comp9517" sweep-sift "$@"
