#!/usr/bin/env bash
# Run from any directory; script is stored inside H2GCN/scripts.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-pilot}"
case "$MODE" in
  pilot) LIMIT=(--limit 1) ;;
  full) LIMIT=(--limit 0) ;;
  *) echo "Usage: bash scripts/run_h2gcn_prediction_extraction.sh [pilot|full]" >&2; exit 2 ;;
esac
mkdir -p repro_logs results/h2gcn_node_predictions
LOG="repro_logs/h2gcn_checkpoint_extraction_${MODE}.log"
docker run --rm --platform linux/amd64 \
  -v "$PWD":/workspace \
  -w /workspace \
  h2gcn-repro:v1 \
  python scripts/extract_h2gcn_node_predictions.py \
  --repo-root /workspace \
  "${LIMIT[@]}" \
  2>&1 | tee "$LOG"
