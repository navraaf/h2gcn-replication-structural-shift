#!/usr/bin/env bash
# Run from the repository root: bash scripts/run_lastfm_h2gcn.sh check|pilot|full
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-pilot}"
graph='existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz'
split='existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz'
if [[ ! -f "$graph" || ! -f "$split" ]]; then
  echo 'Missing LastFM NPZ graph or split; check existing_data/lastfm_asia/processed/' >&2
  exit 1
fi

case "$mode" in
  check) epochs=1; patience=1; seed=123; label='check'; check='--check-only' ;;
  pilot) epochs=20; patience=12; seed=123; label='pilot20_seed123'; check='' ;;
  full) epochs=2000; patience=150; seed=123; label='full_seed123'; check='' ;;
  *) echo 'Usage: bash scripts/run_lastfm_h2gcn.sh check|pilot|full' >&2; exit 2 ;;
esac

# Each run has a distinct output directory and does not touch H2GCN's earlier checkpoints.
output="results/lastfm_h2gcn/${label}"
mkdir -p results/lastfm_h2gcn
if [[ "$mode" != check && -e "$output/results.json" ]]; then
  echo "Already complete: $output/results.json"; exit 0
fi
if [[ "$mode" != check && -d "$output" && -n "$(ls -A "$output")" ]]; then
  echo "Output directory already contains files: $output" >&2
  echo 'Inspect the prior run rather than overwriting it.' >&2
  exit 1
fi

args=(python scripts/train_lastfm_h2gcn_validated.py
  --graph "$graph" --split "$split" --seed "$seed"
  --epochs "$epochs" --patience "$patience"
  --output-dir "$output")
if [[ -n "$check" ]]; then args+=("$check"); fi
# Architecture: x86_64 TensorFlow 2.2 Docker, as used for previous H2GCN runs.
docker run --rm --platform linux/amd64 \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  -v "$PWD":/workspace -w /workspace \
  h2gcn-repro:v1 "${args[@]}"
