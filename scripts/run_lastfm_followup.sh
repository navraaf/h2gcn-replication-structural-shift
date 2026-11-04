#!/usr/bin/env bash
# Run at repo root: bash scripts/run_lastfm_followup.sh audit|pilot|baselines|h2gcn-seeds
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-audit}"
graph='existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz'
split='existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz'
[[ -f "$graph" && -f "$split" ]] || { echo 'Missing processed LastFM graph or split' >&2; exit 1; }

run_baseline() {
  local model="$1" seed="$2" epochs="$3" patience="$4" label="$5"
  local out="results/lastfm_baselines/${label}"
  if [[ -f "$out/results.json" ]]; then
    echo "SKIP completed: $out/results.json"
    return
  fi
  if [[ -d "$out" && -n "$(ls -A "$out")" ]]; then
    echo "ERROR: partial output in $out; inspect before rerunning" >&2
    exit 1
  fi
  echo "RUN $model, seed=$seed, max_epochs=$epochs, patience=$patience"
  docker run --rm --platform linux/amd64 \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v "$PWD":/workspace -w /workspace \
    h2gcn-repro:v1 \
    python scripts/lastfm_gcn_mlp_validated.py \
      --model "$model" --graph "$graph" --split "$split" \
      --seed "$seed" --hidden 64 --epochs "$epochs" --patience "$patience" \
      --dropout 0.5 --lr 0.01 --weight-decay 0.0005 \
      --output-dir "$out"
}

run_h2gcn_extra() {
  local seed="$1"
  local out="results/lastfm_h2gcn/full_seed${seed}"
  if [[ -f "$out/results.json" ]]; then
    echo "SKIP completed: $out/results.json"
    return
  fi
  if [[ -d "$out" && -n "$(ls -A "$out")" ]]; then
    echo "ERROR: partial output in $out; inspect before rerunning" >&2
    exit 1
  fi
  [[ -f scripts/train_lastfm_h2gcn_validated.py ]] || {
    echo 'Missing original LastFM H2GCN evaluation script' >&2;exit 1;
  }
  echo "RUN H2GCN repeat seed=$seed using existing fixed split"
  docker run --rm --platform linux/amd64 \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v "$PWD":/workspace -w /workspace \
    h2gcn-repro:v1 \
    python scripts/train_lastfm_h2gcn_validated.py \
      --graph "$graph" --split "$split" --seed "$seed" \
      --epochs 2000 --patience 150 --output-dir "$out"
}

case "$mode" in
  audit)
    python3 scripts/audit_lastfm_class_performance.py
    ;;
  pilot)
    run_baseline gcn 123 20 12 pilot20_gcn_seed123
    run_baseline mlp 123 20 12 pilot20_mlp_seed123
    ;;
  baselines)
    run_baseline gcn 123 2000 150 full_gcn_seed123
    run_baseline mlp 123 2000 150 full_mlp_seed123
    ;;
  h2gcn-seeds)
    run_h2gcn_extra 124
    run_h2gcn_extra 125
    ;;
  *)
    echo 'Usage: bash scripts/run_lastfm_followup.sh audit|pilot|baselines|h2gcn-seeds' >&2
    exit 2
    ;;
esac
