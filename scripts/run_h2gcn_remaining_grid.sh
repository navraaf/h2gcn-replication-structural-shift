#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p repro_logs

for family in sbm dcsbm; do
  for graph in constructed_data/"${family}_grid"/*.npz; do

    name="$(basename "$graph" .npz)"

    # The h=0.10 experiments are already complete.
    case "$name" in
      *_h0.10_*) continue ;;
    esac

    log="repro_logs/h2gcn_${name}_full2000.log"

    if [ -f "$log" ] && \
       grep -q '^Best performance:' "$log"; then
      echo "Already completed: $name"
      continue
    fi

    echo "================================"
    echo "Training: $name"
    echo "================================"

    docker run --rm --platform linux/amd64 \
      -v "$PWD":/workspace \
      -w /workspace/h2gcn \
      h2gcn-repro:v1 \
      python run_experiments.py H2GCN npz \
      --dataset "$name" \
      --dataset_path "../${graph}" \
      --split_path ../constructed_data/shared_split_seed42.npz \
      --epochs 2000 \
      --random_seed 123 \
      --network_setup M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO \
      --adj_nhood 1 2 \
      --best_val_criteria val_acc \
      2>&1 | tee "$log"

    if ! grep -q '^Best performance:' "$log"; then
      echo "Missing final results: $name"
      exit 1
    fi

  done
done

echo "H2GCN grid experiments completed."
