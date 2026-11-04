#!/usr/bin/env bash
# Validate all nine GCN configurations from the original paper's syn-cora grid:
# hidden widths 16, 32, 64 x early-stop patience 40, 100, 200.
# This runs the TF2-compatible baseline, not the original TF1 code.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f scripts/gcn_npz_tf2.py ]]; then
  echo "ERROR: scripts/gcn_npz_tf2.py not found. Run from the existing H2GCN project." >&2
  exit 1
fi
if [[ ! -f constructed_data/shared_split_seed42.npz ]]; then
  echo "ERROR: shared split is missing." >&2
  exit 1
fi

mkdir -p results/gcn_tuning repro_logs/gcn_tuning

verify_result() {
  local output="$1" graph="$2" hidden="$3" patience="$4"
  [[ -s "$output" ]] || return 1
  python3 - "$output" "$graph" "$hidden" "$patience" <<'PY'
import json, sys
path, graph, hidden, patience = sys.argv[1:]
with open(path) as f:
    result = json.load(f)
assert result['graph'] == graph
assert result['split'] == 'constructed_data/shared_split_seed42.npz'
assert int(result['hidden']) == int(hidden)
assert int(result['patience']) == int(patience)
assert int(result['seed']) == 123
assert int(result['max_epochs']) == 2000
assert abs(float(result['dropout']) - 0.5) < 1e-8
assert abs(float(result['lr']) - 0.01) < 1e-8
assert abs(float(result['weight_decay']) - 0.0005) < 1e-8
assert 1 <= int(result['best_epoch']) <= int(result['epochs_executed']) <= 2000
assert 0 <= float(result['best_val_accuracy']) <= 1
assert 0 <= float(result['test_accuracy']) <= 1
assert result['checkpoint_selection'].startswith('validation')
PY
}

counter=0
for family in sbm dcsbm; do
  for graph in "constructed_data/${family}_grid"/*.npz; do
    if [[ ! -f "$graph" ]]; then
      echo "ERROR: missing graphs in ${family}_grid" >&2
      exit 1
    fi
    name="$(basename "$graph" .npz)"
    for hidden in 16 32 64; do
      for patience in 40 100 200; do
        counter=$((counter + 1))

        if [[ "$hidden" == 64 && "$patience" == 200 ]]; then
          # Reuse the already completed 64-hidden/patience-200 grid.
          if [[ "$name" == 'sbm_h0.10_r1_seed2027' ]]; then
            output='results/gcn_sbm_h010_r1_full.json'
          else
            output="results/gcn_${name}_h64_seed123_full.json"
          fi
          if ! verify_result "$output" "$graph" "$hidden" "$patience"; then
            echo "ERROR: earlier 64-hidden result missing or mismatched: $output" >&2
            exit 1
          fi
          echo "REUSE: $name h=64 patience=200"
          continue
        fi

        label="${name}_h${hidden}_p${patience}_s123"
        output="results/gcn_tuning/${label}.json"
        log="repro_logs/gcn_tuning/${label}.log"

        if verify_result "$output" "$graph" "$hidden" "$patience"; then
          echo "SKIP: $label (verified result already present)"
          continue
        fi

        echo "RUN ${counter}/270: $label"
        docker run --rm --platform linux/amd64 \
          -v "$PWD":/workspace \
          -w /workspace \
          h2gcn-repro:v1 \
          python scripts/gcn_npz_tf2.py \
          --graph "$graph" \
          --split constructed_data/shared_split_seed42.npz \
          --hidden "$hidden" \
          --epochs 2000 \
          --patience "$patience" \
          --seed 123 \
          --dropout 0.5 \
          --lr 0.01 \
          --weight-decay 0.0005 \
          --output "$output" \
          2>&1 | tee "$log" | tail -n 6

        if ! verify_result "$output" "$graph" "$hidden" "$patience"; then
          echo "ERROR: failed output verification for $label" >&2
          exit 1
        fi
      done
    done
  done
done

echo "SUCCESS: All 270 GCN graph/configuration combinations have valid results."
echo "Next: choose hyperparameters using validation accuracy, NEVER test accuracy."
