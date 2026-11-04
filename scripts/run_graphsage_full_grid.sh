#!/usr/bin/env bash
# Original H2GCN-fork GraphSAGE architecture, modern PyTorch NPZ adapter.
# 30 synthetic graphs x 4 configurations = 120 full 500-epoch trials.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="graphsage-repro:pt251"
SEED=123
SPLIT="constructed_data/shared_split_seed42.npz"
RESULT_DIR="results/graphsage_tuning"
LOG_DIR="repro_logs/graphsage_tuning"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

command -v docker >/dev/null || { echo 'Docker is not installed' >&2; exit 1; }
test -s "$SPLIT" || { echo "Missing split: $SPLIT" >&2; exit 1; }
test -s scripts/graphsage_npz_reference.py || { echo 'Missing GraphSAGE adapter' >&2; exit 1; }

graph_count=0
run_count=0
skip_count=0
for family in sbm dcsbm; do
  for graph in "constructed_data/${family}_grid/"*.npz; do
    [[ -f "$graph" ]] || { echo "No NPZ files in ${family}_grid" >&2; exit 1; }
    ((graph_count+=1))
    name="$(basename "$graph" .npz)"
    for hidden in 128 64; do
      for lr in 0.7 0.1; do
        lr_tag="${lr/./p}"
        key="${name}_h${hidden}_lr${lr_tag}_seed${SEED}"
        output="${RESULT_DIR}/${key}.json"
        log="${LOG_DIR}/${key}.log"
        # Always verify the actual graph, full training horizon, split,
        # hyperparameters and finite metrics before reusing a result.
        if [[ -f "$output" ]] && python3 - "$output" "$graph" "$SPLIT" "$hidden" "$lr" "$SEED" <<'PY' >/dev/null 2>&1
import json, math, sys
path, graph, split, hidden, lr, seed = sys.argv[1:]
with open(path, encoding='utf8') as f:
    d=json.load(f)
assert d['graph']==graph and d['split']==split
assert int(d['hidden'])==int(hidden) and int(d['seed'])==int(seed)
assert math.isclose(float(d['lr']),float(lr),abs_tol=1e-10)
assert int(d['epochs'])==500 and int(d['batch_size'])==256
assert list(d['num_samples'])==[5,5]
assert 1<=int(d['best_epoch'])<=500
assert all(math.isfinite(float(d[k])) for k in
           ('best_val_accuracy','best_val_loss','test_accuracy','test_loss'))
assert 0<=float(d['best_val_accuracy'])<=1
assert 0<=float(d['test_accuracy'])<=1
assert int(d['n_train'])==370 and int(d['n_val'])==370 and int(d['n_test'])==750
assert int(d['val_sampling_seed'])==int(seed)+1000
assert int(d['test_sampling_seed'])==int(seed)+2000
PY
        then
          echo "SKIP verified: $key"
          ((skip_count+=1))
          continue
        fi

        echo "==========================================================="
        echo "RUN $key"
        echo "==========================================================="
        docker run --rm --platform linux/arm64 \
          -v "$PWD":/workspace -w /workspace \
          "$IMAGE" \
          python scripts/graphsage_npz_reference.py \
          --graph "$graph" \
          --split "$SPLIT" \
          --seed "$SEED" \
          --hidden "$hidden" \
          --lr "$lr" \
          --epochs 500 \
          --batch-size 256 \
          --num-samples 5 5 \
          --log-every 50 \
          --output "$output" \
          2>&1 | tee "$log" | tail -n 12
        test -s "$output" || { echo "Missing result: $output" >&2; exit 1; }
        ((run_count+=1))
        echo "SAVED $output"
      done
    done
  done
done

if [[ "$graph_count" -ne 30 ]]; then
  echo "Expected exactly 30 graphs, found $graph_count; review inputs" >&2
  exit 1
fi

echo "COMPLETE: graphs=$graph_count new_trials=$run_count existing_verified=$skip_count total=$((run_count+skip_count))"
test "$((run_count+skip_count))" -eq 120
