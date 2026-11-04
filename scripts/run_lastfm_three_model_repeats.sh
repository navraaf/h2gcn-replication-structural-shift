#!/usr/bin/env bash
# LastFM-Asia fixed-split replication: seeds 124 and 125 for H2GCN, GCN and MLP.
# Keeps all successful seed-123 results untouched. Resumes safely.
set -euo pipefail
cd "$(dirname "$0")/.."

GRAPH='existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz'
SPLIT='existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz'
IMAGE='h2gcn-repro:v1'
MODE="${1:-run}"
if [[ "$MODE" != run && "$MODE" != dry-run ]]; then
  echo 'Usage: bash scripts/run_lastfm_three_model_repeats.sh [run|dry-run]' >&2
  exit 2
fi
for f in "$GRAPH" "$SPLIT" scripts/train_lastfm_h2gcn_validated.py scripts/lastfm_gcn_mlp_validated.py scripts/summarize_lastfm_models.py; do
  [[ -f "$f" ]] || { echo "ERROR: missing prerequisite $f" >&2; exit 1; }
done

verify_record() {
  local record="$1" model="$2" seed="$3"
  python3 - "$record" "$model" "$seed" "$GRAPH" "$SPLIT" <<'PY'
import json, math, pathlib, sys
p, model, seed, graph, split = sys.argv[1:]
r = json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
assert str(r.get('model', 'h2gcn')) == model, 'incorrect model in saved record'
assert int(r['seed']) == int(seed), 'incorrect seed in saved record'
assert r['graph'] == graph and r['split'] == split, 'incorrect graph or split'
assert int(r['epochs_max']) == 2000 and int(r['patience']) == 150, 'incorrect training protocol'
assert 0 < int(r['best_epoch']) <= 2000, 'invalid best epoch'
for k in ('val_accuracy', 'test_accuracy', 'test_macro_f1', 'test_weighted_f1', 'test_balanced_accuracy'):
    v = float(r[k]); assert math.isfinite(v) and 0 <= v <= 1, 'invalid metric ' + k
print('VERIFIED:', model, 'seed='+seed, 'test_acc={:.2%}'.format(float(r['test_accuracy'])),
      'macro_f1={:.2%}'.format(float(r['test_macro_f1'])))
PY
}

for model in h2gcn gcn mlp; do
  for seed in 124 125; do
    if [[ "$model" == h2gcn ]]; then
      out="results/lastfm_h2gcn/full_seed${seed}"
      driver='scripts/train_lastfm_h2gcn_validated.py'
    else
      out="results/lastfm_baselines/full_${model}_seed${seed}"
      driver='scripts/lastfm_gcn_mlp_validated.py'
    fi
    if [[ -f "$out/results.json" ]]; then
      verify_record "$out/results.json" "$model" "$seed"
      echo "SKIP completed: $out"
      continue
    fi
    if [[ -d "$out" && -n "$(ls -A "$out")" ]]; then
      echo "ERROR: partial output in $out. Inspect it before rerunning; no files were overwritten." >&2
      exit 1
    fi
    echo "RUN: $model seed=$seed fixed_split42 max_epochs=2000 patience=150"
    if [[ "$MODE" == dry-run ]]; then
      echo "DRY RUN: would write $out"
      continue
    fi
    if [[ "$model" == h2gcn ]]; then
      docker run --rm --platform linux/amd64 \
        -e TF_CPP_MIN_LOG_LEVEL=2 \
        -v "$PWD":/workspace -w /workspace "$IMAGE" \
        python "$driver" --graph "$GRAPH" --split "$SPLIT" \
          --seed "$seed" --epochs 2000 --patience 150 --output-dir "$out"
    else
      docker run --rm --platform linux/amd64 \
        -e TF_CPP_MIN_LOG_LEVEL=2 \
        -v "$PWD":/workspace -w /workspace "$IMAGE" \
        python "$driver" --model "$model" --graph "$GRAPH" --split "$SPLIT" \
          --seed "$seed" --hidden 64 --epochs 2000 --patience 150 \
          --dropout 0.5 --lr 0.01 --weight-decay 0.0005 --output-dir "$out"
    fi
    [[ -f "$out/results.json" ]] || { echo "ERROR: missing expected result: $out/results.json" >&2; exit 1; }
    verify_record "$out/results.json" "$model" "$seed"
  done
done

if [[ "$MODE" == dry-run ]]; then
  echo 'PASS: dry-run; no training or output files created.'
else
  python3 scripts/summarize_lastfm_models.py
  echo 'PASS: all six requested repeat runs verified, summary generated.'
fi
