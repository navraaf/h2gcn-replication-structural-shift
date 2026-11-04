#!/usr/bin/env bash
# Freeze the validation-selected H2GCN configuration. Train only seeds 124, 125.
# Bash 3.2/macOS compatible. Does not overwrite the seed-123 result.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-dry-run}"
if [[ "$MODE" != 'dry-run' && "$MODE" != 'run' && "$MODE" != 'summary' ]]; then
  echo 'Usage: bash scripts/run_lastfm_h2gcn_tuned_repeats.sh [dry-run|run|summary]' >&2
  exit 2
fi
GRAPH='existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz'
SPLIT='existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz'
SEED123='results/lastfm_h2gcn_sensitivity/selected_l2small_final_test/selected_test_results.json'
SELECTION='results/lastfm_h2gcn_sensitivity/validation_selected_config.json'
SCRIPT='scripts/train_lastfm_h2gcn_tuned_repeat.py'
SUMMARY='scripts/summarize_lastfm_h2gcn_tuned_repeats.py'
for f in "$GRAPH" "$SPLIT" "$SEED123" "$SELECTION" "$SCRIPT" "$SUMMARY"; do
  if [[ ! -f "$f" ]]; then echo "ERROR: missing $f" >&2; exit 1; fi
done
python3 - "$SELECTION" "$SEED123" <<'PY'
import sys,json,math
sel=json.load(open(sys.argv[1]))['selected']
orig=json.load(open(sys.argv[2]))
assert sel['config']=='l2small' and orig['selected_config']=='l2small'
assert int(sel['hidden'])==64 and math.isclose(float(sel['learning_rate']),0.01)
assert math.isclose(float(sel['dropout']),0.5)
assert math.isclose(float(sel['weight_decay']),0.00005,abs_tol=1e-12)
assert orig['seed']==123 and math.isclose(float(orig['l2_regularization']),0.00005,abs_tol=1e-12)
print('PASS: frozen validation-selected configuration matches completed seed-123 test.')
PY
if [[ "$MODE" == 'summary' ]]; then
  python3 "$SUMMARY"
  exit 0
fi
mkdir -p repro_logs/lastfm_h2gcn_tuned_repeats
for SEED in 124 125; do
  OUT="results/lastfm_h2gcn_tuned_repeats/full_seed${SEED}"
  if [[ -f "$OUT/results.json" ]]; then
    python3 - "$OUT/results.json" "$OUT/test_node_predictions.csv" "$SEED" "$GRAPH" "$SPLIT" <<'PY'
import sys,json,math,csv,os
rec=json.load(open(sys.argv[1])); csv_path=sys.argv[2]
assert rec['model']=='h2gcn' and rec['selected_configuration']=='l2small'
assert int(rec['seed'])==int(sys.argv[3]) and rec['graph']==sys.argv[4] and rec['split']==sys.argv[5]
assert int(rec['epochs_max'])==2000 and int(rec['patience'])==150
assert int(rec['hidden'])==64 and math.isclose(float(rec['learning_rate']),0.01)
assert math.isclose(float(rec['dropout']),0.5) and math.isclose(float(rec['weight_decay']),0.00005,abs_tol=1e-12)
assert all(0<=float(rec[x])<=1 for x in ('test_accuracy','test_macro_f1','test_weighted_f1','test_balanced_accuracy'))
assert os.path.exists(csv_path)
with open(csv_path,newline='') as f:
 rows=list(csv.DictReader(f))
assert len(rows)==3828 and len(set(int(r['node_id']) for r in rows))==3828
assert abs(sum(int(x['correct']) for x in rows)/3828-float(rec['test_accuracy']))<1e-12
print('PASS: existing seed {} output and node predictions checked.'.format(sys.argv[3]))
PY
    echo "SKIP verified: seed $SEED"
    continue
  fi
  if [[ -d "$OUT" && -n "$(ls -A "$OUT")" ]]; then
    echo "ERROR: partial output in $OUT. Preserve and inspect it; not overwriting." >&2
    exit 1
  fi
  if [[ "$MODE" == 'dry-run' ]]; then
    echo "DRY-RUN: would train H2GCN seed=$SEED, L2=0.00005, output=$OUT"
    continue
  fi
  echo "RUN: selected H2GCN L2=0.00005 seed=$SEED (2000 epochs max, patience 150)"
  docker run --rm --platform linux/amd64 \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v "$PWD":/workspace -w /workspace h2gcn-repro:v1 \
    python "$SCRIPT" --graph "$GRAPH" --split "$SPLIT" \
    --seed "$SEED" --hidden 64 --learning-rate 0.01 \
    --dropout 0.5 --weight-decay 0.00005 \
    --epochs 2000 --patience 150 --output-dir "$OUT" \
    2>&1 | tee "repro_logs/lastfm_h2gcn_tuned_repeats/seed${SEED}.log" | tail -n 30
  [[ -f "$OUT/results.json" ]] || { echo "ERROR: no verified results.json for seed $SEED" >&2; exit 1; }
done
if [[ "$MODE" == 'dry-run' ]]; then
  echo 'PASS: dry-run only. No training launched.'
else
  python3 "$SUMMARY"
  echo 'PASS: all three fixed-configuration training seeds complete.'
fi
