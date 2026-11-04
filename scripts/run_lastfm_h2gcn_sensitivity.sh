#!/usr/bin/env bash
# Mac bash 3.2 compatible. From repo root:
# bash scripts/run_lastfm_h2gcn_sensitivity.sh dry-run|pilot|full|summary
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-dry-run}"
graph='existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz'
split='existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz'
image='h2gcn-repro:v1'
if [[ ! -f "$graph" || ! -f "$split" || ! -f 'scripts/train_lastfm_h2gcn_sensitivity.py' ]]; then
  echo 'ERROR: required LastFM graph/split or sensitivity script is missing.' >&2
  exit 1
fi

run_one() {
  label="$1"; hidden="$2"; lr="$3"; dropout="$4"; weight_decay="$5"; epochs="$6"; patience="$7"
  output="results/lastfm_h2gcn_sensitivity/$label"
  mkdir -p 'results/lastfm_h2gcn_sensitivity' 'repro_logs/lastfm_h2gcn_sensitivity'
  if [[ -f "$output/results.json" ]]; then
    if python3 - "$output/results.json" "$hidden" "$lr" "$dropout" "$weight_decay" "$epochs" <<'PY'
import json,math,sys
r=json.load(open(sys.argv[1]))
assert r['validation_only'] is True
assert r['test_accuracy'] is None and r['test_macro_f1'] is None
assert int(r['seed'])==123 and int(r['hidden'])==int(sys.argv[2])
assert math.isclose(float(r['learning_rate']),float(sys.argv[3]),abs_tol=1e-12)
assert math.isclose(float(r['dropout']),float(sys.argv[4]),abs_tol=1e-12)
assert math.isclose(float(r['weight_decay']),float(sys.argv[5]),abs_tol=1e-12)
assert int(r['epochs_max'])==int(sys.argv[6])
print('Verified cached validation-only result.')
PY
    then
      echo "SKIP verified: $label"; return
    else
      echo "ERROR: existing result is inconsistent with requested config: $output" >&2
      exit 1
    fi
  fi
  if [[ -d "$output" && -n "$(ls -A "$output")" ]]; then
    echo "ERROR: incomplete or unexpected output. Inspect and preserve $output" >&2
    exit 1
  fi
  echo "RUN: $label hidden=$hidden lr=$lr dropout=$dropout l2=$weight_decay (validation only)"
  docker run --rm --platform linux/amd64 \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -v "$PWD":/workspace -w /workspace \
    "$image" python scripts/train_lastfm_h2gcn_sensitivity.py \
      --graph "$graph" --split "$split" --seed 123 \
      --epochs "$epochs" --patience "$patience" \
      --hidden "$hidden" --learning-rate "$lr" \
      --dropout "$dropout" --weight-decay "$weight_decay" \
      --validation-only --output-dir "$output" \
    2>&1 | tee "repro_logs/lastfm_h2gcn_sensitivity/${label}.log" | tail -n 22
  python3 - "$output/results.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r.get('validation_only') is True and r.get('test_accuracy') is None
assert r.get('test_macro_f1') is None
print('PASS: validation-only output; no test scores. Val acc {:.2%}, val macro-F1 {:.2%}, best epoch {}'.format(r['val_accuracy'],r['val_macro_f1'],r['best_epoch']))
PY
}

case "$mode" in
  dry-run)
    echo 'Planned validation-only H2GCN sensitivity, seed123, fixed LastFM split:'
    echo 'reference: hidden64 lr0.01 dropout0.5 L2=0.0005 (already trained)'
    echo 'h32: hidden32 lr0.01 dropout0.5 L2=0.0005'
    echo 'h128: hidden128 lr0.01 dropout0.5 L2=0.0005'
    echo 'lr003: hidden64 lr0.003 dropout0.5 L2=0.0005'
    echo 'drop02: hidden64 lr0.01 dropout0.2 L2=0.0005'
    echo 'l2small: hidden64 lr0.01 dropout0.5 L2=0.00005'
    echo 'No test labels are scored during candidate training.'
    ;;
  pilot)
    run_one pilot_h128_20ep 128 0.01 0.5 0.0005 20 12
    ;;
  full)
    run_one h32 32 0.01 0.5 0.0005 2000 150
    run_one h128 128 0.01 0.5 0.0005 2000 150
    run_one lr003 64 0.003 0.5 0.0005 2000 150
    run_one drop02 64 0.01 0.2 0.0005 2000 150
    run_one l2small 64 0.01 0.5 0.00005 2000 150
    python3 scripts/summarize_lastfm_h2gcn_sensitivity.py
    ;;
  summary)
    python3 scripts/summarize_lastfm_h2gcn_sensitivity.py
    ;;
  *) echo 'Usage: bash scripts/run_lastfm_h2gcn_sensitivity.sh dry-run|pilot|full|summary' >&2; exit 2 ;;
esac
