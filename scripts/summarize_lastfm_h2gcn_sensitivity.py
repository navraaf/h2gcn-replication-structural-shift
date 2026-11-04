#!/usr/bin/env python3
"""Summarize predeclared H2GCN sensitivity settings using validation only."""
import csv
import json
import math
from pathlib import Path

root = Path('results/lastfm_h2gcn_sensitivity')
reference_path = Path('results/lastfm_h2gcn/full_seed123/results.json')
if not reference_path.is_file():
    raise SystemExit('Missing reference H2GCN run: ' + str(reference_path))
with reference_path.open() as f:
    r = json.load(f)
# Intentionally do not import reference test metrics into selection.
all_rows = [dict(config='reference', hidden=64, learning_rate=.01,
                 dropout=.5, weight_decay=.0005,
                 val_accuracy=r['val_accuracy'], val_macro_f1=r['val_macro_f1'],
                 best_val_loss=r['best_val_loss'], best_epoch=r['best_epoch'],
                 checkpoint='results/lastfm_h2gcn/full_seed123/best_checkpoint')]
for key in ('h32', 'h128', 'lr003', 'drop02', 'l2small'):
    path = root / key / 'results.json'
    if not path.is_file():
        raise SystemExit('Missing candidate; finish full sweep before summary: ' + str(path))
    with path.open() as f:
        v = json.load(f)
    if v.get('validation_only') is not True or v.get('test_accuracy') is not None or v.get('test_macro_f1') is not None:
        raise SystemExit('Candidate contains test scores; refusing to use it for validation-only selection: '+str(path))
    if int(v['seed'])!=123 or int(v['epochs_max'])!=2000:
        raise SystemExit('Candidate seed or epoch budget mismatch: '+str(path))
    all_rows.append(dict(config=key, hidden=v['hidden'], learning_rate=v['learning_rate'],
                         dropout=v['dropout'], weight_decay=v['weight_decay'],
                         val_accuracy=v['val_accuracy'], val_macro_f1=v['val_macro_f1'],
                         best_val_loss=v['best_val_loss'], best_epoch=v['best_epoch'],
                         checkpoint=v['checkpoint']))
# primary validation accuracy, loss tie breaker, lexical config name for exact ties.
for v in all_rows:
    for metric in ('val_accuracy','val_macro_f1','best_val_loss'):
        if not math.isfinite(float(v[metric])):
            raise SystemExit('Nonfinite validation metric: '+v['config'])
ranked = sorted(all_rows, key=lambda v:(-float(v['val_accuracy']),float(v['best_val_loss']),v['config']))
root.mkdir(exist_ok=True,parents=True)
with (root/'validation_sensitivity_summary.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader(); writer.writerows(ranked)
selected = ranked[0]
with (root/'validation_selected_config.json').open('w') as f:
    json.dump({'selection_metric':'maximum validation accuracy, minimum validation loss tie-break',
               'seed':123, 'split':'existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz',
               'candidate_count':len(ranked), 'selected':selected,
               'test_metrics_examined_for_selection':False},f,indent=2)
print('Validation-only sensitivity summary')
for row in ranked:
    print('{:<10} val_acc {:6.2%}, macro-F1 {:6.2%}, loss {:.4f}, epoch {}'.format(row['config'],row['val_accuracy'],row['val_macro_f1'],row['best_val_loss'],row['best_epoch']))
print('Selected config: {} | no candidate test metrics used'.format(selected['config']))
print('Saved: '+str(root/'validation_sensitivity_summary.csv'))
print('Saved: '+str(root/'validation_selected_config.json'))
