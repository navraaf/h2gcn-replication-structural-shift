#!/usr/bin/env python3
"""Summarize completed LastFM models on fixed split 42 across training seeds.

Rows correspond to runs; seed-wise standard deviations reflect initialization
(and training stochasticity) on ONE fixed split, not dataset-split uncertainty.
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    paths=[]
    paths.extend(sorted(Path('results/lastfm_h2gcn').glob('full_seed*/results.json')))
    paths.extend(sorted(Path('results/lastfm_baselines').glob('full_*_seed*/results.json')))
    if not paths:
        raise SystemExit('No completed LastFM model output found')
    rows=[]
    for path in paths:
        record=json.loads(path.read_text())
        model=record.get('model','h2gcn')
        if model not in ('h2gcn','gcn','mlp'):
            raise ValueError('Unexpected LastFM model name: '+model)
        rows.append(dict(model=model,seed=int(record['seed']),
             split=str(record['split']),graph=str(record['graph']),
             best_epoch=int(record['best_epoch']),
             val_accuracy=float(record['val_accuracy']),
             val_macro_f1=float(record['val_macro_f1']),
             test_accuracy=float(record['test_accuracy']),
             test_macro_f1=float(record['test_macro_f1']),
             test_weighted_f1=float(record['test_weighted_f1']),
             test_balanced_accuracy=float(record['test_balanced_accuracy'])))
    if len({(r['model'],r['seed']) for r in rows}) != len(rows):
        raise ValueError('Duplicate model/seed combinations')
    if len({r['split'] for r in rows}) != 1 or len({r['graph'] for r in rows}) != 1:
        raise ValueError('Runs do not use an identical graph and split')
    output=Path('results/lastfm_comparison')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'lastfm_model_runs.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    for r in rows:groups[r['model']].append(r)
    summaries=[]
    for model,subset in sorted(groups.items()):
        item={'model':model,'n_training_seeds':len(subset),
              'seeds':';'.join(str(r['seed']) for r in subset)}
        for metric in ('test_accuracy','test_macro_f1','test_weighted_f1','test_balanced_accuracy'):
            values=[r[metric] for r in subset]
            item[metric+'_mean']=statistics.mean(values)
            item[metric+'_sample_sd']=statistics.stdev(values) if len(values)>=2 else ''
        summaries.append(item)
    with (output/'lastfm_model_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print('PASS: %d completed runs; shared LastFM graph and split.' % len(rows))
    for r in summaries:
        print('%s n=%d test_acc=%.2f%% macro-F1=%.2f%%' % (
            r['model'],r['n_training_seeds'],r['test_accuracy_mean']*100,
            r['test_macro_f1_mean']*100))
    print('These are training-seed replicates on fixed split 42, NOT independent dataset splits.')
    print('Outputs:',output)

if __name__=='__main__':main()
