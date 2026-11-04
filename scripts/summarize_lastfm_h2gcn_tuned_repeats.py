#!/usr/bin/env python3
"""Combine frozen seed-123 test and independently validated seed-124/125 runs."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path('results/lastfm_h2gcn_tuned_repeats')
FIRST = Path('results/lastfm_h2gcn_sensitivity/selected_l2small_final_test/selected_test_results.json')
SEL = Path('results/lastfm_h2gcn_sensitivity/validation_selected_config.json')
GRAPH = Path('existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz')
SPLIT = Path('existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz')
METRICS = ('test_accuracy','test_macro_f1','test_weighted_f1','test_balanced_accuracy')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def write_csv(path, header, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
    tmp.replace(path)

def main():
    first=json.loads(FIRST.read_text(encoding='utf-8'))
    selected=json.loads(SEL.read_text(encoding='utf-8'))['selected']
    assert selected['config']=='l2small' and first['selected_config']=='l2small'
    assert sha256(GRAPH)==first['graph_sha256'], 'Current graph differs from selected seed-123 test'
    assert sha256(SPLIT)==first['split_sha256'], 'Current split differs from selected seed-123 test'
    assert first['seed']==123 and math.isclose(first['l2_regularization'],0.00005,abs_tol=1e-12)
    runrecords=[{'seed':123,**{m:float(first[m]) for m in METRICS},'best_epoch':int(first['best_epoch']),
                 'val_accuracy':float(first['val_accuracy']),'val_macro_f1':float(first['val_macro_f1'])}]
    for seed in (124,125):
        path=ROOT/('full_seed%d'%seed)/'results.json'
        r=json.loads(path.read_text(encoding='utf-8'))
        assert r['model']=='h2gcn' and r['selected_configuration']=='l2small'
        assert int(r['seed'])==seed and r['graph']==str(GRAPH) and r['split']==str(SPLIT)
        assert int(r['hidden'])==64 and float(r['learning_rate'])==0.01
        assert float(r['dropout'])==0.5 and math.isclose(float(r['weight_decay']),0.00005,abs_tol=1e-12)
        assert int(r['epochs_max'])==2000 and int(r['patience'])==150
        if any(not math.isfinite(float(r[m])) or not 0<=float(r[m])<=1 for m in METRICS):
            raise ValueError('Bad metric in '+str(path))
        nodes=ROOT/('full_seed%d'%seed)/'test_node_predictions.csv'
        with nodes.open(newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==3828 and len(set(int(x['node_id']) for x in rows))==3828
        assert all(int(x['correct'])==int(x['true_label']==x['predicted_label']) for x in rows)
        assert math.isclose(sum(int(x['correct']) for x in rows)/3828,float(r['test_accuracy']),abs_tol=1e-12)
        runrecords.append({'seed':seed,**{m:float(r[m]) for m in METRICS},
                           'best_epoch':int(r['best_epoch']),'val_accuracy':float(r['val_accuracy']),
                           'val_macro_f1':float(r['val_macro_f1'])})
    headers=('seed','best_epoch','val_accuracy','val_macro_f1')+METRICS
    write_csv(ROOT/'tuned_lastfm_h2gcn_runs.csv',headers,[[r[k] for k in headers] for r in runrecords])
    summary_header=['model','selected_config','n_training_seeds','seeds']
    summary_vals=['h2gcn','l2small',3,'123;124;125']
    for metric in METRICS:
        values=[r[metric] for r in runrecords]
        summary_header+=['{}_mean'.format(metric),'{}_sample_sd'.format(metric)]
        summary_vals += [statistics.mean(values),statistics.stdev(values)]
        print('{}: {:.2%} ± {:.2%} (SD across training seeds)'.format(metric,statistics.mean(values),statistics.stdev(values)))
    write_csv(ROOT/'tuned_lastfm_h2gcn_summary.csv',summary_header,[summary_vals])
    print('PASS: selected H2GCN seed123/124/125, unchanged graph/split, metrics checked.')
    print('SAVED:',ROOT/'tuned_lastfm_h2gcn_runs.csv')
    print('SAVED:',ROOT/'tuned_lastfm_h2gcn_summary.csv')

if __name__=='__main__':
    main()
