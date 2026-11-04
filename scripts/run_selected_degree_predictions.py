#!/usr/bin/env python3
"""Rerun ONLY validation-selected GCN and GraphSAGE configurations to recover
per-test-node predictions, cross-checked against the original result CSVs.

Unmodified original model training scripts and their result JSONs are never
changed. Generated files live in results/{model}_node_predictions/.
"""
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT = 'constructed_data/shared_split_seed42.npz'
SETS = {
    'gcn': 'results/gcn_selected_by_validation.csv',
    'graphsage': 'results/graphsage_selected_by_validation.csv',
}


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def validate_prediction(path, reference):
    """Require 750 unique test nodes, good degree bins and matched test score."""
    if not path.is_file():
        raise ValueError('Missing node predictions: '+str(path))
    rows = read_rows(path)
    if len(rows) != 750:
        raise ValueError('%s: expected 750 predictions, got %d' % (path, len(rows)))
    ids = [int(row['node_id']) for row in rows]
    if len(set(ids)) != 750 or min(ids) < 0 or max(ids) > 1489:
        raise ValueError(str(path) + ': incorrect test-node IDs')
    for row in rows:
        deg = int(row['degree'])
        b = '0' if deg == 0 else ('1-2' if deg <= 2 else '3-5' if deg <= 5 else '6+')
        if row['degree_bin'] != b:
            raise ValueError(str(path) + ': invalid degree bin')
        expected = int(int(row['predicted_label']) == int(row['true_label']))
        if int(row['correct']) != expected:
            raise ValueError(str(path) + ': incorrect per-node correctness')
    observed = sum(int(row['correct']) for row in rows) / 750 * 100
    if abs(observed - reference) > 0.05:
        raise ValueError('%s: test-score mismatch %.5f versus %.5f' %
                         (path, observed, reference))
    return observed


def verify_result(json_path, predictions_path, row, model):
    expected_test = float(row['test_accuracy_pct'])
    expected_val = float(row['best_val_accuracy_pct'])
    with json_path.open(encoding='utf-8') as f:
        saved = json.load(f)
    assert abs(100*float(saved['test_accuracy'])-expected_test) < 0.05
    assert abs(100*float(saved['best_val_accuracy'])-expected_val) < 0.05
    assert saved['graph'] == 'constructed_data/%s_grid/%s.npz' % (row['family'], row['graph'])
    assert saved['split'] == SPLIT
    assert int(saved['seed']) == 123
    assert int(saved['hidden']) == int(row['hidden'])
    assert int(saved['n_test']) == 750
    if model == 'gcn':
        assert int(saved['patience']) == int(row['patience'])
        assert int(saved['max_epochs']) == 2000
        assert math.isclose(float(saved['dropout']), 0.5)
    else:
        assert math.isclose(float(saved['lr']), float(row['lr']), abs_tol=1e-8)
        assert int(saved['epochs']) == 500
        assert int(saved['best_epoch']) == int(row['best_epoch'])
        assert saved['num_samples'] == [5, 5]
    return validate_prediction(predictions_path, expected_test)


def execute_one(model, row):
    graph_name=row['graph']
    graph_path='constructed_data/%s_grid/%s.npz' % (row['family'],graph_name)
    if not (ROOT/graph_path).is_file():
        raise FileNotFoundError(graph_path)
    folder=ROOT/'results'/(model+'_node_predictions')
    folder.mkdir(parents=True, exist_ok=True)
    log_dir=ROOT/'repro_logs'/'degree_model_exports'
    log_dir.mkdir(parents=True, exist_ok=True)
    result_path=folder/(graph_name+'.json')
    preds_path=folder/(graph_name+'.csv')
    if result_path.exists() and preds_path.exists():
        try:
            acc=verify_result(result_path,preds_path,row,model)
            print('SKIP verified:',model,graph_name,'test=%.2f%%'%acc,flush=True)
            return
        except (ValueError,KeyError,AssertionError) as ex:
            print('Existing export invalid; rerunning:',ex,flush=True)
    base=['docker','run','--rm']
    if model=='gcn':
        base += ['--platform','linux/amd64']
        image='h2gcn-repro:v1'
        command=['python','scripts/gcn_npz_with_predictions.py',
                 '--hidden',row['hidden'],'--patience',row['patience'],
                 '--epochs','2000','--dropout','0.5','--lr','0.01',
                 '--weight-decay','0.0005']
    else:
        base += ['--platform','linux/arm64']
        image='graphsage-repro:pt251'
        command=['python','scripts/graphsage_npz_with_predictions.py',
                 '--hidden',row['hidden'],'--lr',row['lr'],
                 '--epochs','500','--batch-size','256','--num-samples','5','5',
                 '--log-every','100']
    cmd=base+['-v',str(ROOT)+':/workspace','-w','/workspace',image]
    cmd+=command+['--graph',graph_path,'--split',SPLIT,'--seed','123',
                  '--output',str(result_path.relative_to(ROOT)),
                  '--predictions-output',str(preds_path.relative_to(ROOT))]
    log_path=log_dir/(model+'_'+graph_name+'.log')
    print('RUN:',model,graph_name,'hidden='+row['hidden'],
          ('patience='+row['patience'] if model=='gcn' else 'lr='+row['lr']),flush=True)
    with log_path.open('w',encoding='utf-8') as log:
        completed=subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
    lines=log_path.read_text(encoding='utf-8',errors='replace').splitlines()
    for line in lines[-9:]:print(' ',line,flush=True)
    if completed.returncode != 0:
        raise RuntimeError('%s failed (exit=%s): %s'%(graph_name,completed.returncode,log_path))
    try:
        score=verify_result(result_path,preds_path,row,model)
    except Exception as ex:
        raise RuntimeError('RERUN DID NOT REPRODUCE SAVED BASELINE for '+graph_name+': '+str(ex)) from ex
    print('PASS:',model,graph_name,'recovered %.3f%%'%score,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['gcn','graphsage','all'],default='all')
    parser.add_argument('--limit-per-model',type=int,default=0,
                        help='0=all 30 graphs per model; 1=one-graph pilot per model')
    args=parser.parse_args()
    if args.limit_per_model<0:
        parser.error('limit-per-model must be >=0')
    for model in (['gcn','graphsage'] if args.model=='all' else [args.model]):
        source=ROOT/SETS[model]
        if not source.exists():
            raise FileNotFoundError('Missing selected configurations: '+str(source))
        rows=read_rows(source)
        keys=[(r['family'],r['homophily'],int(r['replicate'])) for r in rows]
        if len(rows)!=30 or len(set(keys))!=30:
            raise ValueError('%s: expected 30 unique graph selections'%source)
        rows.sort(key=lambda r:(r['family'],float(r['homophily']),int(r['replicate'])))
        if args.limit_per_model:
            rows=rows[:args.limit_per_model]
        print('\n===',model,len(rows),'graph(s) ===',flush=True)
        for n,row in enumerate(rows,1):
            print('[%d/%d]'%(n,len(rows)),end=' ',flush=True)
            execute_one(model,row)
    print('\nPASS: selected-baseline exports validated.',flush=True)

if __name__=='__main__':
    main()
