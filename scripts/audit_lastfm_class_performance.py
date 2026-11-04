#!/usr/bin/env python3
"""Read-only verification and class-wise audit of a completed LastFM run.

No model training or test-based checkpoint selection. Operates on the model's
already-fixed predictions and emits diagnostics in a different output folder.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='results/lastfm_h2gcn/full_seed123')
    p.add_argument('--output', default='results/lastfm_class_audit')
    args = p.parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_dir():
        p.error('Missing model output directory: '+str(source))
    with (source/'results.json').open(encoding='utf8') as f:
        stats = json.load(f)
    prediction = read_csv(source/'test_node_predictions.csv')
    recorded = read_csv(source/'test_class_metrics.csv')
    support = read_csv(source/'class_support_by_split.csv') if (source/'class_support_by_split.csv').exists() else []
    pairs = [(int(r['true_label']),int(r['predicted_label'])) for r in prediction]
    if len(pairs) != int(stats['data']['split_sizes']['test']):
        raise ValueError('Test prediction count does not match recorded split size')
    k = int(stats['data']['classes'])
    if len(recorded) != k:
        raise ValueError('Expected one class metric per label')
    if any(t<0 or t>=k or q<0 or q>=k for t,q in pairs):
        raise ValueError('Predicted/true label out of range')
    n = len(pairs)
    hits = sum(t==q for t,q in pairs)
    total_accuracy = hits/n
    expected=float(stats['test_accuracy'])
    if abs(total_accuracy - expected)>1e-8:
        raise ValueError('Recorded test accuracy does not match node predictions')
    tcounts=Counter(t for t,_ in pairs)
    pcounts=Counter(q for _,q in pairs)
    tpcounts=Counter(t for t,q in pairs if t==q)
    table=[]
    for i in range(k):
        t=tcounts[i];q=pcounts[i];tp=tpcounts[i]
        prec=tp/q if q else 0.0
        rec=tp/t if t else 0.0
        f1=2*tp/(t+q) if t+q else 0.0
        item=dict(class_id=i,test_support=t,predicted_count=q,true_positive=tp,
                  precision=prec,recall=rec,f1=f1)
        if support:
            sr=next((r for r in support if int(r['class_id'])==i),None)
            if sr is None:raise ValueError('Class support missing class '+str(i))
            item['train_count']=int(sr['train_count'])
            item['val_count']=int(sr['val_count'])
            if int(sr['test_count'])!=t:raise ValueError('Class support mismatch')
        rt=next((r for r in recorded if int(r.get('class_id',-1))==i),None)
        if rt is None:raise ValueError('Missing recorded test class '+str(i))
        for key in ('precision','recall','f1'):
            if abs(float(rt[key])-item[key])>1e-8:
                raise ValueError('Metric mismatch at class %s (%s)' % (i,key))
        table.append(item)
    macro=sum(r['f1'] for r in table)/k
    if abs(macro-float(stats['test_macro_f1']))>1e-8:
        raise ValueError('Recorded macro-F1 differs from recomputation')
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'class_performance_audit.csv').open('w',newline='',encoding='utf8') as f:
        columns=list(table[0]);w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader();w.writerows(table)
    with (destination/'confusion_matrix.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['true_class']+['pred_'+str(i) for i in range(k)])
        pair_counts=Counter(pairs)
        for i in range(k):w.writerow([i]+[pair_counts[i,j] for j in range(k)])
    zero_recall=[r['class_id'] for r in table if r['recall']==0]
    zero_pred=[r['class_id'] for r in table if r['predicted_count']==0]
    summary={
        'dataset':'LastFM-Asia','source_results':str(source),
        'accuracy':total_accuracy,'macro_f1':macro,
        'test_nodes':n,'classes':k,
        'classes_never_predicted':zero_pred,
        'classes_with_zero_recall':zero_recall,
        'classes_predicted_at_least_once':k-len(zero_pred),
        'majority_predicted_fraction':max(r['predicted_count'] for r in table)/n,
        'most_frequently_predicted_class':max(table,key=lambda r:r['predicted_count'])['class_id'],
        'largest_test_class':max(table,key=lambda r:r['test_support'])['class_id'],
        'smallest_test_class_size':min(r['test_support'] for r in table),
    }
    with (destination/'class_audit_summary.json').open('w',encoding='utf8') as f:
        json.dump(summary,f,indent=2)
    print('PASS: all class metrics agree with saved predictions and summary.',flush=True)
    print('Test accuracy %.2f%%; macro-F1 %.2f%%; classes predicted %d/%d' % (
        100*total_accuracy,100*macro,summary['classes_predicted_at_least_once'],k),flush=True)
    print('Classes never predicted:',zero_pred,flush=True)
    print('Classes with zero recall:',zero_recall,flush=True)
    print('Most frequent predicted class: %d (%.2f%% of test predictions)' % (
        summary['most_frequently_predicted_class'],
        100*summary['majority_predicted_fraction']),flush=True)
    print('Saved:',destination,flush=True)

if __name__=='__main__':main()
