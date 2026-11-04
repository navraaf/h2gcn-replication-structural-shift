#!/usr/bin/env python3
"""Audit all four GraphSAGE settings for each synthetic graph, choose by validation
only, then produce run-level and paired-condition H2GCN/GCN comparisons.

Run from project root: python3 scripts/summarize_graphsage_grid.py
No external Python packages are required.
"""
import csv
from collections import defaultdict
from itertools import product
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'
TUNING=RESULTS/'graphsage_tuning'
PATTERN=re.compile(r'^(sbm|dcsbm)_h(0\.(?:10|30|50|70|90))_r([123])_seed(\d+)_h(64|128)_lr(0p1|0p7)_seed123\.json$')

def save_csv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=header,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('Wrote:',path.relative_to(ROOT),'rows=',len(rows))

def read_csv(path):
    if not path.is_file():
        raise SystemExit(f'Missing prerequisite CSV: {path}')
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def mean_sd(vals):
    return statistics.mean(vals),statistics.stdev(vals)

def valid_accuracy(x):
    v=float(x)
    if not math.isfinite(v) or not 0<=v<=1:
        raise ValueError(f'Invalid accuracy: {x}')
    return v

def positive_finite(x):
    v=float(x)
    if not math.isfinite(v) or v<0:
        raise ValueError(f'Invalid loss: {x}')
    return v

# 1. Audit all 120 JSONs. No test score enters the selection key.
records=[]
for path in sorted(TUNING.glob('*.json')):
    match=PATTERN.match(path.name)
    if not match:
        raise SystemExit(f'Unexpected JSON filename in tuning folder: {path.name}')
    fam,h,rep,graph_seed,hidden,lr_tag=match.groups()
    with path.open(encoding='utf8') as f: d=json.load(f)
    name=path.name.rsplit('_h',1)[0] # the final _h64/_h128 suffix
    expected_graph=f'constructed_data/{fam}_grid/{fam}_h{h}_r{rep}_seed{graph_seed}.npz'
    assert d['graph']==expected_graph,(path,'graph mismatch')
    assert d['split']=='constructed_data/shared_split_seed42.npz'
    assert d['model']=='2-layer GraphSAGE-mean' and d['optimizer']=='SGD'
    assert int(d['hidden'])==int(hidden)
    assert math.isclose(float(d['lr']),float(lr_tag.replace('p','.')),abs_tol=1e-10)
    assert int(d['seed'])==123 and int(d['epochs'])==500
    assert int(d['val_sampling_seed'])==1123 and int(d['test_sampling_seed'])==2123
    assert int(d['batch_size'])==256 and list(d['num_samples'])==[5,5]
    assert 1<=int(d['best_epoch'])<=500
    assert [int(d[k]) for k in ('n_train','n_val','n_test')]==[370,370,750]
    val=valid_accuracy(d['best_val_accuracy'])
    tst=valid_accuracy(d['test_accuracy'])
    vl=positive_finite(d['best_val_loss'])
    tl=positive_finite(d['test_loss'])
    records.append({
        'family':fam,'homophily':h,'replicate':int(rep),
        'graph':f'{fam}_h{h}_r{rep}_seed{graph_seed}',
        'hidden':int(hidden),'lr':float(d['lr']),
        'best_epoch':int(d['best_epoch']),
        'best_val_accuracy_pct':100*val,'best_val_loss':vl,
        'test_accuracy_pct':100*tst,'test_loss':tl,
        'isolated_nodes':int(d['isolated_nodes']),
        'result_file':str(path.relative_to(ROOT)),
    })

if len(records)!=120:
    raise SystemExit(f'Expected 120 result JSON files, got {len(records)}. Complete the grid first.')

groups=defaultdict(list)
for r in records: groups[(r['family'],r['homophily'],r['replicate'])].append(r)
expected_keys=set(product(('sbm','dcsbm'),('0.10','0.30','0.50','0.70','0.90'),(1,2,3)))
assert set(groups)==expected_keys,(len(groups),set(groups)^expected_keys)
for key, trials in groups.items():
    assert len(trials)==4,(key,len(trials))
    assert {(r['hidden'],r['lr']) for r in trials}=={(64,0.1),(64,0.7),(128,0.1),(128,0.7)}

head_all=['family','homophily','replicate','graph','hidden','lr','best_epoch',
          'best_val_accuracy_pct','best_val_loss','test_accuracy_pct','test_loss',
          'isolated_nodes','result_file']
save_csv(RESULTS/'graphsage_all_validation_candidates.csv',
         sorted(records,key=lambda r:(r['family'],r['homophily'],r['replicate'],r['hidden'],r['lr'])),head_all)

# Highest validation accuracy; lowest validation loss on ties.
# The final two fields ensure stable deterministic tie-breaking without testing metrics.
selected=[]
for key in sorted(groups):
    winner=min(groups[key],key=lambda r:(-r['best_val_accuracy_pct'],
                                          r['best_val_loss'],r['hidden'],r['lr']))
    selected.append(winner.copy())

# 2. Match original H2GCN three runs by graph family, target h and replicate.
ref_rows=read_csv(RESULTS/'h2gcn_grid_runs.csv')
h2={}
for r in ref_rows:
    key=(r['family'],f"{float(r['homophily']):.2f}",int(r['replicate']))
    assert key not in h2
    h2[key]=float(r['test_accuracy'])
assert set(h2)==expected_keys,(len(h2),set(h2)^expected_keys)

for row in selected:
    key=(row['family'],row['homophily'],row['replicate'])
    row['h2gcn_test_accuracy_pct']=h2[key]
    row['h2gcn_minus_graphsage_pp']=h2[key]-row['test_accuracy_pct']

head_sel=head_all+['h2gcn_test_accuracy_pct','h2gcn_minus_graphsage_pp']
save_csv(RESULTS/'graphsage_selected_by_validation.csv',selected,head_sel)

# 3. Mean ± sample SD (n=3) and paired advantage by h and family.
summary=[]
for fam in ('sbm','dcsbm'):
    for h in ('0.10','0.30','0.50','0.70','0.90'):
        group=sorted((r for r in selected if r['family']==fam and r['homophily']==h),
                     key=lambda r:r['replicate'])
        assert len(group)==3
        hmean,hsd=mean_sd([r['h2gcn_test_accuracy_pct'] for r in group])
        smean,ssd=mean_sd([r['test_accuracy_pct'] for r in group])
        dmean,dsd=mean_sd([r['h2gcn_minus_graphsage_pp'] for r in group])
        summary.append({'family':fam,'homophily':h,'n':3,
                       'h2gcn_mean_pct':hmean,'h2gcn_sd_pp':hsd,
                       'graphsage_mean_pct':smean,'graphsage_sd_pp':ssd,
                       'mean_h2gcn_advantage_pp':dmean,'sd_paired_advantage_pp':dsd,
                       'mean_isolated_nodes':statistics.mean(r['isolated_nodes'] for r in group)})
head_sum=['family','homophily','n','h2gcn_mean_pct','h2gcn_sd_pp',
          'graphsage_mean_pct','graphsage_sd_pp',
          'mean_h2gcn_advantage_pp','sd_paired_advantage_pp','mean_isolated_nodes']
save_csv(RESULTS/'h2gcn_vs_graphsage_validation_selected.csv',summary,head_sum)

# 4. Include the validation-tuned GCN when its audit CSV is available.
gcn_path=RESULTS/'gcn_selected_by_validation.csv'
if gcn_path.is_file():
    gcn_rows=read_csv(gcn_path)
    gcn={}
    for r in gcn_rows:
        key=(r['family'],f"{float(r['homophily']):.2f}",int(r['replicate']))
        assert key not in gcn
        gcn[key]=float(r['test_accuracy_pct'])
    assert set(gcn)==expected_keys,(len(gcn),set(gcn)^expected_keys)
    combined=[]
    for r in selected:
        key=(r['family'],r['homophily'],r['replicate'])
        combined.append({'family':r['family'],'homophily':r['homophily'],
                         'replicate':r['replicate'],
                         'h2gcn_pct':r['h2gcn_test_accuracy_pct'],
                         'gcn_tuned_pct':gcn[key],
                         'graphsage_tuned_pct':r['test_accuracy_pct'],
                         'h2gcn_minus_gcn_pp':r['h2gcn_test_accuracy_pct']-gcn[key],
                         'h2gcn_minus_graphsage_pp':r['h2gcn_minus_graphsage_pp']})
    head_comb=['family','homophily','replicate','h2gcn_pct','gcn_tuned_pct',
               'graphsage_tuned_pct','h2gcn_minus_gcn_pp','h2gcn_minus_graphsage_pp']
    save_csv(RESULTS/'h2gcn_gcn_graphsage_paired_runs.csv',combined,head_comb)

print('\nVALIDATION-SELECTED SUMMARY (mean ± sample SD, n=3)')
for row in summary:
    print(f"{row['family']:5} h={row['homophily']} | "
          f"H2GCN {row['h2gcn_mean_pct']:6.2f}±{row['h2gcn_sd_pp']:4.2f}% | "
          f"GraphSAGE {row['graphsage_mean_pct']:6.2f}±{row['graphsage_sd_pp']:4.2f}% | "
          f"difference {row['mean_h2gcn_advantage_pp']:+6.2f} pp")
print('\nSelection audit passed: 120 candidates, 30 validation-selected models, 10 summaries.')
