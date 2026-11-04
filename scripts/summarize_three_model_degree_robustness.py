#!/usr/bin/env python3
"""Cross-check H2GCN/GCN/GraphSAGE test-node predictions and derive
published-grade descriptive degree-robustness tables (no pseudo-independence).
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BINS=('0','1-2','3-5','6+')
FAMILIES=('sbm','dcsbm')
LEVELS=('0.10','0.30','0.50','0.70','0.90')
MODELS=('h2gcn','gcn','graphsage')


def load(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write(name, rows):
    if not rows:
        raise RuntimeError(name+': no rows')
    output=ROOT/'results'/name
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('SAVED:',output.relative_to(ROOT),len(rows),'rows')


def main():
    h2rows=load(ROOT/'results/h2gcn_node_predictions/node_predictions.csv')
    h2=defaultdict(dict)
    for r in h2rows:
        if r['split'] != 'test':
            continue
        key=(r['family'],r['target_homophily'],int(r['replicate']))
        nid=int(r['node_id'])
        if nid in h2[key]:
            raise ValueError('Duplicate H2GCN node for '+str(key))
        h2[key][nid]=r
    if len(h2)!=30:
        raise ValueError('Expected exactly 30 H2GCN graph conditions')
    checks_h2_rows=load(ROOT/'results/h2gcn_node_predictions/checkpoint_accuracy_verification.csv')
    checks_h2={(z['family'],z['target_homophily'],int(z['replicate'])):z for z in checks_h2_rows}
    if len(checks_h2)!=30 or any(z['state']!='PASS' for z in checks_h2_rows):
        raise ValueError('H2GCN checkpoint verification incomplete')
    bymodel={m:{} for m in MODELS}
    selected={}
    for model in ('gcn','graphsage'):
        rows=load(ROOT/('results/%s_selected_by_validation.csv'%model))
        selected[model]={(z['family'],z['homophily'],int(z['replicate'])):z for z in rows}
        if len(rows)!=30 or len(selected[model])!=30:
            raise ValueError('Selection manifest must contain 30 unique graph rows')
    accuracy_checks=[]
    paired=[]
    for fam in FAMILIES:
        for h in LEVELS:
            for replicate in (1,2,3):
                key=(fam,h,replicate)
                source=h2[key]
                if len(source)!=750:
                    raise ValueError('%s: expected 750 H2GCN test nodes'%str(key))
                bymodel['h2gcn'][key]=source
                # Rather than infer a filename from its naming convention, use
                # selection manifest as the authoritative graph identifier.
                for model in ('gcn','graphsage'):
                    manifest=selected[model].get(key)
                    if manifest is None:
                        raise ValueError('Missing selected configuration '+str((model,key)))
                    graph=manifest['graph']
                    path=ROOT/'results'/(model+'_node_predictions')/(graph+'.csv')
                    rows=load(path)
                    if len(rows)!=750:
                        raise ValueError(str(path)+': expected 750 test node predictions')
                    mapped={int(x['node_id']):x for x in rows}
                    if len(mapped)!=750 or set(mapped)!=set(source):
                        raise ValueError('Test-node mismatch: '+str((model,key)))
                    for nid, z in mapped.items():
                        ref=source[nid]
                        if (int(z['true_label']) != int(ref['true_label']) or
                                int(z['degree']) != int(ref['degree']) or
                                z['degree_bin'] != ref['degree_bin']):
                            raise ValueError('Graph/node mismatch '+str((model,key,nid)))
                        expected=int(int(z['true_label'])==int(z['predicted_label']))
                        if int(z['correct'])!=expected:
                            raise ValueError('Incorrect correctness flag '+str((model,key,nid)))
                    bymodel[model][key]=mapped
                for nid,z in source.items():
                    row={'family':fam,'target_homophily':h,'replicate':replicate,
                         'node_id':nid,'degree':int(z['degree']),
                         'degree_bin':z['degree_bin'],'true_label':int(z['true_label'])}
                    for model in MODELS:
                        item=bymodel[model][key][nid]
                        row[model+'_prediction']=int(item['predicted_label'])
                        row[model+'_correct']=int(item['correct'])
                    paired.append(row)

    if len(paired)!=22500:
        raise ValueError('Expected 22,500 paired test-node observations')

    records=[]
    for model in MODELS:
        for fam in FAMILIES:
            for h in LEVELS:
                for rep in (1,2,3):
                    key=(fam,h,rep)
                    rows=bymodel[model][key]
                    if len(rows)!=750:
                        raise ValueError('Missing graph-level test predictions '+str((model,key)))
                    total=sum(int(x['correct']) for x in rows.values())
                    actual=100*total/750
                    if model == 'h2gcn':
                        # Check against author-extracted checkpoint verification.
                        ck=checks_h2.get(key)
                        if ck is None:
                            raise ValueError('Missing H2GCN checkpoint check '+str(key))
                        expected=float(ck['recovered_test_accuracy_pct'])
                    else:
                        expected=float(selected[model][key]['test_accuracy_pct'])
                    if abs(actual-expected) > 0.05:
                        raise ValueError('Score mismatch %s: %.5f vs %.5f'%
                                         (str((model,key)),actual,expected))
                    accuracy_checks.append({'model':model,'family':fam,
                        'target_homophily':h,'replicate':rep,'test_nodes':750,
                        'correct':total,'recovered_test_accuracy_pct':round(actual,6),
                        'expected_test_accuracy_pct':round(expected,6),
                        'difference_pp':round(actual-expected,6),'status':'PASS'})
                    for b in BINS:
                        group=[z for z in rows.values() if z['degree_bin']==b]
                        if not group:
                            raise ValueError('Empty degree bin '+str((model,key,b)))
                        count=sum(int(z['correct']) for z in group)
                        records.append({'model':model,'family':fam,'target_homophily':h,
                            'replicate':rep,'degree_bin':b,'n_test':len(group),
                            'n_correct':count,'accuracy_pct':round(100*count/len(group),6)})

    sums=[]
    for model in MODELS:
        for fam in FAMILIES:
            for h in LEVELS:
                for b in BINS:
                    v=[z for z in records if z['model']==model and z['family']==fam
                       and z['target_homophily']==h and z['degree_bin']==b]
                    if len(v)!=3:
                        raise ValueError('Expected three graph repetitions '+str((model,fam,h,b)))
                    accu=[z['accuracy_pct'] for z in v]
                    n=sum(z['n_test'] for z in v)
                    c=sum(z['n_correct'] for z in v)
                    sums.append({'model':model,'family':fam,'target_homophily':h,
                       'degree_bin':b,'n_graphs':3,'total_test_observations':n,
                       'total_correct':c,'mean_test_nodes_per_graph':round(n/3,3),
                       'mean_accuracy_pct':round(statistics.mean(accu),6),
                       'sample_sd_accuracy_pp':round(statistics.stdev(accu),6),
                       'pooled_accuracy_pct':round(100*c/n,6)})

    # Bin-composition reweighting is DESCRIPTIVE, not causal.
    decompositions=[]
    for model in MODELS:
        for h in LEVELS:
            s={}
            for fam in FAMILIES:
                subset=[z for z in sums if z['model']==model and z['family']==fam
                        and z['target_homophily']==h]
                n=sum(z['total_test_observations'] for z in subset)
                w={z['degree_bin']:z['total_test_observations']/n for z in subset}
                a={z['degree_bin']:z['pooled_accuracy_pct'] for z in subset}
                s[fam]=(w,a)
            ws,asbm=s['sbm']; wd,adc=s['dcsbm']
            observed_sbm=sum(ws[b]*asbm[b] for b in BINS)
            observed_dcsbm=sum(wd[b]*adc[b] for b in BINS)
            standard_dcsbm=sum(ws[b]*adc[b] for b in BINS)
            decompositions.append({'model':model,'target_homophily':h,
                'sbm_observed_accuracy_pct':round(observed_sbm,6),
                'dcsbm_observed_accuracy_pct':round(observed_dcsbm,6),
                'dcsbm_standardized_to_sbm_degree_mix_pct':round(standard_dcsbm,6),
                'observed_dcsbm_minus_sbm_pp':round(observed_dcsbm-observed_sbm,6),
                'within_bin_standardized_difference_pp':round(standard_dcsbm-observed_sbm,6),
                'degree_bin_composition_component_pp':round(observed_dcsbm-standard_dcsbm,6),
                'interpretation':'descriptive; graph realizations are the independent units'})

    write('degree_robustness_accuracy_checks.csv',accuracy_checks)
    write('degree_robustness_three_model_runs.csv',records)
    write('degree_robustness_three_model_summary.csv',sums)
    write('degree_robustness_three_model_composition.csv',decompositions)
    write('degree_robustness_three_model_paired_nodes.csv',paired)
    print('PASS: 90 accuracy checks; 360 degree-bin run records; 120 degree-bin summaries;'
          ' 15 descriptive decompositions; 22,500 paired test-node rows.')


if __name__=='__main__':
    main()
