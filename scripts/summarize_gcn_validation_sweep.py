#!/usr/bin/env python3
"""Validation-select GCN among 9 configs per graph and compare with H2GCN.

Run after run_gcn_validation_sweep.sh has completed. Requires only Python stdlib.
This does not use test accuracy to select hyperparameters.
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FAMILIES = ('sbm', 'dcsbm')
HOMOPHILIES = ('0.10', '0.30', '0.50', '0.70', '0.90')


def graph_path(family, h, replicate):
    folder = ROOT / 'constructed_data' / (family + '_grid')
    matches = sorted(folder.glob('%s_h%s_r%d_seed*.npz' % (family, h, replicate)))
    if len(matches) != 1:
        raise RuntimeError('Expected one graph for %s h=%s r=%d: %s' %
                           (family, h, replicate, matches))
    return matches[0]


def result_path(graph, hidden, patience):
    name = graph.stem
    if hidden == 64 and patience == 200:
        if name == 'sbm_h0.10_r1_seed2027':
            return RESULTS / 'gcn_sbm_h010_r1_full.json'
        return RESULTS / ('gcn_%s_h64_seed123_full.json' % name)
    return RESULTS / 'gcn_tuning' / (
        '%s_h%d_p%d_s123.json' % (name, hidden, patience))


def load_result(graph, hidden, patience):
    path = result_path(graph, hidden, patience)
    if not path.is_file():
        raise RuntimeError('Missing GCN result: %s' % path)
    with path.open() as f:
        record = json.load(f)
    expected_graph = str(graph.relative_to(ROOT))
    assert record['graph'] == expected_graph, (path, record['graph'])
    assert int(record['hidden']) == hidden, path
    assert int(record['patience']) == patience, path
    assert int(record['seed']) == 123, path
    assert record['split'] == 'constructed_data/shared_split_seed42.npz', path
    assert 0 <= record['best_val_accuracy'] <= 1, path
    assert 0 <= record['test_accuracy'] <= 1, path
    return record


def write_csv(path, cols, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', path.relative_to(ROOT))


with (RESULTS / 'h2gcn_grid_runs.csv').open(newline='') as f:
    h2 = {(r['family'], '%.2f' % float(r['homophily']), int(r['replicate'])):
          float(r['test_accuracy'])
          for r in csv.DictReader(f)}

chosen_rows = []
all_candidates = []
summary = []
for family in FAMILIES:
    for h in HOMOPHILIES:
        for replicate in (1, 2, 3):
            key = (family, h, replicate)
            if key not in h2:
                raise RuntimeError('Missing H2GCN result: %s' % (key,))
            graph = graph_path(*key)
            candidates = []
            for hidden in (16, 32, 64):
                for patience in (40, 100, 200):
                    r = load_result(graph, hidden, patience)
                    candidate = dict(family=family, homophily=h,
                                     replicate=replicate, graph=graph.stem,
                                     hidden=hidden, patience=patience,
                                     best_val_accuracy_pct=100*r['best_val_accuracy'],
                                     best_val_loss=r['best_val_loss'],
                                     test_accuracy_pct=100*r['test_accuracy'])
                    candidates.append(candidate)
                    all_candidates.append(candidate)
            # Select by validation accuracy, breaking ties with lower validation loss.
            selected = max(candidates, key=lambda r:
                           (r['best_val_accuracy_pct'], -r['best_val_loss']))
            chosen_rows.append(dict(**selected, h2gcn_test_accuracy_pct=h2[key],
                                    h2gcn_minus_gcn_pct_points=
                                    h2[key]-selected['test_accuracy_pct']))

for family in FAMILIES:
    for h in HOMOPHILIES:
        rows = [r for r in chosen_rows
                if r['family'] == family and r['homophily'] == h]
        if len(rows) != 3:
            raise RuntimeError('Expected exactly three graphs for %s h=%s' % (family, h))
        h2scores = [r['h2gcn_test_accuracy_pct'] for r in rows]
        gcnscores = [r['test_accuracy_pct'] for r in rows]
        gaps = [r['h2gcn_minus_gcn_pct_points'] for r in rows]
        summary.append(dict(family=family, homophily=h, n=3,
                            h2gcn_mean_pct=statistics.mean(h2scores),
                            h2gcn_sd_pp=statistics.stdev(h2scores),
                            gcn_tuned_mean_pct=statistics.mean(gcnscores),
                            gcn_tuned_sd_pp=statistics.stdev(gcnscores),
                            mean_h2gcn_advantage_pp=statistics.mean(gaps),
                            sd_paired_advantage_pp=statistics.stdev(gaps)))

write_csv(RESULTS / 'gcn_all_validation_candidates.csv',
          list(all_candidates[0]), all_candidates)
write_csv(RESULTS / 'gcn_selected_by_validation.csv',
          list(chosen_rows[0]), chosen_rows)
write_csv(RESULTS / 'h2gcn_vs_gcn_validation_selected.csv',
          list(summary[0]), summary)

print('\nFULL VALIDATION-SELECTED GCN COMPARISON')
for r in summary:
    print('%5s h=%s | H2GCN %5.2f ± %4.2f | GCN %5.2f ± %4.2f | gap %+5.2f pp' %
          (r['family'], r['homophily'], r['h2gcn_mean_pct'],
           r['h2gcn_sd_pp'], r['gcn_tuned_mean_pct'],
           r['gcn_tuned_sd_pp'], r['mean_h2gcn_advantage_pp']))
print('\nPASS: 270 GCN configs verified; 30 validation-selected graph results.')
