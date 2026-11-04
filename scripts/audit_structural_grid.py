#!/usr/bin/env python3
"""COMP8240: reproducible quality audit for the 30 synthetic SBM/DC-SBM graphs.

Run inside ~/Documents/H2GCN, using NumPy and SciPy (or the existing Docker image).
Produces per-replicate and grouped structural CSVs without modifying datasets.
"""
import argparse
import csv
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

GRAPH_RE = re.compile(r"^(sbm|dcsbm)_h(0\.(?:10|30|50|70|90))_r([123])_seed(\d+)\.npz$")


def read_csr(archive, name):
    keys = (f"{name}_data", f"{name}_indices", f"{name}_indptr", f"{name}_shape")
    missing = [key for key in keys if key not in archive]
    if missing:
        raise ValueError(f"Missing {missing}")
    array = sp.csr_matrix((archive[keys[0]], archive[keys[1]], archive[keys[2]]),
                          shape=tuple(archive[keys[3]]))
    array.sum_duplicates()
    array.sort_indices()
    return array


def sha_array(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def sha_csr(mat):
    mat = mat.tocsr(copy=True)
    mat.sum_duplicates()
    mat.sort_indices()
    h = hashlib.sha256()
    h.update(str(mat.shape).encode())
    for data in (mat.indptr, mat.indices, mat.data):
        h.update(sha_array(data).encode())
    return h.hexdigest()


def sample_sd(items):
    if len(items) < 2:
        return float('nan')
    return float(np.std(np.array(items, dtype=np.float64), ddof=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.', help='H2GCN project root')
    p.add_argument('--outdir', default='results')
    p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args()
    root = Path(args.root)
    out = root / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    split_file = root / 'constructed_data/shared_split_seed42.npz'
    with np.load(split_file, allow_pickle=False) as archive:
        split = {k: np.asarray(archive[k], dtype=np.int64).reshape(-1)
                 for k in ('idx_train', 'idx_val', 'idx_test')}
    graph_files = sorted((root / 'constructed_data').glob('*_grid/*.npz'))
    if not graph_files:
        raise FileNotFoundError('No *.npz graphs under constructed_data/*_grid')
    rows = []
    reference = None
    observed_keys = set()
    for path in graph_files:
        match = GRAPH_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f'Unexpected graph filename: {path.name}')
        family, target, replicate, seed = match.groups()
        key = (family, target, int(replicate))
        if key in observed_keys:
            raise ValueError(f'Duplicate graph condition: {key}')
        observed_keys.add(key)
        with np.load(path, allow_pickle=False) as z:
            adj = read_csr(z, 'adj')
            features = read_csr(z, 'attr')
            labels = np.asarray(z['labels'], dtype=np.int64).reshape(-1)
        n = len(labels)
        if adj.shape != (n, n) or features.shape[0] != n:
            raise ValueError(f'Invalid graph dimensions: {path}')
        if np.any(~np.isfinite(features.data)) or np.any(~np.isfinite(adj.data)):
            raise ValueError(f'Nonfinite inputs: {path}')
        if not np.allclose(adj.diagonal(), 0):
            raise ValueError(f'Self loops in supplied graph: {path}')
        if (adj != adj.transpose()).nnz:
            raise ValueError(f'Asymmetric adjacency: {path}')
        if np.any(adj.data != 1):
            raise ValueError(f'Expected simple unweighted binary adjacency: {path}')
        if n != 1490 or features.shape[1] != 1433 or len(np.unique(labels)) != 5:
            raise ValueError(f'Unexpected dataset dimensions or classes: {path}')
        ids = np.concatenate(list(split.values()))
        if len(ids) != n or len(np.unique(ids)) != n or set(ids) != set(range(n)):
            raise ValueError('The shared split is not a node partition')
        signature = (sha_csr(features), sha_array(labels))
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f'Feature or label matrices differ across graphs: {path}')
        degree = np.asarray(adj.sum(axis=1)).reshape(-1)
        undirected_edges = int(adj.nnz // 2)
        if undirected_edges == 0:
            raise ValueError(f'Empty graph: {path}')
        upper = sp.triu(adj, k=1).tocoo()
        same_class_edges = int(np.count_nonzero(labels[upper.row] == labels[upper.col]))
        n_components, component_ids = connected_components(adj, directed=False, return_labels=True)
        components = np.bincount(component_ids, minlength=n_components)
        isolated = (degree == 0)
        row = {
            'family': family, 'target_h': target, 'replicate': int(replicate),
            'generator_seed': int(seed), 'graph': str(path.relative_to(root)),
            'n_nodes': n, 'n_features': features.shape[1],
            'n_classes': len(np.unique(labels)), 'edges': undirected_edges,
            'actual_edge_homophily': same_class_edges / undirected_edges,
            'homophily_minus_target': same_class_edges / undirected_edges - float(target),
            'mean_degree': float(np.mean(degree)),
            'median_degree': float(np.median(degree)),
            'degree_cv': float(np.std(degree, ddof=0) / np.mean(degree)),
            'degree_p90': float(np.quantile(degree, 0.90)),
            'degree_p99': float(np.quantile(degree, 0.99)),
            'max_degree': float(np.max(degree)),
            'isolated_nodes': int(np.sum(isolated)),
            'isolated_train': int(np.sum(isolated[split['idx_train']])),
            'isolated_val': int(np.sum(isolated[split['idx_val']])),
            'isolated_test': int(np.sum(isolated[split['idx_test']])),
            'num_components': int(n_components),
            'largest_component_pct': 100.0 * float(max(components)) / n,
            'feature_sha256': signature[0], 'label_sha256': signature[1],
        }
        if row['isolated_nodes'] != row['isolated_train'] + row['isolated_val'] + row['isolated_test']:
            raise AssertionError('Isolated-node split counts inconsistent')
        rows.append(row)
    expected = {(f, h, r) for f in ('sbm', 'dcsbm')
                for h in ('0.10', '0.30', '0.50', '0.70', '0.90') for r in (1, 2, 3)}
    if not args.allow_incomplete and observed_keys != expected:
        raise ValueError(f'Missing {sorted(expected - observed_keys)}, extra {sorted(observed_keys - expected)}')
    rows.sort(key=lambda x: (x['family'], float(x['target_h']), x['replicate']))
    output_runs = out / 'structural_grid_runs.csv'
    with output_runs.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ('edges', 'actual_edge_homophily', 'homophily_minus_target',
               'mean_degree', 'degree_cv', 'degree_p90', 'degree_p99',
               'max_degree', 'isolated_nodes', 'isolated_test',
               'num_components', 'largest_component_pct')
    summaries = []
    for family in ('sbm', 'dcsbm'):
        for h in ('0.10', '0.30', '0.50', '0.70', '0.90'):
            subset = [r for r in rows if r['family'] == family and r['target_h'] == h]
            if not subset:
                continue
            summary = {'family': family, 'target_h': h, 'n': len(subset)}
            for m in metrics:
                values = [r[m] for r in subset]
                summary[f'{m}_mean'] = float(np.mean(values))
                summary[f'{m}_sd'] = sample_sd(values)
            summaries.append(summary)
    output_summary = out / 'structural_grid_summary.csv'
    with output_summary.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f'PASS: {len(rows)} graphs; {len(summaries)} conditions; fixed labels and features across graphs.')
    print('FAMILY  h    measured_h  degree_CV  isolates  LCC (%)')
    for r in summaries:
        print(f"{r['family']:5s} {r['target_h']:4s}   "
              f"{r['actual_edge_homophily_mean']:.3f}       "
              f"{r['degree_cv_mean']:.3f}       "
              f"{r['isolated_nodes_mean']:.1f}     "
              f"{r['largest_component_pct_mean']:.2f}")
    print(f'Saved: {output_runs}')
    print(f'Saved: {output_summary}')


if __name__ == '__main__':
    main()
