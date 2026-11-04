#!/usr/bin/env python3
"""One-time, test-only evaluation of a validation-selected LastFM H2GCN checkpoint.

Uses the original NPZ loader, H2GCN model and preprocessing from the prior
validation-only experiment. Does not train, tune, or change any checkpoints.
Fails closed if the frozen selection or checkpoint cannot be reproduced.
Test metrics are calculated only after frozen selection and validation checks.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ARCH = 'M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO'
GRAPH = 'existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz'
SPLIT = 'existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz'
SELECTION = 'results/lastfm_h2gcn_sensitivity/validation_selected_config.json'
SUMMARY = 'results/lastfm_h2gcn_sensitivity/validation_sensitivity_summary.csv'
CKPT = 'results/lastfm_h2gcn_sensitivity/l2small/best_checkpoint/ckpt'
OUTPUT = 'results/lastfm_h2gcn_sensitivity/selected_l2small_final_test'


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_csv(path, header, rows):
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        w = csv.writer(stream)
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(path)


def require_close(got, expected, abs_tol, name):
    if not math.isfinite(float(got)) or not math.isclose(float(got), float(expected), rel_tol=0.0, abs_tol=abs_tol):
        raise RuntimeError('{} mismatch: got {:.12g}, expected {:.12g}'.format(name, got, expected))


def preflight(selection_path, summary_path, checkpoint_prefix, out_dir):
    if out_dir.exists():
        raise RuntimeError('REFUSE: output directory already exists: {}. Final test is one-time; do not overwrite.'.format(out_dir))
    if not selection_path.is_file() or not summary_path.is_file():
        raise RuntimeError('REFUSE: selected config or sensitivity summary missing.')
    sel = json.loads(selection_path.read_text(encoding='utf-8'))
    selected = sel.get('selected') or {}
    if sel.get('candidate_count') != 6 or sel.get('test_metrics_examined_for_selection') is not False:
        raise RuntimeError('REFUSE: selection manifest is not the expected validation-only six-candidate selection.')
    if sel.get('seed') != 123 or sel.get('selection_metric') != 'maximum validation accuracy, minimum validation loss tie-break':
        raise RuntimeError('REFUSE: unexpected selection metric or model seed.')
    if sel.get('split') != SPLIT:
        raise RuntimeError('REFUSE: selected split differs from frozen split.')
    expected = {'config': 'l2small', 'hidden': 64, 'learning_rate': 0.01,
                'dropout': 0.5, 'weight_decay': 0.00005, 'best_epoch': 404}
    for key, val in expected.items():
        if selected.get(key) != val:
            raise RuntimeError('REFUSE: selected {} differs from the frozen experiment.'.format(key))
    recorded_ckpt = selected.get('checkpoint', '')
    if recorded_ckpt not in (CKPT, '/workspace/' + CKPT):
        raise RuntimeError('REFUSE: checkpoint differs from frozen selection: {}'.format(recorded_ckpt))
    if not Path(str(checkpoint_prefix) + '.index').is_file():
        raise RuntimeError('REFUSE: checkpoint index not found.')
    shards = list(checkpoint_prefix.parent.glob('ckpt.data-*'))
    if not shards:
        raise RuntimeError('REFUSE: checkpoint data shard not found.')
    with summary_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 6 or len({r['config'] for r in rows}) != 6:
        raise RuntimeError('REFUSE: six unique candidates are required.')
    ranked = sorted(rows, key=lambda r: (-float(r['val_accuracy']), float(r['best_val_loss'])))
    if ranked[0]['config'] != 'l2small':
        raise RuntimeError('REFUSE: summary winner is not l2small.')
    row = ranked[0]
    for key in ('val_accuracy', 'val_macro_f1', 'best_val_loss'):
        require_close(float(row[key]), float(selected[key]), 1e-11, key + ' summary')
    if int(row['best_epoch']) != 404:
        raise RuntimeError('REFUSE: summary best epoch differs.')
    print('PASS: frozen validation selection verified; selected=l2small, epoch=404', flush=True)
    print('Checkpoint:', str(checkpoint_prefix), flush=True)
    print('No training or hyperparameter selection will be performed.', flush=True)
    return sel, selected, shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Preflight only; never load labels or evaluate a model.')
    args = parser.parse_args()

    root = Path.cwd()
    selection_path = root / SELECTION
    summary_path = root / SUMMARY
    checkpoint_prefix = root / CKPT
    output_path = root / OUTPUT
    sel, selected, shards = preflight(selection_path, summary_path, checkpoint_prefix, output_path)
    if args.check_only:
        print('PASS: preflight-only. No test labels accessed or model evaluated.', flush=True)
        return

    sys.path.insert(0, str(root / 'scripts'))
    from train_lastfm_h2gcn_sensitivity import classification_scores, graph_and_split_metadata

    graph = root / GRAPH
    split = root / SPLIT
    if not graph.is_file() or not split.is_file():
        raise RuntimeError('REFUSE: frozen LastFM graph or split missing.')
    y, idx, meta = graph_and_split_metadata(graph, split)
    if (meta['nodes'], meta['edges'], meta['features'], meta['classes']) != (7624, 27806, 7842, 18):
        raise RuntimeError('REFUSE: LastFM graph metadata unexpected.')
    if meta['split_sizes'] != {'train': 1898, 'val': 1898, 'test': 3828}:
        raise RuntimeError('REFUSE: test split cardinality unexpected.')

    import tensorflow as tf
    sys.path.insert(0, str(root / 'h2gcn'))
    from datasets.npz import NPZData
    from models import parse_network_setup
    from models.H2GCN import H2GCN

    np.random.seed(123)
    tf.random.set_seed(123)
    tf.config.experimental_run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    data = NPZData(str(graph.resolve()), str(split.resolve()))
    data.row_normalize_features()
    data.adj_remove_eye()
    tensors = vars(data.getTensors(getDenseAdj=False, getAdjNormHops=['1', '2']))
    setup = parse_network_setup(ARCH, data.num_labels, _dense_units=64,
                                _dropout_rate=0.5, parse_preprocessing=True)
    model = H2GCN(setup, l2_regularize_weight=0.00005)
    _ = model(tensors['adj'], tensors['features'], tensors['adj_hops'], training=False)

    saved_variables = [(name, tuple(shape)) for name, shape in tf.train.list_variables(str(checkpoint_prefix))
                       if name.startswith('model/') and name.endswith('/.ATTRIBUTES/VARIABLE_VALUE')]
    expected_shapes = collections.Counter(tuple(int(d) for d in v.shape) for v in model.weights)
    saved_shapes = collections.Counter(shape for _, shape in saved_variables)
    if not saved_variables or expected_shapes != saved_shapes:
        raise RuntimeError('REFUSE: model variable shape coverage mismatch.')

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(str(checkpoint_prefix)).expect_partial()
    reference_weights = [v.numpy().copy() for v in model.weights]
    for v in model.weights:
        v.assign(tf.zeros_like(v) + tf.cast(3.14, v.dtype))
    checkpoint.restore(str(checkpoint_prefix)).expect_partial()
    for v, ref in zip(model.weights, reference_weights):
        if not np.array_equal(v.numpy(), ref):
            raise RuntimeError('REFUSE: repeated checkpoint restoration mismatch.')

    logits = model(tensors['adj'], tensors['features'], tensors['adj_hops'], training=False).numpy()
    if not np.all(np.isfinite(logits)):
        raise RuntimeError('REFUSE: nonfinite logits.')
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    val = classification_scores(y[idx['val']], predictions[idx['val']], meta['classes'])
    val_loss = float(model._loss(tf.convert_to_tensor(logits), tensors['y_val'], tensors['val_mask']).numpy())
    require_close(val['accuracy'], selected['val_accuracy'], 1e-10, 'Restored validation accuracy')
    require_close(val['macro_f1'], selected['val_macro_f1'], 1e-10, 'Restored validation macro-F1')
    require_close(val_loss, selected['best_val_loss'], 1e-4, 'Restored validation loss')
    print('PASS: restored validation accuracy {:.4%}, macro-F1 {:.4%}, loss {:.5f}'.format(
        val['accuracy'], val['macro_f1'], val_loss), flush=True)

    # Frozen selection and checkpoint have now been verified.
    # This is the first and only point where test labels are scored.
    test = classification_scores(y[idx['test']], predictions[idx['test']], meta['classes'])
    baseline_majority_class = int(np.argmax(np.bincount(y[idx['train']], minlength=meta['classes'])))
    majority_accuracy = float(np.mean(y[idx['test']] == baseline_majority_class))
    n_predicted = sum(r['predicted_count'] > 0 for r in test['per_class'])
    n_zero_recall = sum(r['recall'] == 0 for r in test['per_class'])

    record = {
        'dataset': 'LastFM-Asia', 'model': 'H2GCN-2', 'selection': 'validation-only, six candidates',
        'selected_config': 'l2small', 'seed': 123, 'split_seed': 42,
        'hidden': 64, 'learning_rate': 0.01, 'dropout': 0.5, 'l2_regularization': 0.00005,
        'best_epoch': 404, 'val_accuracy': val['accuracy'], 'val_macro_f1': val['macro_f1'],
        'val_loss': val_loss, 'test_accuracy': test['accuracy'], 'test_macro_f1': test['macro_f1'],
        'test_weighted_f1': test['weighted_f1'], 'test_balanced_accuracy': test['balanced_accuracy'],
        'test_classes_predicted': n_predicted, 'test_classes_zero_recall': n_zero_recall,
        'test_class_count': 18, 'test_node_count': len(idx['test']),
        'majority_class_test_accuracy': majority_accuracy,
        'checkpoint_prefix': CKPT, 'selection_file_sha256': sha256(selection_path),
        'sensitivity_summary_sha256': sha256(summary_path),
        'checkpoint_index_sha256': sha256(Path(str(checkpoint_prefix) + '.index')),
        'checkpoint_data_shard_sha256': {p.name: sha256(p) for p in shards},
        'graph_sha256': sha256(graph), 'split_sha256': sha256(split),
        'selection_based_on_test_metrics': False,
        'note': 'One fixed held-out test evaluation of the validation-selected checkpoint; training seeds are not independent graph splits.',
    }
    # Output directory is created only after all validation checks pass and test scoring finishes.
    output_path.mkdir(parents=True, exist_ok=False)
    save_csv(output_path / 'test_node_predictions.csv',
             ['node_id', 'true_label', 'predicted_label', 'correct'],
             ((int(i), int(y[i]), int(predictions[i]), int(y[i] == predictions[i]))
              for i in idx['test']))
    save_csv(output_path / 'test_class_metrics.csv',
             ['class_id', 'test_support', 'predicted_count', 'precision', 'recall', 'f1'],
             ((r['label'], r['support'], r['predicted_count'], r['precision'], r['recall'], r['f1'])
              for r in test['per_class']))
    with (output_path / 'selected_test_results.json.partial').open('w', encoding='utf-8') as file:
        json.dump(record, file, indent=2)
    (output_path / 'selected_test_results.json.partial').replace(output_path / 'selected_test_results.json')

    print('TEST (once): accuracy={:.2%} macro-F1={:.2%} weighted-F1={:.2%} balanced-acc={:.2%}'.format(
        test['accuracy'], test['macro_f1'], test['weighted_f1'], test['balanced_accuracy']), flush=True)
    print('Predicted classes: {}/18 | Zero-recall classes: {}'.format(n_predicted, n_zero_recall), flush=True)
    print('PASS: selected checkpoint evaluated; results in {}'.format(output_path), flush=True)
    print('This script will refuse to overwrite these final-test outputs.', flush=True)


if __name__ == '__main__':
    main()
