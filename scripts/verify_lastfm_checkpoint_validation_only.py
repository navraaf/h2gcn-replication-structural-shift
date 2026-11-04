#!/usr/bin/env python3
"""Validate the frozen LastFM H2GCN checkpoint in a clean Docker image.

Read-only and validation-only. Does not train, score held-out test labels,
modify checkpoint files, or overwrite any existing final-test result.
Requires the project's published evaluator and training helpers.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_lastfm_h2gcn_selected_once import (
    ARCH, CKPT, GRAPH, SELECTION, SPLIT, SUMMARY,
    preflight, require_close, sha256,
)
from train_lastfm_h2gcn_sensitivity import (
    classification_scores, graph_and_split_metadata,
)


def main():
    # The original one-time evaluator refuses any existing final-test directory.
    # Use an unused, never-created path so preflight can verify the frozen
    # selection and checkpoint without touching final-test artifacts.
    selection, chosen, shards = preflight(
        ROOT / SELECTION,
        ROOT / SUMMARY,
        ROOT / CKPT,
        Path('/tmp/h2gcn_validation_check_only_unused_output'),
    )

    final_result = ROOT / 'results/lastfm_h2gcn_sensitivity/selected_l2small_final_test/selected_test_results.json'
    if final_result.is_file():
        evidence = json.loads(final_result.read_text(encoding='utf-8'))
        digests = {
            'selection_file_sha256': ROOT / SELECTION,
            'sensitivity_summary_sha256': ROOT / SUMMARY,
            'checkpoint_index_sha256': ROOT / (CKPT + '.index'),
            'graph_sha256': ROOT / GRAPH,
            'split_sha256': ROOT / SPLIT,
        }
        for field, path in digests.items():
            actual_digest = sha256(path)
            if evidence.get(field) != actual_digest:
                if field == 'sensitivity_summary_sha256':
                    raw = path.read_bytes()
                    old_crlf = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                    if hashlib.sha256(old_crlf).hexdigest() == evidence[field]:
                        print('NOTE: sensitivity summary changed only CRLF to LF during Git cleanup.', flush=True)
                        continue
                raise RuntimeError('Previously recorded {} differs from current file'.format(field))
        for shard in shards:
            recorded = evidence.get('checkpoint_data_shard_sha256', {}).get(shard.name)
            if recorded != sha256(shard):
                raise RuntimeError('Previously recorded checkpoint shard differs: {}'.format(shard.name))
        print('PASS: selection, graph, split, and checkpoint hashes match prior final-test provenance.', flush=True)

    graph_path, split_path = ROOT / GRAPH, ROOT / SPLIT
    if not graph_path.is_file() or not split_path.is_file():
        raise RuntimeError('Frozen graph or split missing.')
    y, idx, meta = graph_and_split_metadata(graph_path, split_path)
    if (meta['nodes'], meta['edges'], meta['features'], meta['classes']) != (7624, 27806, 7842, 18):
        raise RuntimeError('Unexpected LastFM graph dimensions.')
    if meta['split_sizes'] != {'train': 1898, 'val': 1898, 'test': 3828}:
        raise RuntimeError('Unexpected split cardinalities.')

    import tensorflow as tf
    sys.path.insert(0, str(ROOT / 'h2gcn'))
    from datasets.npz import NPZData
    from models import parse_network_setup
    from models.H2GCN import H2GCN

    np.random.seed(123)
    tf.random.set_seed(123)
    tf.config.experimental_run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    data = NPZData(str(graph_path.resolve()), str(split_path.resolve()))
    data.row_normalize_features()
    data.adj_remove_eye()
    t = vars(data.getTensors(getDenseAdj=False, getAdjNormHops=['1', '2']))
    setup = parse_network_setup(ARCH, data.num_labels, _dense_units=64,
                                _dropout_rate=0.5, parse_preprocessing=True)
    model = H2GCN(setup, l2_regularize_weight=0.00005)
    _ = model(t['adj'], t['features'], t['adj_hops'], training=False)

    checkpoint = ROOT / CKPT
    saved_variables = [
        (name, tuple(shape))
        for name, shape in tf.train.list_variables(str(checkpoint))
        if name.startswith('model/') and name.endswith('/.ATTRIBUTES/VARIABLE_VALUE')
    ]
    saved_shapes = collections.Counter(shape for _, shape in saved_variables)
    weight_shapes = collections.Counter(tuple(int(d) for d in v.shape) for v in model.weights)
    if not saved_variables or saved_shapes != weight_shapes:
        raise RuntimeError('Checkpoint variable coverage mismatch.')

    restore = tf.train.Checkpoint(model=model)
    restore.restore(str(checkpoint)).expect_partial()
    restored_weights = [v.numpy().copy() for v in model.weights]
    for variable in model.weights:
        variable.assign(tf.zeros_like(variable) + tf.cast(3.14, variable.dtype))
    restore.restore(str(checkpoint)).expect_partial()
    for variable, expected in zip(model.weights, restored_weights):
        if not np.array_equal(variable.numpy(), expected):
            raise RuntimeError('Checkpoint restoration was not repeatable.')

    logits_tensor = model(t['adj'], t['features'], t['adj_hops'], training=False)
    logits = logits_tensor.numpy()
    if not np.all(np.isfinite(logits)):
        raise RuntimeError('Nonfinite model logits.')
    pred = np.argmax(logits, axis=1).astype(np.int64)
    val = classification_scores(y[idx['val']], pred[idx['val']], meta['classes'])
    loss = float(model._loss(logits_tensor, t['y_val'], t['val_mask']).numpy())

    require_close(val['accuracy'], chosen['val_accuracy'], 1e-10, 'Validation accuracy')
    require_close(val['macro_f1'], chosen['val_macro_f1'], 1e-10, 'Validation macro-F1')
    require_close(loss, chosen['best_val_loss'], 1e-4, 'Validation loss')
    print('PASS: restored checkpoint variables match and restore repeatably.', flush=True)
    print('PASS: validation accuracy {:.4%}, macro-F1 {:.4%}, loss {:.5f}'.format(
        val['accuracy'], val['macro_f1'], loss), flush=True)
    print('PASS: clean Docker image reproduced the frozen LastFM VALIDATION metrics.', flush=True)
    print('No training, held-out test scoring, final-test overwrite, or file modification performed.', flush=True)


if __name__ == '__main__':
    main()
