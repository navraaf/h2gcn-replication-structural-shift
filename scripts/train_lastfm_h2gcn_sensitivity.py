#!/usr/bin/env python3
"""Validation-selected H2GCN-2 evaluation for LastFM-Asia (TF 2.2 / Python 3.8).

Uses the fork's exact NPZData loader, H2GCN class, network specification,
feature normalisation and adjacency-hop preprocessing. Never evaluates test
labels until the best checkpoint has been selected on validation data.

This is a purpose-built evaluation driver, NOT the unmodified reference runner.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ARCH = 'M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO'


def csr_from_npz(data, name):
    return sp.csr_matrix((data[name + '_data'], data[name + '_indices'],
                          data[name + '_indptr']), shape=tuple(data[name + '_shape']))


def graph_and_split_metadata(graph_file, split_file):
    with np.load(str(graph_file), allow_pickle=False) as z:
        adj = csr_from_npz(z, 'adj')
        feat = csr_from_npz(z, 'attr')
        y = np.asarray(z['labels'], dtype=np.int64)
    with np.load(str(split_file), allow_pickle=False) as z:
        idx = {p: np.asarray(z['idx_' + p], dtype=np.int64)
               for p in ('train', 'val', 'test')}
    n = len(y)
    if adj.shape != (n, n) or feat.shape[0] != n:
        raise ValueError('Graph, features and labels have incompatible shapes')
    if (adj != adj.T).nnz:
        raise ValueError('Adjacency is not symmetric')
    if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))):
        raise ValueError('Labels must be consecutive zero-based integers')
    merged = np.concatenate(list(idx.values()))
    if (len(merged) != n or len(np.unique(merged)) != n or
            merged.min() < 0 or merged.max() >= n):
        raise ValueError('The train/val/test indices must partition nodes exactly once')
    degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    support = {}
    for p, ids in idx.items():
        support[p] = np.bincount(y[ids], minlength=len(np.unique(y))).astype(int).tolist()
    return y, idx, {
        'nodes': n, 'edges': int(adj.nnz // 2),
        'features': int(feat.shape[1]), 'classes': int(len(np.unique(y))),
        'isolated_nodes': int(np.count_nonzero(degree == 0)),
        'zero_feature_nodes': int(np.count_nonzero(np.asarray(feat.sum(axis=1)).ravel() == 0)),
        'split_sizes': {p: int(len(v)) for p, v in idx.items()},
        'class_support': support,
    }


def classification_scores(true, predicted, k):
    true = np.asarray(true, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if not len(true) or true.shape != predicted.shape:
        raise ValueError('Cannot score empty or mismatched predictions')
    support = np.bincount(true, minlength=k).astype(np.float64)
    predicted_support = np.bincount(predicted, minlength=k).astype(np.float64)
    tp = np.bincount(true[true == predicted], minlength=k).astype(np.float64)
    # F1 = 2 TP / (2 TP + FP + FN) = 2 TP / (predicted count + true count).
    f1 = np.divide(2 * tp, support + predicted_support,
                   out=np.zeros(k, dtype=np.float64), where=(support + predicted_support) > 0)
    recall = np.divide(tp, support, out=np.zeros(k, dtype=np.float64), where=support > 0)
    precision = np.divide(tp, predicted_support,
                          out=np.zeros(k, dtype=np.float64), where=predicted_support > 0)
    return {
        'accuracy': float(np.mean(true == predicted)),
        'macro_f1': float(np.mean(f1)),
        'weighted_f1': float(np.sum(f1 * support) / len(true)),
        'balanced_accuracy': float(np.mean(recall[support > 0])),
        'per_class': [dict(label=i, support=int(support[i]), predicted_count=int(predicted_support[i]),
                           precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]))
                      for i in range(k)],
    }


def save_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', default='existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz')
    parser.add_argument('--split', default='existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--hidden', type=int, choices=[32, 64, 128], default=64)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--validation-only', action='store_true',
                        help='Train using train/val only; never score test set')
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.log_every < 1:
        parser.error('epochs, patience and log-every must be positive')
    if not 0 <= args.dropout < 1 or not args.learning_rate > 0 or args.weight_decay < 0:
        parser.error('Invalid dropout, learning rate or weight decay')
    architecture = 'M{}-R-T1-G-V-T2-G-V-C1-C2-D{}-MO'.format(args.hidden, args.dropout)
    graph_file = Path(args.graph).resolve()
    split_file = Path(args.split).resolve()
    if not graph_file.is_file() or not split_file.is_file():
        parser.error('Missing preprocessed LastFM graph or split; see lastfm_full_run_audit.txt')

    y, idx, metadata = graph_and_split_metadata(graph_file, split_file)
    print('LastFM-Asia: N={nodes}, E={edges}, F={features}, C={classes}'.format(**metadata), flush=True)
    print('Train/val/test: {}'.format(metadata['split_sizes']), flush=True)
    print('Zero-feature nodes: {}; isolated nodes: {}'.format(
        metadata['zero_feature_nodes'], metadata['isolated_nodes']), flush=True)
    print('Smallest class count in test split:', min(metadata['class_support']['test']), flush=True)
    if args.check_only:
        print('PASS: LastFM NPZ and split are structurally valid; training NOT started.', flush=True)
        return

    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty run directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    sys.path.insert(0, str(Path.cwd() / 'h2gcn'))
    from datasets.npz import NPZData
    from models import parse_network_setup
    from models.H2GCN import H2GCN
    from models._metrics import masked_softmax_cross_entropy, masked_accuracy

    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.experimental_run_functions_eagerly(True)
    tf.keras.backend.clear_session()

    data = NPZData(str(graph_file), str(split_file))
    data.row_normalize_features()
    data.adj_remove_eye()
    tensors = vars(data.getTensors(getDenseAdj=False, getAdjNormHops=['1', '2']))
    layers = parse_network_setup(architecture, data.num_labels,
                                 _dense_units=args.hidden, _dropout_rate=args.dropout,
                                 parse_preprocessing=True)
    model = H2GCN(layers, l2_regularize_weight=args.weight_decay)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Build variables before training so checkpoint/weight ordering stays fixed.
    _ = model(tensors['adj'], tensors['features'], tensors['adj_hops'], training=False)
    # Save an actual TensorFlow checkpoint at each validation improvement.
    # This preserves all trackable model state, not only trainable variables.
    # The checkpoint is overwritten at the same path to avoid disk growth.
    best_checkpoint_dir = out / 'best_checkpoint'
    best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_prefix = str(best_checkpoint_dir / 'ckpt')
    best_checkpoint = tf.train.Checkpoint(model=model)
    best_checkpoint_written = False
    best_epoch = 0
    best_val_accuracy = -1.
    best_val_loss = float('inf')
    train_trace = []
    class_count = metadata['classes']
    print('Model: H2GCN-2; seed={}; max_epochs={}; patience={}; hidden={}; lr={}; dropout={}; L2={}'.format(
        args.seed, args.epochs, args.patience, args.hidden, args.learning_rate, args.dropout,
        args.weight_decay), flush=True)
    print('Selection: maximum validation accuracy; validation loss breaks ties.', flush=True)
    if args.validation_only:
        print('VALIDATION-ONLY: test labels will NOT be scored.', flush=True)
    else:
        print('Test labels are evaluated only AFTER selection.', flush=True)

    for epoch in range(1, args.epochs + 1):
        with tf.GradientTape() as tape:
            train_logits = model(tensors['adj'], tensors['features'],
                                 tensors['adj_hops'], training=True)
            train_loss_tensor = model._loss(train_logits,
                                            tensors['y_train'], tensors['train_mask'])
        gradients = tape.gradient(train_loss_tensor, model.trainable_variables)
        if any(g is None for g in gradients):
            raise RuntimeError('Missing gradient for trainable H2GCN variable')
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        val_logits = model(tensors['adj'], tensors['features'],
                           tensors['adj_hops'], training=False)
        val_acc = float(masked_accuracy(val_logits, tensors['y_val'], tensors['val_mask']).numpy())
        val_loss = float(model._loss(val_logits, tensors['y_val'], tensors['val_mask']).numpy())
        train_loss = float(train_loss_tensor.numpy())
        if not all(math.isfinite(v) for v in (train_loss, val_acc, val_loss)):
            raise FloatingPointError('Nonfinite train/validation metrics at epoch {}'.format(epoch))
        improved = (val_acc > best_val_accuracy + 1e-7 or
                    (abs(val_acc - best_val_accuracy) <= 1e-7 and
                     val_loss < best_val_loss - 1e-9))
        if improved:
            best_epoch = epoch
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            best_checkpoint.write(best_checkpoint_prefix)
            best_checkpoint_written = True
        train_trace.append([epoch, train_loss, val_loss, val_acc, int(improved)])
        if epoch <= 5 or epoch % args.log_every == 0:
            print('Epoch {:4d} | train_loss {:.4f} | val_loss {:.4f} | val_acc {:.2%}{}'.format(
                epoch, train_loss, val_loss, val_acc, ' *' if improved else ''), flush=True)
        if epoch - best_epoch >= args.patience:
            print('Early stopping at epoch {}'.format(epoch), flush=True)
            break

    if not best_checkpoint_written:
        raise RuntimeError('No checkpoint chosen; refusing to evaluate test split')
    # Restore the checkpoint selected during training. Never fall back to
    # last-epoch weights if the best checkpoint cannot be restored faithfully.
    restore_status = best_checkpoint.restore(best_checkpoint_prefix)
    restore_status.assert_existing_objects_matched()
    final = model(tensors['adj'], tensors['features'], tensors['adj_hops'], training=False)
    prediction = np.argmax(final.numpy(), axis=1).astype(np.int64)
    val_metrics = classification_scores(y[idx['val']], prediction[idx['val']], class_count)
    restored_masked_val_acc = float(masked_accuracy(
        final, tensors['y_val'], tensors['val_mask']).numpy())
    if abs(val_metrics['accuracy'] - best_val_accuracy) > 1e-6 or abs(restored_masked_val_acc - best_val_accuracy) > 1e-6:
        print('CHECKPOINT MISMATCH: selected={:.8f}, restored_classification={:.8f}, restored_masked={:.8f}, best_epoch={}'.format(
            best_val_accuracy, val_metrics['accuracy'], restored_masked_val_acc, best_epoch), flush=True)
        raise RuntimeError('Restored validation accuracy mismatch; do not use results')
    # No test-set access until the checkpoint is selected and verified.
    test_metrics = None if args.validation_only else classification_scores(
        y[idx['test']], prediction[idx['test']], class_count)

    save_csv(out / 'train_validation_trace.csv',
             ['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'best_checkpoint'], train_trace)
    if not args.validation_only:
        save_csv(out / 'test_node_predictions.csv',
                 ['node_id', 'true_label', 'predicted_label', 'correct'],
                 [(int(i), int(y[i]), int(prediction[i]), int(y[i] == prediction[i]))
                  for i in idx['test']])
        save_csv(out / 'test_class_metrics.csv',
                 ['class_id', 'test_support', 'predicted_count', 'precision', 'recall', 'f1'],
                 [(r['label'], r['support'], r['predicted_count'], r['precision'],
                   r['recall'], r['f1']) for r in test_metrics['per_class']])
    save_csv(out / 'class_support_by_split.csv',
             ['class_id', 'train_count', 'val_count', 'test_count'],
             [(i, metadata['class_support']['train'][i],
               metadata['class_support']['val'][i], metadata['class_support']['test'][i])
              for i in range(class_count)])

    # The best checkpoint was already written upon validation improvement.
    prefix = best_checkpoint_prefix
    record = {
        'dataset': 'LastFM-Asia', 'graph': str(args.graph), 'split': str(args.split),
        'implementation': 'original H2GCN-2 architecture and preprocessing; evaluation-only training harness',
        'seed': args.seed, 'architecture': architecture,
        'hidden': args.hidden, 'learning_rate': args.learning_rate,
        'dropout': args.dropout, 'weight_decay': args.weight_decay,
        'validation_only': bool(args.validation_only),
        'epochs_max': args.epochs, 'epochs_executed': epoch, 'patience': args.patience,
        'best_epoch': best_epoch, 'selection': 'validation accuracy, validation loss tie-break',
        'best_val_accuracy': best_val_accuracy,
        'best_val_loss': best_val_loss,
        'val_accuracy': val_metrics['accuracy'], 'val_macro_f1': val_metrics['macro_f1'],
        'test_accuracy': None if test_metrics is None else test_metrics['accuracy'],
        'test_macro_f1': None if test_metrics is None else test_metrics['macro_f1'],
        'test_weighted_f1': None if test_metrics is None else test_metrics['weighted_f1'],
        'test_balanced_accuracy': None if test_metrics is None else test_metrics['balanced_accuracy'],
        'majority_class_test_baseline': None if args.validation_only else float(np.mean(y[idx['test']] == np.bincount(y[idx['train']], minlength=class_count).argmax())),
        'checkpoint': str(prefix), 'data': metadata,
    }
    with (out / 'results.json').open('w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    print('BEST: epoch={} val_acc={:.2%}, val_macro_f1={:.2%}'.format(
        best_epoch, val_metrics['accuracy'], val_metrics['macro_f1']), flush=True)
    if test_metrics is None:
        print('VALIDATION-ONLY: no test-set metrics or predictions generated.', flush=True)
    else:
        print('TEST (one final evaluation): accuracy={:.2%}, macro-F1={:.2%}, weighted-F1={:.2%}, balanced_acc={:.2%}'.format(
            test_metrics['accuracy'], test_metrics['macro_f1'],
            test_metrics['weighted_f1'], test_metrics['balanced_accuracy']), flush=True)
    if record['majority_class_test_baseline'] is not None:
        print('Majority-class baseline (train-majority label): {:.2%}'.format(
            record['majority_class_test_baseline']), flush=True)
    print('PASS: results and read-only evaluation artifacts saved to ' + str(out), flush=True)


if __name__ == '__main__':
    main()
