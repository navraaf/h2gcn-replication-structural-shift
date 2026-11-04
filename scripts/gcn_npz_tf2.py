#!/usr/bin/env python3
"""Standalone two-layer GCN baseline for H2GCN structural-shift NPZ graphs.

Compatible with TensorFlow 2.2 / Python 3.8. This is an independent
TF2-compatible implementation of the conventional Kipf-Welling GCN,
NOT a direct execution of the original authors' TensorFlow 1.x script.

Validation accuracy (tie-break: validation cross entropy) selects the
checkpoint; test labels are evaluated only after checkpoint selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def read_csr(z, prefix):
    return sp.csr_matrix(
        (z[prefix + "_data"], z[prefix + "_indices"], z[prefix + "_indptr"]),
        shape=tuple(z[prefix + "_shape"]), dtype=np.float32
    )


def tf_sparse(matrix):
    matrix = matrix.tocoo()
    indices = np.column_stack([matrix.row, matrix.col]).astype(np.int64)
    tensor = tf.SparseTensor(indices=indices,
                             values=matrix.data.astype(np.float32),
                             dense_shape=np.asarray(matrix.shape, dtype=np.int64))
    return tf.sparse.reorder(tensor)


def load_inputs(graph_path, split_path):
    with np.load(graph_path) as z:
        adj = read_csr(z, "adj")
        features = read_csr(z, "attr")
        y = np.asarray(z["labels"], dtype=np.int64)
    n = len(y)
    if adj.shape != (n, n) or features.shape[0] != n:
        raise ValueError("Inconsistent node counts")
    if (adj != adj.T).nnz:
        raise ValueError("Expected symmetric adjacency")
    if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))):
        raise ValueError("Expected labels 0 through K-1")

    with np.load(split_path) as z:
        idx = {p: np.asarray(z[p], dtype=np.int64)
               for p in ("idx_train", "idx_val", "idx_test")}
    merged = np.concatenate(list(idx.values()))
    if (len(merged) != n or len(np.unique(merged)) != n or
            merged.min() < 0 or merged.max() >= n):
        raise ValueError("Split must partition all nodes exactly once")

    # Row-normalize sparse node features (as in the classic GCN).
    row_sums = np.asarray(features.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums != 0
    inverse[nonzero] = 1.0 / row_sums[nonzero]
    features = sp.diags(inverse).dot(features).tocsr().astype(np.float32)

    # Kipf-Welling renormalization: D^-1/2 (A + I) D^-1/2.
    adj = adj.copy().tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = (adj + sp.eye(n, dtype=np.float32, format="csr")).tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = degree ** (-0.5)
    d = sp.diags(inv_sqrt)
    adj = d.dot(adj).dot(d).tocsr().astype(np.float32)

    return tf_sparse(features), tf_sparse(adj), y, idx, features.shape[1], len(np.unique(y))


class TwoLayerGCN(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super(TwoLayerGCN, self).__init__()
        self.dropout_rate = dropout
        self.w0 = self.add_weight(name="gcn_layer1_weights",
                                  shape=(input_dim, hidden_dim),
                                  initializer="glorot_uniform", trainable=True)
        self.w1 = self.add_weight(name="gcn_layer2_weights",
                                  shape=(hidden_dim, num_classes),
                                  initializer="glorot_uniform", trainable=True)

    def sparse_dropout(self, x):
        if self.dropout_rate <= 0.0:
            return x
        keep = 1.0 - self.dropout_rate
        mask = tf.random.uniform(tf.shape(x.values)) < keep
        retained = tf.sparse.retain(x, mask)
        return tf.SparseTensor(retained.indices, retained.values / keep,
                               retained.dense_shape)

    def call(self, inputs, training=False):
        x, adj = inputs
        if training:
            x = self.sparse_dropout(x)
        x = tf.sparse.sparse_dense_matmul(x, self.w0)
        x = tf.sparse.sparse_dense_matmul(adj, x)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = tf.matmul(x, self.w1)
        return tf.sparse.sparse_dense_matmul(adj, x)


def eval_partition(logits, y, indices):
    scores = tf.gather(logits, indices)
    truth = tf.convert_to_tensor(y[indices], dtype=tf.int64)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=truth, logits=scores))
    pred = tf.argmax(scores, axis=1, output_type=tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))
    return float(loss.numpy()), float(accuracy.numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args()
    if args.patience < 1 or args.epochs < 1 or not 0 <= args.dropout < 1:
        parser.error("Use epochs >= 1, patience >= 1 and 0 <= dropout < 1")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    x, adj, y, idx, input_dim, classes = load_inputs(args.graph, args.split)
    print("Graph:", args.graph, flush=True)
    print("Train/val/test:", *(len(idx[k]) for k in ("idx_train", "idx_val", "idx_test")), flush=True)
    print("GCN: %d -> %d -> %d, seed=%d" % (input_dim, args.hidden, classes, args.seed), flush=True)

    model = TwoLayerGCN(input_dim, args.hidden, classes, args.dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    best_weights = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        with tf.GradientTape() as tape:
            logits = model((x, adj), training=True)
            # Use tensors for gradients, not the numeric evaluation helper.
            tr_logits = tf.gather(logits, idx["idx_train"])
            tr_labels = tf.convert_to_tensor(y[idx["idx_train"]], dtype=tf.int64)
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tr_labels, logits=tr_logits))
            loss = cross_entropy + args.weight_decay * tf.nn.l2_loss(model.w0)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        logits = model((x, adj), training=False)
        val_loss, val_acc = eval_partition(logits, y, idx["idx_val"])
        if val_acc > best_val_acc + 1e-7 or (
                abs(val_acc - best_val_acc) <= 1e-7 and val_loss < best_val_loss):
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_weights = [v.numpy().copy() for v in model.trainable_variables]

        if epoch <= 5 or epoch % 25 == 0:
            print("Epoch %4d | train loss %.4f | val loss %.4f | val acc %.2f%%" %
                  (epoch, float(loss.numpy()), val_loss, val_acc * 100), flush=True)
        if epoch - best_epoch >= args.patience:
            print("Early stopping at epoch", epoch, flush=True)
            break

    for variable, weight in zip(model.trainable_variables, best_weights):
        variable.assign(weight)
    final_logits = model((x, adj), training=False)
    test_loss, test_acc = eval_partition(final_logits, y, idx["idx_test"])
    print("BEST: epoch=%d val_acc=%.2f%% test_acc=%.2f%%" %
          (best_epoch, best_val_acc * 100, test_acc * 100), flush=True)

    result = {
        "implementation": "independent TF2-compatible two-layer GCN baseline; not unmodified TF1 reference",
        "graph": args.graph, "split": args.split, "seed": args.seed,
        "hidden": args.hidden, "dropout": args.dropout, "lr": args.lr,
        "weight_decay": args.weight_decay, "max_epochs": args.epochs,
        "patience": args.patience, "best_epoch": best_epoch,
        "epochs_executed": epoch, "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss, "test_accuracy": test_acc,
        "test_loss": test_loss, "n_train": len(idx["idx_train"]),
        "n_val": len(idx["idx_val"]), "n_test": len(idx["idx_test"]),
        "checkpoint_selection": "validation accuracy, validation loss for ties",
        "preprocessing": "row-normalized features, self-loop symmetric GCN adjacency"
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(result, f, indent=2)
    print("Saved:", str(output), flush=True)


if __name__ == "__main__":
    main()
