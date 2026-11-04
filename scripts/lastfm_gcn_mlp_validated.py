#!/usr/bin/env python3
"""Two-layer GCN and matched graph-free MLP for LastFM-Asia, TF 2.2.

The GCN is an independent TF2 adaptation, not original TF1 reference code.
Both models use the same fixed graph, stratified split, input row normalization,
hidden dimension, dropout and regularization. Model selection uses validation
accuracy (val loss breaks ties), and test labels are evaluated only once after
best-weight restoration. This is an external evaluation, not a hyperparameter
sweep or a faithful reproduction of the paper's reported real-world numbers.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def read_csr(z, prefix):
    return sp.csr_matrix((z[prefix + '_data'], z[prefix + '_indices'],
                          z[prefix + '_indptr']), shape=tuple(z[prefix + '_shape']),
                         dtype=np.float32)


def to_sparse(matrix):
    coo = matrix.tocoo()
    ids = np.column_stack([coo.row, coo.col]).astype(np.int64)
    return tf.sparse.reorder(tf.SparseTensor(ids, coo.data.astype(np.float32),
                                             np.asarray(coo.shape, dtype=np.int64)))


def load_graph_and_split(graph_file, split_file, use_graph):
    with np.load(graph_file, allow_pickle=False) as z:
        adj = read_csr(z, 'adj')
        feat = read_csr(z, 'attr')
        y = np.asarray(z['labels'], dtype=np.int64)
    n = len(y)
    if adj.shape != (n,n) or feat.shape[0] != n:
        raise ValueError('Inconsistent graph shapes')
    if (adj != adj.T).nnz:
        raise ValueError('Adjacency must be symmetric')
    if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))):
        raise ValueError('Labels must be encoded from 0 to K-1')
    if not np.all(np.isfinite(feat.data)):
        raise ValueError('Nonfinite feature values')
    with np.load(split_file, allow_pickle=False) as z:
        idx = {x: np.asarray(z['idx_' + x], dtype=np.int64) for x in ('train','val','test')}
    all_ids = np.concatenate(list(idx.values()))
    if (len(all_ids) != n or len(np.unique(all_ids)) != n or
            all_ids.min() < 0 or all_ids.max() >= n):
        raise ValueError('Split does not partition all nodes')
    sums = np.asarray(feat.sum(1)).ravel()
    inv = np.zeros_like(sums, dtype=np.float32)
    positive = sums != 0
    inv[positive] = 1. / sums[positive]
    feat = sp.diags(inv).dot(feat).tocsr().astype(np.float32)
    if use_graph:
        adj = adj.tocsr(copy=True)
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj = adj + sp.eye(n, dtype=np.float32, format='csr')
        d = np.asarray(adj.sum(1)).ravel()
        inv_sqrt = d ** (-0.5)
        norm_adj = sp.diags(inv_sqrt).dot(adj).dot(sp.diags(inv_sqrt)).tocsr()
    else:
        norm_adj = sp.eye(n, dtype=np.float32, format='csr')
    return to_sparse(feat), to_sparse(norm_adj), y, idx, feat.shape[1], len(np.unique(y))


class TwoLayerModel(tf.keras.Model):
    def __init__(self, input_dim, hidden, k, dropout):
        super().__init__()
        self.dropout_rate = dropout
        self.w0 = self.add_weight(name='first_layer', shape=(input_dim, hidden),
                                  initializer='glorot_uniform', trainable=True)
        self.w1 = self.add_weight(name='second_layer', shape=(hidden, k),
                                  initializer='glorot_uniform', trainable=True)

    def call(self, tensors, training=False):
        x, adjacency = tensors
        if training and self.dropout_rate:
            keep = 1. - self.dropout_rate
            mask = tf.random.uniform(tf.shape(x.values)) < keep
            subset = tf.sparse.retain(x, mask)
            x = tf.SparseTensor(subset.indices, subset.values / keep, subset.dense_shape)
        hidden = tf.sparse.sparse_dense_matmul(x, self.w0)
        hidden = tf.sparse.sparse_dense_matmul(adjacency, hidden)
        hidden = tf.nn.relu(hidden)
        if training and self.dropout_rate:
            hidden = tf.nn.dropout(hidden, rate=self.dropout_rate)
        logits = tf.matmul(hidden, self.w1)
        return tf.sparse.sparse_dense_matmul(adjacency, logits)


def evaluate(logits, y, selected):
    ids = np.asarray(selected, dtype=np.int64)
    pred_logits = tf.gather(logits, ids)
    truth = tf.convert_to_tensor(y[ids], dtype=tf.int64)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=truth, logits=pred_logits))
    pred = tf.argmax(pred_logits, axis=1, output_type=tf.int64)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))
    return float(loss.numpy()), float(acc.numpy())


def scores(true, pred, k):
    true = np.asarray(true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    support = np.bincount(true, minlength=k)
    guessed = np.bincount(pred, minlength=k)
    tp = np.bincount(true[true == pred], minlength=k)
    p = np.divide(tp, guessed, out=np.zeros(k, dtype=float), where=guessed != 0)
    r = np.divide(tp, support, out=np.zeros(k, dtype=float), where=support != 0)
    f = np.divide(2 * tp, support + guessed, out=np.zeros(k, dtype=float),
                  where=(support+guessed) != 0)
    return dict(accuracy=float(np.mean(true == pred)), macro_f1=float(np.mean(f)),
                weighted_f1=float(np.dot(support,f) / len(true)),
                balanced_accuracy=float(np.mean(r[support > 0])),
                per_class=[dict(class_id=int(i), test_support=int(support[i]),
                                predicted_count=int(guessed[i]), precision=float(p[i]),
                                recall=float(r[i]), f1=float(f[i])) for i in range(k)])


def write_csv(path, header, rows):
    with path.open('w',newline='',encoding='utf8') as f:
        wr=csv.writer(f);wr.writerow(header);wr.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['gcn','mlp'])
    p.add_argument('--graph', default='existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz')
    p.add_argument('--split', default='existing_data/lastfm_asia/processed/lastfm_asia_split_seed42.npz')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--epochs',type=int,default=2000)
    p.add_argument('--patience',type=int,default=150)
    p.add_argument('--dropout',type=float,default=.5)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--weight-decay',type=float,default=5e-4)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    if not (args.epochs>=1 and args.patience>=1 and args.hidden>0 and 0<=args.dropout<1 and args.lr>0):
        p.error('Invalid training parameters')
    destination=Path(args.output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite nonempty directory: '+str(destination))
    if not Path(args.graph).is_file() or not Path(args.split).is_file():
        p.error('Missing existing LastFM graph or split')
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.keras.backend.clear_session()
    x, adj, y, indices, d, k=load_graph_and_split(args.graph, args.split, args.model=='gcn')
    model=TwoLayerModel(d,args.hidden,k,args.dropout)
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr)
    # Ensure best weights are restored from the same initialized variables.
    _=model((x,adj),training=False)
    best_acc=-1.;best_loss=float('inf');best_epoch=0;best_weights=None
    trace=[]
    print('LastFM %s | features=%d classes=%d | seed=%d | split=%d/%d/%d' % (
        args.model.upper(),d,k,args.seed,*[len(indices[q]) for q in ('train','val','test')]),flush=True)
    print('Checkpoint rule: maximum validation accuracy, validation loss tie-break.',flush=True)
    print('Test evaluated only after checkpoint selection.',flush=True)
    for epoch in range(1,args.epochs+1):
        with tf.GradientTape() as tape:
            logits=model((x,adj),training=True)
            train_logits=tf.gather(logits,indices['train'])
            labels=tf.convert_to_tensor(y[indices['train']],dtype=tf.int64)
            ce=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,logits=train_logits))
            loss=ce+args.weight_decay*tf.nn.l2_loss(model.w0)
        grads=tape.gradient(loss,model.trainable_variables)
        if any(g is None for g in grads):
            raise RuntimeError('Missing model gradient')
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        val_logits=model((x,adj),training=False)
        vl,va=evaluate(val_logits,y,indices['val'])
        if not all(math.isfinite(t) for t in (float(loss.numpy()),vl,va)):
            raise FloatingPointError('Nonfinite metrics at epoch %d' % epoch)
        improved=va>best_acc+1e-7 or (abs(va-best_acc)<=1e-7 and vl<best_loss-1e-9)
        if improved:
            best_acc=va;best_loss=vl;best_epoch=epoch
            best_weights=[q.numpy().copy() for q in model.trainable_variables]
        trace.append((epoch,float(loss.numpy()),vl,va,int(improved)))
        if epoch<=5 or epoch%25==0:
            print('Epoch %4d | val %.2f%% loss %.4f%s' % (
                epoch,100*va,vl,' *' if improved else ''),flush=True)
        if epoch-best_epoch>=args.patience:
            print('Early stopping at',epoch,flush=True);break
    if best_weights is None:raise RuntimeError('No validation-selected checkpoint')
    for var,weight in zip(model.trainable_variables,best_weights):var.assign(weight)
    final=model((x,adj),training=False)
    preds=np.argmax(final.numpy(),axis=1).astype(np.int64)
    val=scores(y[indices['val']],preds[indices['val']],k)
    test=scores(y[indices['test']],preds[indices['test']],k)
    if abs(val['accuracy']-best_acc)>1e-6:
        raise RuntimeError('Restored validation mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    write_csv(destination/'train_validation_trace.csv',
              ['epoch','train_loss','val_loss','val_accuracy','is_best'],trace)
    write_csv(destination/'test_node_predictions.csv',
              ['node_id','true_label','predicted_label','correct'],
              [(int(i),int(y[i]),int(preds[i]),int(y[i]==preds[i])) for i in indices['test']])
    write_csv(destination/'test_class_metrics.csv',
              ['class_id','test_support','predicted_count','precision','recall','f1'],
              [(r['class_id'],r['test_support'],r['predicted_count'],r['precision'],r['recall'],r['f1'])
               for r in test['per_class']])
    record={'dataset':'LastFM-Asia','model':args.model,'seed':args.seed,
            'split':args.split,'graph':args.graph,'hidden':args.hidden,
            'dropout':args.dropout,'lr':args.lr,'weight_decay':args.weight_decay,
            'epochs_max':args.epochs,'epochs_executed':epoch,'best_epoch':best_epoch,
            'patience':args.patience,'selection':'max validation accuracy; loss tie-break',
            'best_val_accuracy':best_acc,'best_val_loss':best_loss,
            'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1'],
            'test_accuracy':test['accuracy'],'test_macro_f1':test['macro_f1'],
            'test_weighted_f1':test['weighted_f1'],'test_balanced_accuracy':test['balanced_accuracy'],
            'majority_class_test_baseline':float(np.mean(y[indices['test']]==np.bincount(y[indices['train']],minlength=k).argmax())),
            'n_train':len(indices['train']),'n_val':len(indices['val']),
            'n_test':len(indices['test']),
            'implementation':'independent TF2 conventional GCN; MLP uses same two-layer model with identity adjacency',
            'preprocessing':'row-normalised features, GCN symmetric normalised self-loop adjacency or identity for MLP'}
    with (destination/'results.json').open('w',encoding='utf8') as f:json.dump(record,f,indent=2)
    print('BEST epoch=%d val_acc=%.2f%% val_macro_F1=%.2f%%' % (
        best_epoch,100*val['accuracy'],100*val['macro_f1']),flush=True)
    print('TEST accuracy=%.2f%% macro-F1=%.2f%% weighted-F1=%.2f%% balanced_acc=%.2f%%' % (
        100*test['accuracy'],100*test['macro_f1'],100*test['weighted_f1'],100*test['balanced_accuracy']),flush=True)
    print('PASS: outputs written to',destination,flush=True)

if __name__=='__main__':main()
