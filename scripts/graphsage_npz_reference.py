#!/usr/bin/env python3
"""NPZ adapter and validation-selected trainer for the H2GCN fork of GraphSAGE.

Reuses the original fork's Encoder, MeanAggregator, and SupervisedGraphSage.
Changes from the original experiment runner (explicitly report these):
  * NPZ CSR loader and shared split instead of Planetoid/signac dataset.
  * For isolated nodes, inserts self as the sole aggregation neighbour to
    avoid undefined mean of an empty set; reports the count in each JSON.
  * Explicit seed, validation-accuracy/loss selection, selected-weight restore,
    and no test evaluation until selection is complete.
  * Fixed random neighbour sampling for validation and test, restoring RNG
    state before training continues to ensure comparable evaluation.
  * Modern PyTorch 2.5.1 and Python 3.10 environment.

This is an *adaptation of the authors' reference model*, not an unmodified run.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import random
import sys
import types

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


def read_csr(archive, prefix):
    required = [prefix + key for key in ("_data", "_indices", "_indptr", "_shape")]
    absent = [key for key in required if key not in archive]
    if absent:
        raise ValueError("NPZ missing CSR keys: " + ", ".join(absent))
    return sp.csr_matrix(
        (archive[prefix + "_data"], archive[prefix + "_indices"],
         archive[prefix + "_indptr"]),
        shape=tuple(archive[prefix + "_shape"]), dtype=np.float32)


def load_graph(graph_file, split_file):
    with np.load(graph_file, allow_pickle=False) as archive:
        adjacency = read_csr(archive, "adj")
        features = read_csr(archive, "attr")
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)

    num_nodes = len(labels)
    if adjacency.shape != (num_nodes, num_nodes) or features.shape[0] != num_nodes:
        raise ValueError("Graph/features/labels have inconsistent node counts")
    if adjacency.nnz and not np.all(np.isfinite(adjacency.data)):
        raise ValueError("Adjacency contains nonfinite values")
    if features.nnz and not np.all(np.isfinite(features.data)):
        raise ValueError("Features contain nonfinite values")
    if (adjacency != adjacency.T).nnz:
        raise ValueError("Graph must be undirected with symmetric adjacency")
    if not np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))):
        raise ValueError("Labels must be encoded consecutively from zero")

    with np.load(split_file, allow_pickle=False) as archive:
        indices = {key: np.asarray(archive[key], dtype=np.int64).reshape(-1)
                   for key in ("idx_train", "idx_val", "idx_test")}
    all_ids = np.concatenate(list(indices.values()))
    if (len(all_ids) != num_nodes or len(np.unique(all_ids)) != num_nodes
            or all_ids.min() < 0 or all_ids.max() >= num_nodes):
        raise ValueError("The split must partition all nodes exactly once")
    if min(len(v) for v in indices.values()) == 0:
        raise ValueError("All split partitions must be nonempty")

    adjacency = adjacency.tocsr(copy=True)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1.0
    adjacency.sort_indices()

    # Zero-neighbour nodes otherwise yield a division by zero in the reference
    # MeanAggregator. Self-neighbour fallback is the sole topology adaptation.
    adjacency_lists = defaultdict(set)
    isolated = []
    for node in range(num_nodes):
        neigh = set(int(j) for j in
                    adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]])
        if not neigh:
            neigh = {node}
            isolated.append(node)
        adjacency_lists[node] = neigh

    features = features.toarray().astype(np.float32, copy=False)
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature matrix contains nonfinite values")

    return features, labels, adjacency_lists, indices, isolated


def import_original_modules(repo_root):
    reference_dir = (repo_root / "baselines" / "graphsage-simple" / "graphsage").resolve()
    needed = ("model.py", "encoders.py", "aggregators.py")
    missing = [name for name in needed if not (reference_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing GraphSAGE reference source: " + ", ".join(missing))
    # model.py imports 'dataset' at module load; its original symlink may point
    # outside this repository. Dataset is not used by our standalone NPZ path.
    sys.modules["dataset"] = types.ModuleType("dataset")
    sys.path.insert(0, str(reference_dir))
    original = importlib.import_module("model")
    encoders = importlib.import_module("encoders")
    aggregators = importlib.import_module("aggregators")
    return original, encoders, aggregators


@contextmanager
def fixed_evaluation_rng(seed):
    """Make evaluation-neighbour samples repeatable without changing training RNG.

    GraphSAGE's reference Encoder uses stochastic neighbour sampling even in
    eval() mode. Saving/restoring Python, NumPy and PyTorch RNG state prevents
    validation sampling from changing the model's subsequent training path.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def evaluate(model, nodes, labels):
    model.eval()
    with torch.no_grad():
        logits = model.forward(nodes)
        truth = torch.as_tensor(labels[np.asarray(nodes)], dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, truth).item()
        acc = (logits.argmax(dim=1) == truth).float().mean().item()
    if not (math.isfinite(loss) and math.isfinite(acc)):
        raise FloatingPointError("Nonfinite evaluation metrics")
    return float(loss), float(acc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if (args.hidden < 1 or args.lr <= 0 or args.epochs < 1 or args.batch_size < 1
            or any(x < 1 for x in args.num_samples) or args.log_every < 1):
        parser.error("Hidden/batch/sampling/epochs/log-frequency must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    feats, labels, adj, idx, isolated = load_graph(args.graph, args.split)
    repo_root = Path(args.repo_root).resolve()
    original, encoders, aggregators = import_original_modules(repo_root)

    embedding = nn.Embedding.from_pretrained(torch.from_numpy(feats), freeze=True)
    agg1 = aggregators.MeanAggregator(embedding, cuda=False, gcn=False)
    enc1 = encoders.Encoder(embedding, feats.shape[1], args.hidden, adj, agg1,
                            gcn=False, cuda=False)
    agg2 = aggregators.MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False, gcn=False)
    enc2 = encoders.Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim,
                            args.hidden, adj, agg2, base_model=enc1,
                            gcn=False, cuda=False)
    # Both naming conventions occur in forks of the reference Encoder.
    enc1.num_samples = args.num_samples[0]
    enc2.num_samples = args.num_samples[1]
    enc1.num_sample = args.num_samples[0]
    enc2.num_sample = args.num_samples[1]
    network = original.SupervisedGraphSage(len(np.unique(labels)), enc2)

    optimizer = torch.optim.SGD((p for p in network.parameters() if p.requires_grad),
                                lr=args.lr)
    train_ids = list(map(int, idx["idx_train"]))
    val_ids = list(map(int, idx["idx_val"]))
    test_ids = list(map(int, idx["idx_test"]))
    val_sampling_seed = args.seed + 1000
    test_sampling_seed = args.seed + 2000
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    print(f"Graph: {args.graph}", flush=True)
    print(f"Split: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}", flush=True)
    print(f"GraphSAGE: features={feats.shape[1]} hidden={args.hidden} "
          f"classes={len(np.unique(labels))} samples={args.num_samples} "
          f"SGD-lr={args.lr} isolated={len(isolated)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        network.train()
        # Reference runner uses one SGD minibatch of up to 256 per epoch.
        batch = train_ids[:args.batch_size]
        random.shuffle(train_ids)
        optimizer.zero_grad()
        targets = torch.as_tensor(labels[batch, None], dtype=torch.long)
        loss = network.loss(batch, targets)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Nonfinite train loss at epoch {epoch}")
        loss.backward()
        optimizer.step()

        with fixed_evaluation_rng(val_sampling_seed):
            val_loss, val_acc = evaluate(network, val_ids, labels)
        if (val_acc > best_val_accuracy + 1e-8 or
                (abs(val_acc - best_val_accuracy) <= 1e-8 and val_loss < best_val_loss)):
            best_epoch = epoch
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            best_state = deepcopy(network.state_dict())

        if epoch <= 3 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:4d} | train_loss={loss.item():.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%", flush=True)

    if best_state is None:
        raise RuntimeError("No valid checkpoint obtained")
    network.load_state_dict(best_state)
    # Crucially, the test partition is first evaluated AFTER selecting the
    # best validation checkpoint. No test results influence model selection.
    with fixed_evaluation_rng(test_sampling_seed):
        test_loss, test_acc = evaluate(network, test_ids, labels)
    print(f"BEST: epoch={best_epoch} val_acc={best_val_accuracy*100:.2f}% "
          f"test_acc={test_acc*100:.2f}%", flush=True)

    result = {
        "implementation": "H2GCN fork GraphSAGE classes with independent NPZ trainer; not unmodified reference runner",
        "graph": args.graph,
        "split": args.split,
        "seed": args.seed,
        "hidden": args.hidden,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "optimizer": "SGD",
        "model": "2-layer GraphSAGE-mean",
        "checkpoint_selection": "validation accuracy, validation loss for ties",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "isolated_nodes": len(isolated),
        "isolated_node_policy": "self-neighbour fallback only for zero-degree nodes",
        "feature_preprocessing": "raw CSR features densified; no extra row normalisation",
        "torch_version": torch.__version__,
        "evaluation_sampling": "fixed neighbour sampling seeds, RNG restored after evaluation",
        "val_sampling_seed": val_sampling_seed,
        "test_sampling_seed": test_sampling_seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
