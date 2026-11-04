#!/usr/bin/env python3
"""Recover node-level H2GCN predictions from original saved TF2 checkpoints.

Read-only toward input graphs, splits, and checkpoint directories.

Important: DO NOT import or invoke modules.logger.init_checkpoint_path or
modules.arguments.parse_args: the logger moves existing checkpoint directories.
The model is built directly with the exact source classes and preprocessor.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

NETWORK_SETUP = "M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO"
HOP_RADII = ["1", "2"]
BINS = ("0", "1-2", "3-5", "6+")


def degree_bin(degree):
    if degree == 0:
        return "0"
    if degree <= 2:
        return "1-2"
    if degree <= 5:
        return "3-5"
    return "6+"


def read_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"family", "target_homophily", "replicate", "best_epoch",
                "expected_test_accuracy_pct", "state", "checkpoint_index"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError("Checkpoint manifest lacks required columns")
    keys = [(r["family"], float(r["target_homophily"]), int(r["replicate"])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate family/homophily/replicate in manifest")
    return sorted(rows, key=lambda r: (r["family"], float(r["target_homophily"]), int(r["replicate"])))


def find_graph(repo, family, homophily, replicate):
    directory = repo / "constructed_data" / (family + "_grid")
    pattern = "{}_h{:.2f}_r{}_seed*.npz".format(family, homophily, replicate)
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one graph matching {} in {}, found {}".format(
            pattern, directory, len(matches)))
    return matches[0]


def get_degrees(graph):
    with np.load(str(graph), allow_pickle=False) as data:
        adj = sp.csr_matrix((data["adj_data"], data["adj_indices"], data["adj_indptr"]),
                            shape=tuple(data["adj_shape"]))
        adj = adj.tocsr(copy=True)
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj.sum_duplicates()
        degree = np.asarray((adj != 0).sum(axis=1)).reshape(-1).astype(np.int32)
        labels = data["labels"].astype(np.int32)
    if adj.shape[0] != len(labels):
        raise ValueError("Graph/label mismatch: " + str(graph))
    return degree, labels


def get_masks(split_path, n):
    with np.load(str(split_path), allow_pickle=False) as split:
        masks = {}
        for label, name in (("train", "idx_train"), ("val", "idx_val"), ("test", "idx_test")):
            indexes = np.asarray(split[name], dtype=np.int64)
            if np.any(indexes < 0) or np.any(indexes >= n):
                raise ValueError("Invalid index in " + str(split_path))
            mask = np.zeros(n, dtype=bool)
            mask[indexes] = True
            if mask.sum() != len(indexes):
                raise ValueError("Repeated split index")
            masks[label] = mask
    if (masks["train"] & masks["val"]).any() or (masks["test"] & (masks["train"] | masks["val"])).any():
        raise ValueError("Overlapping split indices")
    masks["wild"] = ~(masks["train"] | masks["val"] | masks["test"])
    return masks


def build_predictor(repo):
    # The 'models' and 'datasets' names are original H2GCN package modules.
    sys.path.insert(0, str(repo / "h2gcn"))
    import tensorflow as tf
    from datasets.npz import NPZData
    from models import parse_network_setup
    from models.H2GCN import H2GCN

    tf.config.experimental_run_functions_eagerly(True)

    def predict(graph, split_path, ckpt_prefix, seed):
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)

        # This mirrors models/H2GCN.py:argparse_callback/preprocessing_data.
        data = NPZData(str(graph), str(split_path))
        layer_setups = parse_network_setup(
            NETWORK_SETUP, data.num_labels,
            _dense_units=64, _dropout_rate=0.5, parse_preprocessing=True)
        data.row_normalize_features()
        data.adj_remove_eye()
        tensors = vars(data.getTensors(getDenseAdj=False, getAdjNormHops=HOP_RADII))
        model = H2GCN(layer_setups, l2_regularize_weight=5e-4)
        model(tensors["adj"], tensors["features"], tensors["adj_hops"], training=False)

        # Match TF checkpoint's original root object name: 'model'.
        # We do not create/restore optimiser slots because inference does not use them.
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(str(ckpt_prefix))
        status.assert_existing_objects_matched()
        status.expect_partial()  # Stored optimizer state intentionally not restored.

        # Original saved test_step uses precisely this inference expression.
        logits = model(tensors["adj"], tensors["features"], tensors["adj_hops"], training=False)
        logits = np.asarray(logits.numpy())
        if logits.ndim != 2:
            raise RuntimeError("Invalid output logit dimensions " + str(logits.shape))
        if not np.all(np.isfinite(logits)):
            raise RuntimeError("Non-finite checkpoint logits")
        pred = logits.argmax(axis=1).astype(np.int32)
        return pred, logits

    return predict


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--manifest", default="results/h2gcn_checkpoint_coverage.csv")
    parser.add_argument("--grid-results", default="results/h2gcn_grid_runs.csv")
    parser.add_argument("--split", default="constructed_data/shared_split_seed42.npz")
    parser.add_argument("--output-dir", default="results/h2gcn_node_predictions")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N graphs (pilot)")
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--tolerance-pp", type=float, default=0.01,
                        help="Maximum tolerated percentage-point difference from rounded saved result")
    args = parser.parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    manifest_path = repo / args.manifest
    split_path = repo / args.split
    output_dir = repo / args.output_dir
    grid_results_path = repo / args.grid_results
    if not manifest_path.is_file() or not split_path.is_file() or not grid_results_path.is_file():
        raise FileNotFoundError("Cannot find the manifest, full H2GCN grid CSV, or shared split")
    with grid_results_path.open(newline="", encoding="utf-8-sig") as handle:
        grid_results = list(csv.DictReader(handle))
    val_by_key = {(r["family"], float(r["homophily"]), int(r["replicate"])): float(r["val_accuracy"])
                  for r in grid_results}
    if len(val_by_key) != 30:
        raise ValueError("Expected 30 unique records in H2GCN grid results")
    rows = read_manifest(manifest_path)
    if len(rows) != 30:
        raise RuntimeError("Expected a 30-row checkpoint manifest; got " + str(len(rows)))
    for row in rows:
        if row["state"] != "FOUND":
            raise RuntimeError("Checkpoint not FOUND for " + str(row))
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.limit:
        rows = rows[:args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    infer = build_predictor(repo)
    overview = []
    degree_rows = []
    node_rows = []
    for i, row in enumerate(rows, 1):
        family = row["family"]
        h = float(row["target_homophily"])
        rep = int(row["replicate"])
        tag = "{}_h{:.2f}_r{}".format(family, h, rep)
        graph = find_graph(repo, family, h, rep)
        index_path = Path(row["checkpoint_index"])
        if not index_path.is_absolute():
            index_path = repo / index_path
        if not index_path.is_file() or index_path.suffix != ".index":
            raise FileNotFoundError("Missing checkpoint index: " + str(index_path))
        prefix = index_path.with_suffix("")
        if not list(prefix.parent.glob(prefix.name + ".data-*")):
            raise FileNotFoundError("Missing checkpoint data shard: " + str(prefix))
        if not prefix.name.endswith("-" + str(row["best_epoch"])):
            raise ValueError("Best epoch / checkpoint mismatch for " + tag)

        degree, truth = get_degrees(graph)
        masks = get_masks(split_path, len(truth))
        print("[{}/{}] {} | best epoch {}".format(i, len(rows), tag, row["best_epoch"]), flush=True)
        predicted, logits = infer(graph, split_path, prefix, args.random_seed)
        if predicted.shape != truth.shape:
            raise ValueError("Prediction count mismatch: " + tag)
        correct = truth == predicted
        test_acc_pct = 100 * float(correct[masks["test"]].mean())
        val_acc_pct = 100 * float(correct[masks["val"]].mean())
        expected = float(row["expected_test_accuracy_pct"])
        expected_val = val_by_key[(family, h, rep)]
        difference = test_acc_pct - expected
        val_difference = val_acc_pct - expected_val
        print("    test {:.4f}% vs {:.2f}% | val {:.4f}% vs {:.2f}%".format(
            test_acc_pct, expected, val_acc_pct, expected_val), flush=True)
        if abs(difference) > args.tolerance_pp or abs(val_difference) > args.tolerance_pp:
            raise RuntimeError("CHECKPOINT ACCURACY MISMATCH in {}: test {:.5f}% vs {:.5f}%, "
                               "validation {:.5f}% vs {:.5f}%; do not use this model".format(
                                   tag, test_acc_pct, expected, val_acc_pct, expected_val))

        np.savez_compressed(str(output_dir / (tag + "_predictions.npz")),
                            predicted=predicted, true_labels=truth, degree=degree,
                            logits=logits, idx_test=np.flatnonzero(masks["test"]),
                            idx_val=np.flatnonzero(masks["val"]),
                            idx_train=np.flatnonzero(masks["train"]))
        overview.append(dict(family=family, target_homophily="{:.2f}".format(h),
                             replicate=rep, graph=str(graph.relative_to(repo)),
                             best_epoch=int(row["best_epoch"]), n_test=int(masks["test"].sum()),
                             expected_test_accuracy_pct=expected,
                             recovered_test_accuracy_pct=round(test_acc_pct, 6),
                             difference_pp=round(difference, 6),
                             expected_val_accuracy_pct=expected_val,
                             recovered_val_accuracy_pct=round(val_acc_pct, 6),
                             val_difference_pp=round(val_difference, 6), state="PASS"))
        for name in BINS:
            within = np.fromiter((degree_bin(int(v)) == name for v in degree), dtype=bool, count=len(degree))
            test_group = within & masks["test"]
            n = int(test_group.sum())
            degree_rows.append(dict(family=family, target_homophily="{:.2f}".format(h),
                                    replicate=rep, degree_bin=name, n_test=n,
                                    n_correct=int(correct[test_group].sum()),
                                    accuracy_pct=round(100*float(correct[test_group].mean()), 6) if n else ""))
        for node in range(len(truth)):
            node_rows.append(dict(family=family, target_homophily="{:.2f}".format(h),
                                  replicate=rep, node_id=node,
                                  split="test" if masks["test"][node] else (
                                      "train" if masks["train"][node] else (
                                      "val" if masks["val"][node] else "wild")),
                                  degree=int(degree[node]), degree_bin=degree_bin(int(degree[node])),
                                  true_label=int(truth[node]), predicted_label=int(predicted[node]),
                                  correct=int(correct[node])))

        # Persist verified progress after every run. Atomic writes avoid partial CSVs.
        write_csv(output_dir / "checkpoint_accuracy_verification.csv", overview,
                  ["family", "target_homophily", "replicate", "graph", "best_epoch", "n_test",
                   "expected_test_accuracy_pct", "recovered_test_accuracy_pct",
                   "difference_pp", "expected_val_accuracy_pct", "recovered_val_accuracy_pct",
                   "val_difference_pp", "state"])
        write_csv(output_dir / "degree_binned_test_accuracy.csv", degree_rows,
                  ["family", "target_homophily", "replicate", "degree_bin", "n_test", "n_correct", "accuracy_pct"])
        write_csv(output_dir / "node_predictions.csv", node_rows,
                  ["family", "target_homophily", "replicate", "node_id", "split", "degree",
                   "degree_bin", "true_label", "predicted_label", "correct"])

    if args.limit == 0:
        assert len(overview) == 30
        assert len(degree_rows) == 120
        assert len(node_rows) == 1490 * 30, "Unexpected graph node count"
        print("PASS: 30 checkpoints restored; all recovered test accuracies matched.")
    else:
        print("PASS: pilot restored {} checkpoint(s) with matching accuracy.".format(len(rows)))
    print("Results directory: " + str(output_dir))


if __name__ == "__main__":
    main()
