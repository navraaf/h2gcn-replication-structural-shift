import hashlib
from collections import defaultdict, Counter

import numpy as np
import scipy.sparse as sp


GRAPH_FILES = {
    "r1": "npz-datasets/syn-cora/h0.10-r1.npz",
    "r2": "npz-datasets/syn-cora/h0.10-r2.npz",
    "r3": "npz-datasets/syn-cora/h0.10-r3.npz",
}

BASE_SPLIT = (
    "npz-datasets/syn-cora/"
    "h0.10-r1-split-seed42.npz"
)

OUTPUT_FILES = {
    "r2": (
        "npz-datasets/syn-cora/"
        "h0.10-r2-split-aligned-seed42.npz"
    ),
    "r3": (
        "npz-datasets/syn-cora/"
        "h0.10-r3-split-aligned-seed42.npz"
    ),
}


def load_graph(path):
    d = np.load(path)

    x = sp.csr_matrix(
        (
            d["attr_data"],
            d["attr_indices"],
            d["attr_indptr"]
        ),
        shape=tuple(d["attr_shape"])
    )

    y = d["labels"].astype(np.int32)

    return x, y


def node_signature(x, y, i):

    row = x.getrow(i)

    h = hashlib.sha256()

    h.update(
        np.asarray(
            row.indices,
            dtype=np.int64
        ).tobytes()
    )

    h.update(
        np.asarray(
            row.data,
            dtype=np.float32
        ).tobytes()
    )

    h.update(
        np.asarray(
            [y[i]],
            dtype=np.int32
        ).tobytes()
    )

    return h.hexdigest()


def get_signatures(x, y):
    return [
        node_signature(x, y, i)
        for i in range(x.shape[0])
    ]


# ------------------------------------------------------------
# Load all graphs
# ------------------------------------------------------------
graphs = {}

for name, path in GRAPH_FILES.items():

    x, y = load_graph(path)
    sigs = get_signatures(x, y)

    graphs[name] = {
        "x": x,
        "y": y,
        "sigs": sigs
    }


# ------------------------------------------------------------
# Load existing r1 split
# ------------------------------------------------------------
split = np.load(BASE_SPLIT)

idx_train_r1 = split["idx_train"].astype(np.int64)
idx_val_r1 = split["idx_val"].astype(np.int64)
idx_test_r1 = split["idx_test"].astype(np.int64)

print("=== BASE R1 SPLIT ===")
print("Train:", len(idx_train_r1))
print("Validation:", len(idx_val_r1))
print("Test:", len(idx_test_r1))
print()


# ------------------------------------------------------------
# Build deterministic r1 -> target mapping
#
# For each feature+label signature:
# sorted r1 indices are paired with sorted target indices.
#
# Duplicate signatures therefore get a deterministic bijection.
# ------------------------------------------------------------
def build_mapping(target_name):

    r1_sigs = graphs["r1"]["sigs"]
    target_sigs = graphs[target_name]["sigs"]

    r1_groups = defaultdict(list)
    target_groups = defaultdict(list)

    for i, sig in enumerate(r1_sigs):
        r1_groups[sig].append(i)

    for i, sig in enumerate(target_sigs):
        target_groups[sig].append(i)

    if Counter(r1_sigs) != Counter(target_sigs):
        raise ValueError(
            f"{target_name} does not contain the "
            "same signature multiset as r1."
        )

    mapping = {}

    for sig in r1_groups:

        source_indices = sorted(r1_groups[sig])
        target_indices = sorted(target_groups[sig])

        if len(source_indices) != len(target_indices):
            raise ValueError(
                "Signature multiplicity mismatch."
            )

        for src, dst in zip(
            source_indices,
            target_indices
        ):
            mapping[src] = dst

    if len(mapping) != len(r1_sigs):
        raise ValueError(
            "Not every r1 node was mapped."
        )

    return mapping


def map_indices(indices, mapping):

    return np.array(
        sorted(mapping[int(i)] for i in indices),
        dtype=np.int64
    )


# ------------------------------------------------------------
# Validate and save r2/r3 aligned splits
# ------------------------------------------------------------
for target_name in ["r2", "r3"]:

    mapping = build_mapping(target_name)

    idx_train = map_indices(
        idx_train_r1,
        mapping
    )

    idx_val = map_indices(
        idx_val_r1,
        mapping
    )

    idx_test = map_indices(
        idx_test_r1,
        mapping
    )

    all_idx = np.concatenate(
        [idx_train, idx_val, idx_test]
    )

    if len(np.unique(all_idx)) != 1490:
        raise ValueError(
            f"{target_name}: split overlap "
            "or missing nodes."
        )

    # --------------------------------------------------------
    # Verify partition-level signature multisets are identical
    # --------------------------------------------------------
    r1_sigs = graphs["r1"]["sigs"]
    target_sigs = graphs[target_name]["sigs"]

    checks = [
        (
            "train",
            idx_train_r1,
            idx_train
        ),
        (
            "validation",
            idx_val_r1,
            idx_val
        ),
        (
            "test",
            idx_test_r1,
            idx_test
        ),
    ]

    print("=" * 70)
    print(
        f"ALIGNMENT: r1 -> {target_name}"
    )

    for partition, src_idx, dst_idx in checks:

        src_counter = Counter(
            r1_sigs[i]
            for i in src_idx
        )

        dst_counter = Counter(
            target_sigs[i]
            for i in dst_idx
        )

        same = (
            src_counter
            == dst_counter
        )

        print(
            f"{partition}:"
        )
        print(
            "  Same feature+label multiset:",
            same
        )

        if not same:
            raise ValueError(
                f"{target_name}: "
                f"{partition} signatures differ."
            )

    # --------------------------------------------------------
    # Per-class counts
    # --------------------------------------------------------
    y = graphs[target_name]["y"]

    print()
    print("Per-class split:")

    for c in np.unique(y):

        print(
            f"Class {c}:",
            "train =",
            np.sum(y[idx_train] == c),
            "val =",
            np.sum(y[idx_val] == c),
            "test =",
            np.sum(y[idx_test] == c)
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out = OUTPUT_FILES[target_name]

    np.savez(
        out,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        seed=42,
        alignment_source="h0.10-r1"
    )

    print()
    print("Saved:", out)

print()
print("All aligned splits created successfully.")
