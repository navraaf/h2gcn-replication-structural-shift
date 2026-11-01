import hashlib
from collections import Counter

import numpy as np
import scipy.sparse as sp


FILES = [
    "npz-datasets/syn-cora/h0.10-r1.npz",
    "npz-datasets/syn-cora/h0.10-r2.npz",
    "npz-datasets/syn-cora/h0.10-r3.npz",
]


def load(path):
    d = np.load(path)

    x = sp.csr_matrix(
        (
            d["attr_data"],
            d["attr_indices"],
            d["attr_indptr"]
        ),
        shape=tuple(d["attr_shape"])
    )

    y = d["labels"]

    return x, y


def node_signature(x, y, i):
    """
    Hash one node's feature row together with its class label.
    """
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


datasets = []

for path in FILES:
    x, y = load(path)

    signatures = [
        node_signature(x, y, i)
        for i in range(x.shape[0])
    ]

    datasets.append(
        (x, y, signatures)
    )

    print("=" * 70)
    print(path)
    print("Unique node signatures:",
          len(set(signatures)))
    print("Total nodes:", len(signatures))

print()
print("=" * 70)
print("MULTISET COMPARISON")
print("=" * 70)

for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):

        sig_i = Counter(
            datasets[i][2]
        )

        sig_j = Counter(
            datasets[j][2]
        )

        same_multiset = (
            sig_i == sig_j
        )

        common = sum(
            (sig_i & sig_j).values()
        )

        print(
            f"r{i+1} vs r{j+1}"
        )
        print(
            "  Same feature+label multiset:",
            same_multiset
        )
        print(
            "  Matching nodes ignoring order:",
            common,
            "/ 1490"
        )

        if common == 1490:
            print(
                "  Interpretation: same node examples,"
                " different ordering."
            )
        else:
            print(
                "  Interpretation: examples differ,"
                " not just ordering."
            )
