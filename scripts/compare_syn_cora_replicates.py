import numpy as np
import scipy.sparse as sp

FILES = [
    "npz-datasets/syn-cora/h0.10-r1.npz",
    "npz-datasets/syn-cora/h0.10-r2.npz",
    "npz-datasets/syn-cora/h0.10-r3.npz",
]


def load_graph(path):
    d = np.load(path)

    adj = sp.csr_matrix(
        (
            d["adj_data"],
            d["adj_indices"],
            d["adj_indptr"]
        ),
        shape=tuple(d["adj_shape"])
    )

    features = sp.csr_matrix(
        (
            d["attr_data"],
            d["attr_indices"],
            d["attr_indptr"]
        ),
        shape=tuple(d["attr_shape"])
    )

    labels = d["labels"]

    return adj, features, labels, d


graphs = []

for path in FILES:
    adj, features, labels, d = load_graph(path)
    graphs.append((adj, features, labels))

    upper = sp.triu(adj, k=1).tocoo()

    same = np.sum(
        labels[upper.row] == labels[upper.col]
    )

    h = same / upper.nnz

    print("=" * 70)
    print(path)
    print("Nodes:", adj.shape[0])
    print("Edges:", upper.nnz)
    print("Features:", features.shape)
    print("Feature nnz:", features.nnz)
    print("Classes:", np.unique(labels))
    print("Class counts:", np.bincount(labels))
    print("Homophily:", round(h, 6))

    if "metadata" in d.files:
        print("Metadata:", d["metadata"])

print()
print("=" * 70)
print("PAIRWISE COMPARISON")
print("=" * 70)

for i in range(len(graphs)):
    for j in range(i + 1, len(graphs)):

        a1, x1, y1 = graphs[i]
        a2, x2, y2 = graphs[j]

        labels_same = np.array_equal(y1, y2)

        features_same = (
            x1.shape == x2.shape
            and (x1 != x2).nnz == 0
        )

        adjacency_same = (
            a1.shape == a2.shape
            and (a1 != a2).nnz == 0
        )

        adjacency_difference = (
            (a1 != a2).nnz
            if a1.shape == a2.shape
            else None
        )

        print(
            f"r{i+1} vs r{j+1}:"
        )
        print(
            "  Same labels:",
            labels_same
        )
        print(
            "  Same features:",
            features_same
        )
        print(
            "  Same adjacency:",
            adjacency_same
        )
        print(
            "  Adjacency differing entries:",
            adjacency_difference
        )
