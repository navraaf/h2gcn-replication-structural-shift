import os
import csv
import numpy as np
import scipy.sparse as sp

REFERENCE = "npz-datasets/syn-cora/h0.10-r1.npz"
OUT_DIR = "constructed_data"

TARGET_H = [0.10, 0.50, 0.90]
BASE_SEED = 2026

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load official syn-cora reference data
# ------------------------------------------------------------
d = np.load(REFERENCE)

ref_adj = sp.csr_matrix(
    (d["adj_data"], d["adj_indices"], d["adj_indptr"]),
    shape=tuple(d["adj_shape"])
)

features = sp.csr_matrix(
    (d["attr_data"], d["attr_indices"], d["attr_indptr"]),
    shape=tuple(d["attr_shape"])
)

labels = d["labels"].astype(np.int32)

n = len(labels)
classes, counts = np.unique(labels, return_counts=True)
k = len(classes)

# Reference average degree
ref_edges = ref_adj.nnz // 2
target_avg_degree = (2.0 * ref_edges) / n

print("Reference graph")
print("Nodes:", n)
print("Edges:", ref_edges)
print("Average degree:", round(target_avg_degree, 4))
print("Classes:", classes)
print("Class counts:", counts)
print("Features:", features.shape)
print()

# For equal class sizes
class_size = counts[0]

if not np.all(counts == class_size):
    raise ValueError("Expected equal class sizes for this controlled SBM experiment.")

summary = []

# Upper-triangle node pairs, so every possible undirected edge
# is considered exactly once.
rows, cols = np.triu_indices(n, k=1)
same_class_pair = labels[rows] == labels[cols]

for i, target_h in enumerate(TARGET_H):

    seed = BASE_SEED + i
    rng = np.random.RandomState(seed)

    # --------------------------------------------------------
    # Choose p_in and p_out so expected average degree is
    # approximately the same as the official syn-cora graph
    # while controlling expected edge homophily.
    #
    # expected same-class degree = h * d
    # expected cross-class degree = (1-h) * d
    # --------------------------------------------------------
    p_in = (
        target_h * target_avg_degree
        / (class_size - 1)
    )

    p_out = (
        (1.0 - target_h) * target_avg_degree
        / (n - class_size)
    )

    probabilities = np.where(
        same_class_pair,
        p_in,
        p_out
    )

    selected = (
        rng.random_sample(len(rows))
        < probabilities
    )

    edge_rows = rows[selected]
    edge_cols = cols[selected]

    # Build symmetric adjacency
    all_rows = np.concatenate([edge_rows, edge_cols])
    all_cols = np.concatenate([edge_cols, edge_rows])
    values = np.ones(len(all_rows), dtype=np.float32)

    adj = sp.csr_matrix(
        (values, (all_rows, all_cols)),
        shape=(n, n),
        dtype=np.float32
    )

    adj.eliminate_zeros()

    # --------------------------------------------------------
    # Validate graph
    # --------------------------------------------------------
    upper = sp.triu(adj, k=1).tocoo()

    num_edges = upper.nnz
    avg_degree = (2.0 * num_edges) / n

    same_edges = np.sum(
        labels[upper.row] == labels[upper.col]
    )

    actual_h = (
        same_edges / num_edges
        if num_edges > 0 else 0.0
    )

    degree = np.asarray(adj.sum(axis=1)).ravel()
    isolated_nodes = int(np.sum(degree == 0))

    symmetry_error = (adj != adj.T).nnz
    self_loops = int(adj.diagonal().sum())

    if symmetry_error != 0:
        raise ValueError("Generated adjacency is not symmetric.")

    if self_loops != 0:
        raise ValueError("Generated graph contains self-loops.")

    metadata = (
        "constructed-sbm:"
        f"target_h={target_h}:"
        f"actual_h={actual_h:.6f}:"
        f"target_avg_degree={target_avg_degree:.6f}:"
        f"actual_avg_degree={avg_degree:.6f}:"
        f"seed={seed}"
    )

    filename = (
        f"sbm_h{target_h:.2f}_seed{seed}.npz"
    )

    path = os.path.join(
        OUT_DIR,
        filename
    )

    # --------------------------------------------------------
    # Save using the SAME NPZ structure as official syn-cora
    # so our H2GCN adapter can load it later.
    # --------------------------------------------------------
    np.savez(
        path,

        adj_data=adj.data,
        adj_indices=adj.indices,
        adj_indptr=adj.indptr,
        adj_shape=np.array(adj.shape),

        attr_data=features.data,
        attr_indices=features.indices,
        attr_indptr=features.indptr,
        attr_shape=np.array(features.shape),

        labels=labels,
        metadata=metadata
    )

    summary.append({
        "file": filename,
        "seed": seed,
        "target_h": target_h,
        "actual_h": actual_h,
        "nodes": n,
        "edges": num_edges,
        "target_avg_degree": target_avg_degree,
        "actual_avg_degree": avg_degree,
        "isolated_nodes": isolated_nodes,
        "p_in": p_in,
        "p_out": p_out
    })

    print("=" * 60)
    print("File:", filename)
    print("Seed:", seed)
    print("Target homophily:", target_h)
    print("Actual homophily:", round(actual_h, 4))
    print("Edges:", num_edges)
    print("Average degree:", round(avg_degree, 4))
    print("Isolated nodes:", isolated_nodes)
    print("p_in:", round(p_in, 6))
    print("p_out:", round(p_out, 6))

# ------------------------------------------------------------
# Save experiment summary for presentation/report evidence
# ------------------------------------------------------------
summary_path = os.path.join(
    OUT_DIR,
    "sbm_summary.csv"
)

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=summary[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary)

print()
print("Generated", len(summary), "SBM graphs.")
print("Summary saved to:", summary_path)
