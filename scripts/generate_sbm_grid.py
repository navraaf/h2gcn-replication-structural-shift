import os
import csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

REFERENCE = "npz-datasets/syn-cora/h0.10-r1.npz"
OUT_DIR = "constructed_data/sbm_grid"

TARGET_H = [0.10, 0.30, 0.50, 0.70, 0.90]
REPLICATES = 3
BASE_SEED = 2026

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load official syn-cora reference
# ------------------------------------------------------------
d = np.load(REFERENCE)

ref_adj = sp.csr_matrix(
    (
        d["adj_data"],
        d["adj_indices"],
        d["adj_indptr"]
    ),
    shape=tuple(d["adj_shape"])
).astype(np.float32)

features = sp.csr_matrix(
    (
        d["attr_data"],
        d["attr_indices"],
        d["attr_indptr"]
    ),
    shape=tuple(d["attr_shape"])
).astype(np.float32)

labels = d["labels"].astype(np.int32)

n = len(labels)
classes, counts = np.unique(labels, return_counts=True)
k = len(classes)

if not np.all(counts == counts[0]):
    raise ValueError("Expected balanced classes.")

class_size = counts[0]

ref_upper = sp.triu(ref_adj, k=1).tocoo()
ref_edges = ref_upper.nnz
target_avg_degree = 2.0 * ref_edges / n

ref_degree = np.asarray(ref_adj.sum(axis=1)).ravel()

print("=== REFERENCE SYN-CORA ===")
print("Nodes:", n)
print("Edges:", ref_edges)
print("Target mean degree:", round(target_avg_degree, 4))
print("Reference degree CV:",
      round(ref_degree.std() / ref_degree.mean(), 4))
print("Reference max degree:", int(ref_degree.max()))
print()

# ------------------------------------------------------------
# Precompute all possible undirected node pairs
# ------------------------------------------------------------
pair_rows, pair_cols = np.triu_indices(n, k=1)

same_class_pair = (
    labels[pair_rows] == labels[pair_cols]
)

summary = []

# ------------------------------------------------------------
# Generate grid
# ------------------------------------------------------------
for h_index, target_h in enumerate(TARGET_H):

    for replicate in range(1, REPLICATES + 1):

        seed = (
            BASE_SEED
            + h_index * 100
            + replicate
        )

        rng = np.random.RandomState(seed)

        # Expected within-class and cross-class degrees:
        #
        # within = h*d
        # cross  = (1-h)*d
        #
        # Convert these to pairwise SBM probabilities.
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
            rng.random_sample(len(pair_rows))
            < probabilities
        )

        edge_rows = pair_rows[selected]
        edge_cols = pair_cols[selected]

        rows = np.concatenate(
            [edge_rows, edge_cols]
        )

        cols = np.concatenate(
            [edge_cols, edge_rows]
        )

        values = np.ones(
            len(rows),
            dtype=np.float32
        )

        adj = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(n, n),
            dtype=np.float32
        )

        adj.eliminate_zeros()

        # ----------------------------------------------------
        # Structural validation
        # ----------------------------------------------------
        upper = sp.triu(adj, k=1).tocoo()

        num_edges = upper.nnz

        degree = np.asarray(
            adj.sum(axis=1)
        ).ravel()

        mean_degree = degree.mean()
        std_degree = degree.std()

        degree_cv = (
            std_degree / mean_degree
            if mean_degree > 0
            else 0.0
        )

        same_edges = np.sum(
            labels[upper.row]
            == labels[upper.col]
        )

        actual_h = (
            same_edges / num_edges
            if num_edges > 0
            else 0.0
        )

        isolated_nodes = int(
            np.sum(degree == 0)
        )

        num_components, component_labels = (
            connected_components(
                adj,
                directed=False,
                return_labels=True
            )
        )

        component_sizes = np.bincount(
            component_labels
        )

        largest_component = int(
            component_sizes.max()
        )

        largest_component_pct = (
            100.0 * largest_component / n
        )

        symmetry_error = (
            adj != adj.T
        ).nnz

        self_loops = int(
            np.count_nonzero(adj.diagonal())
        )

        if symmetry_error != 0:
            raise ValueError(
                "Generated adjacency is not symmetric."
            )

        if self_loops != 0:
            raise ValueError(
                "Generated graph contains self-loops."
            )

        filename = (
            f"sbm_h{target_h:.2f}"
            f"_r{replicate}"
            f"_seed{seed}.npz"
        )

        path = os.path.join(
            OUT_DIR,
            filename
        )

        metadata = (
            "constructed-sbm:"
            f"target_h={target_h}:"
            f"actual_h={actual_h:.6f}:"
            f"replicate={replicate}:"
            f"seed={seed}:"
            f"mean_degree={mean_degree:.6f}:"
            f"degree_cv={degree_cv:.6f}"
        )

        # ----------------------------------------------------
        # Save in H2GCN-compatible NPZ structure
        # ----------------------------------------------------
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

        row = {
            "family": "SBM",
            "file": filename,
            "target_h": target_h,
            "replicate": replicate,
            "seed": seed,
            "actual_h": actual_h,
            "nodes": n,
            "edges": num_edges,
            "mean_degree": mean_degree,
            "degree_std": std_degree,
            "degree_cv": degree_cv,
            "min_degree": degree.min(),
            "max_degree": degree.max(),
            "median_degree": np.median(degree),
            "isolated_nodes": isolated_nodes,
            "components": num_components,
            "largest_component": largest_component,
            "largest_component_pct": largest_component_pct,
            "p_in": p_in,
            "p_out": p_out
        }

        summary.append(row)

        print(
            f"h={target_h:.2f}",
            f"r={replicate}",
            f"seed={seed}",
            f"actual_h={actual_h:.4f}",
            f"edges={num_edges}",
            f"mean_deg={mean_degree:.3f}",
            f"CV={degree_cv:.3f}",
            f"max_deg={int(degree.max())}",
            f"isolated={isolated_nodes}",
            f"components={num_components}",
            f"LCC={largest_component_pct:.2f}%"
        )

# ------------------------------------------------------------
# Save complete graph-level summary
# ------------------------------------------------------------
summary_path = os.path.join(
    OUT_DIR,
    "sbm_grid_summary.csv"
)

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)

# ------------------------------------------------------------
# Print grouped summary across three replicates
# ------------------------------------------------------------
print()
print("=" * 78)
print("MEAN ± SD ACROSS 3 REPLICATES")
print("=" * 78)

for target_h in TARGET_H:

    rows_h = [
        x for x in summary
        if x["target_h"] == target_h
    ]

    actual_h_values = np.array(
        [x["actual_h"] for x in rows_h]
    )

    edge_values = np.array(
        [x["edges"] for x in rows_h]
    )

    degree_values = np.array(
        [x["mean_degree"] for x in rows_h]
    )

    cv_values = np.array(
        [x["degree_cv"] for x in rows_h]
    )

    max_values = np.array(
        [x["max_degree"] for x in rows_h]
    )

    isolated_values = np.array(
        [x["isolated_nodes"] for x in rows_h]
    )

    lcc_values = np.array(
        [x["largest_component_pct"] for x in rows_h]
    )

    print()
    print(f"Target h = {target_h:.2f}")

    print(
        "  Actual h:",
        f"{actual_h_values.mean():.4f}",
        "±",
        f"{actual_h_values.std(ddof=1):.4f}"
    )

    print(
        "  Edges:",
        f"{edge_values.mean():.1f}",
        "±",
        f"{edge_values.std(ddof=1):.1f}"
    )

    print(
        "  Mean degree:",
        f"{degree_values.mean():.4f}",
        "±",
        f"{degree_values.std(ddof=1):.4f}"
    )

    print(
        "  Degree CV:",
        f"{cv_values.mean():.4f}",
        "±",
        f"{cv_values.std(ddof=1):.4f}"
    )

    print(
        "  Max degree:",
        f"{max_values.mean():.2f}",
        "±",
        f"{max_values.std(ddof=1):.2f}"
    )

    print(
        "  Isolated:",
        f"{isolated_values.mean():.2f}",
        "±",
        f"{isolated_values.std(ddof=1):.2f}"
    )

    print(
        "  LCC %:",
        f"{lcc_values.mean():.2f}",
        "±",
        f"{lcc_values.std(ddof=1):.2f}"
    )

print()
print("Generated:", len(summary), "SBM graphs")
print("Summary:", summary_path)
