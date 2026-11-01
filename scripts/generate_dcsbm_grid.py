import os
import csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

REFERENCE = "npz-datasets/syn-cora/h0.10-r1.npz"
OUT_DIR = "constructed_data/dcsbm_grid"

TARGET_H = [0.10, 0.30, 0.50, 0.70, 0.90]
REPLICATES = 3
BETA = 1.00
BASE_SEED = 6000

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load reference syn-cora graph
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
classes = np.unique(labels)

ref_upper = sp.triu(ref_adj, k=1).tocoo()
ref_edges = ref_upper.nnz

ref_degree = np.asarray(
    ref_adj.sum(axis=1)
).ravel()

ref_mean_degree = ref_degree.mean()
ref_degree_cv = ref_degree.std() / ref_degree.mean()

print("=== REFERENCE SYN-CORA ===")
print("Nodes:", n)
print("Edges:", ref_edges)
print("Mean degree:", round(ref_mean_degree, 4))
print("Degree CV:", round(ref_degree_cv, 4))
print("Max degree:", int(ref_degree.max()))
print("Selected DC-SBM beta:", BETA)
print()

# ------------------------------------------------------------
# All possible undirected pairs
# ------------------------------------------------------------
pair_rows, pair_cols = np.triu_indices(n, k=1)

same_mask = (
    labels[pair_rows]
    == labels[pair_cols]
)

cross_mask = ~same_mask


def solve_alpha(products, target_edges):
    """
    Find alpha such that the expected number of sampled
    edges is approximately target_edges using

        p_ij = 1 - exp(-alpha * theta_i * theta_j)
    """

    low = 0.0
    high = 1.0

    def expected(alpha):
        return np.sum(
            1.0 - np.exp(-alpha * products)
        )

    while expected(high) < target_edges:
        high *= 2.0

    for _ in range(60):
        mid = (low + high) / 2.0

        if expected(mid) < target_edges:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


summary = []

# ------------------------------------------------------------
# Generate formal DC-SBM grid
# ------------------------------------------------------------
for h_index, target_h in enumerate(TARGET_H):

    target_same_edges = target_h * ref_edges
    target_cross_edges = (
        (1.0 - target_h) * ref_edges
    )

    for replicate in range(1, REPLICATES + 1):

        seed = (
            BASE_SEED
            + h_index * 100
            + replicate
        )

        rng = np.random.RandomState(seed)

        # ----------------------------------------------------
        # Degree-correction parameters
        # ----------------------------------------------------
        theta = np.zeros(
            n,
            dtype=np.float64
        )

        for c in classes:

            idx = np.where(
                labels == c
            )[0]

            class_degrees = (
                ref_degree[idx]
                .astype(np.float64)
                .copy()
            )

            # Break correspondence with original nodes while
            # retaining the reference degree distribution.
            rng.shuffle(class_degrees)

            propensity = np.power(
                class_degrees,
                BETA
            )

            # Mean propensity = 1 within every class.
            propensity /= propensity.mean()

            theta[idx] = propensity

        pair_product = (
            theta[pair_rows]
            * theta[pair_cols]
        )

        same_products = pair_product[same_mask]
        cross_products = pair_product[cross_mask]

        alpha_in = solve_alpha(
            same_products,
            target_same_edges
        )

        alpha_out = solve_alpha(
            cross_products,
            target_cross_edges
        )

        probabilities = np.empty(
            len(pair_rows),
            dtype=np.float64
        )

        probabilities[same_mask] = (
            1.0
            - np.exp(
                -alpha_in
                * same_products
            )
        )

        probabilities[cross_mask] = (
            1.0
            - np.exp(
                -alpha_out
                * cross_products
            )
        )

        selected = (
            rng.random_sample(
                len(pair_rows)
            )
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

        # ----------------------------------------------------
        # Validate structure
        # ----------------------------------------------------
        upper = sp.triu(
            adj,
            k=1
        ).tocoo()

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
            100.0
            * largest_component
            / n
        )

        symmetry_error = (
            adj != adj.T
        ).nnz

        self_loops = int(
            np.count_nonzero(
                adj.diagonal()
            )
        )

        if symmetry_error != 0:
            raise ValueError(
                "Adjacency is not symmetric."
            )

        if self_loops != 0:
            raise ValueError(
                "Graph contains self-loops."
            )

        filename = (
            f"dcsbm_h{target_h:.2f}"
            f"_r{replicate}"
            f"_seed{seed}.npz"
        )

        path = os.path.join(
            OUT_DIR,
            filename
        )

        metadata = (
            "constructed-dcsbm:"
            f"target_h={target_h}:"
            f"actual_h={actual_h:.6f}:"
            f"beta={BETA}:"
            f"replicate={replicate}:"
            f"seed={seed}:"
            f"mean_degree={mean_degree:.6f}:"
            f"degree_cv={degree_cv:.6f}"
        )

        # ----------------------------------------------------
        # Save H2GCN-compatible file
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

        summary.append({
            "family": "DC-SBM",
            "beta": BETA,
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
            "alpha_in": alpha_in,
            "alpha_out": alpha_out
        })

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
# Save full summary CSV
# ------------------------------------------------------------
summary_path = os.path.join(
    OUT_DIR,
    "dcsbm_grid_summary.csv"
)

with open(summary_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)

# ------------------------------------------------------------
# Grouped statistics
# ------------------------------------------------------------
print()
print("=" * 78)
print("MEAN ± SD ACROSS 3 REPLICATES")
print("=" * 78)

for target_h in TARGET_H:

    group = [
        x for x in summary
        if x["target_h"] == target_h
    ]

    def vals(key):
        return np.array(
            [x[key] for x in group],
            dtype=float
        )

    print()
    print(
        f"Target h = {target_h:.2f}"
    )

    for key, label in [
        ("actual_h", "Actual h"),
        ("edges", "Edges"),
        ("mean_degree", "Mean degree"),
        ("degree_cv", "Degree CV"),
        ("max_degree", "Max degree"),
        ("isolated_nodes", "Isolated"),
        (
            "largest_component_pct",
            "LCC %"
        )
    ]:

        x = vals(key)

        print(
            f"  {label}:",
            f"{x.mean():.4f}",
            "±",
            f"{x.std(ddof=1):.4f}"
        )

print()
print(
    "Reference degree CV:",
    round(ref_degree_cv, 4)
)

print(
    "Selected beta:",
    BETA
)

print(
    "Generated:",
    len(summary),
    "DC-SBM graphs"
)

print(
    "Summary:",
    summary_path
)
