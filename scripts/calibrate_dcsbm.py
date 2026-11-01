import os
import csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

REFERENCE = "npz-datasets/syn-cora/h0.10-r1.npz"
OUT_DIR = "constructed_data/dcsbm_calibration"

TARGET_H = 0.10

# Controls how strongly the reference degree heterogeneity is retained.
BETAS = [0.60, 0.80, 1.00, 1.20]

REPLICATES = 3
BASE_SEED = 5000

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load reference graph
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

labels = d["labels"].astype(np.int32)

n = len(labels)
classes = np.unique(labels)

ref_upper = sp.triu(ref_adj, k=1).tocoo()
ref_edges = ref_upper.nnz

ref_degree = np.asarray(
    ref_adj.sum(axis=1)
).ravel()

ref_mean_degree = ref_degree.mean()
ref_degree_cv = (
    ref_degree.std() / ref_degree.mean()
)

print("=== REFERENCE ===")
print("Nodes:", n)
print("Edges:", ref_edges)
print("Mean degree:", round(ref_mean_degree, 4))
print("Degree CV:", round(ref_degree_cv, 4))
print("Max degree:", int(ref_degree.max()))
print()

# ------------------------------------------------------------
# All possible undirected pairs
# ------------------------------------------------------------
pair_rows, pair_cols = np.triu_indices(n, k=1)

same_mask = (
    labels[pair_rows] == labels[pair_cols]
)

cross_mask = ~same_mask

target_same_edges = TARGET_H * ref_edges
target_cross_edges = (1.0 - TARGET_H) * ref_edges


# ------------------------------------------------------------
# Solve alpha so:
#
# sum(1 - exp(-alpha * theta_i * theta_j))
#
# equals desired expected edge count.
#
# This avoids probabilities greater than 1.
# ------------------------------------------------------------
def solve_alpha(products, target_edges):

    low = 0.0
    high = 1.0

    def expected_edges(alpha):
        return np.sum(
            1.0 - np.exp(-alpha * products)
        )

    while expected_edges(high) < target_edges:
        high *= 2.0

    for _ in range(60):
        mid = (low + high) / 2.0

        if expected_edges(mid) < target_edges:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


summary = []

# ------------------------------------------------------------
# Calibration sweep
# ------------------------------------------------------------
for beta in BETAS:

    for replicate in range(1, REPLICATES + 1):

        seed = (
            BASE_SEED
            + int(beta * 100)
            + replicate
        )

        rng = np.random.RandomState(seed)

        # ----------------------------------------------------
        # Build degree propensities.
        #
        # Reference degrees are shuffled WITHIN each class.
        # This preserves the degree-distribution information
        # but breaks correspondence with original nodes/edges.
        # ----------------------------------------------------
        theta = np.zeros(n, dtype=np.float64)

        for c in classes:

            idx = np.where(labels == c)[0]

            class_degrees = (
                ref_degree[idx]
                .astype(np.float64)
                .copy()
            )

            rng.shuffle(class_degrees)

            # beta controls heterogeneity strength
            propensity = np.power(
                class_degrees,
                beta
            )

            # Normalize each class to mean propensity 1.
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
                -alpha_in * same_products
            )
        )

        probabilities[cross_mask] = (
            1.0
            - np.exp(
                -alpha_out * cross_products
            )
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

        # ----------------------------------------------------
        # Structural statistics
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

        isolated = int(
            np.sum(degree == 0)
        )

        n_components, component_labels = (
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

        lcc_pct = (
            100.0 * largest_component / n
        )

        theta_cv = (
            theta.std() / theta.mean()
        )

        row = {
            "beta": beta,
            "replicate": replicate,
            "seed": seed,
            "actual_h": actual_h,
            "edges": num_edges,
            "mean_degree": mean_degree,
            "degree_std": std_degree,
            "degree_cv": degree_cv,
            "theta_cv": theta_cv,
            "min_degree": degree.min(),
            "max_degree": degree.max(),
            "median_degree": np.median(degree),
            "isolated_nodes": isolated,
            "components": n_components,
            "largest_component_pct": lcc_pct,
            "alpha_in": alpha_in,
            "alpha_out": alpha_out
        }

        summary.append(row)

        print(
            f"beta={beta:.2f}",
            f"r={replicate}",
            f"h={actual_h:.4f}",
            f"edges={num_edges}",
            f"mean_deg={mean_degree:.3f}",
            f"CV={degree_cv:.3f}",
            f"theta_CV={theta_cv:.3f}",
            f"max={int(degree.max())}",
            f"isolated={isolated}",
            f"components={n_components}",
            f"LCC={lcc_pct:.2f}%"
        )

# ------------------------------------------------------------
# Save detailed calibration results
# ------------------------------------------------------------
summary_path = os.path.join(
    OUT_DIR,
    "dcsbm_calibration.csv"
)

with open(summary_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)

# ------------------------------------------------------------
# Grouped results
# ------------------------------------------------------------
print()
print("=" * 78)
print("MEAN ± SD ACROSS 3 REPLICATES")
print("=" * 78)

for beta in BETAS:

    group = [
        x for x in summary
        if x["beta"] == beta
    ]

    def vals(key):
        return np.array(
            [x[key] for x in group],
            dtype=float
        )

    print()
    print(f"beta = {beta:.2f}")

    for key, label in [
        ("actual_h", "Actual h"),
        ("edges", "Edges"),
        ("mean_degree", "Mean degree"),
        ("degree_cv", "Degree CV"),
        ("max_degree", "Max degree"),
        ("isolated_nodes", "Isolated"),
        ("largest_component_pct", "LCC %")
    ]:

        x = vals(key)

        print(
            f"  {label}:",
            f"{x.mean():.4f}",
            "±",
            f"{x.std(ddof=1):.4f}"
        )

print()
print("Reference degree CV:", round(ref_degree_cv, 4))
print("Plain SBM degree CV: approximately 0.50")
print()
print("Calibration summary:", summary_path)
