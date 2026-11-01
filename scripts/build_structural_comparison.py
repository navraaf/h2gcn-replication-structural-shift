import os
import csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

REFERENCE = "npz-datasets/syn-cora/h0.10-r1.npz"
SBM_CSV = "constructed_data/sbm_grid/sbm_grid_summary.csv"
DCSBM_CSV = "constructed_data/dcsbm_grid/dcsbm_grid_summary.csv"
OUT = "results/structural_comparison_h010.csv"

os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------
# Original syn-cora h=0.10 r1
# ------------------------------------------------------------
d = np.load(REFERENCE)

adj = sp.csr_matrix(
    (
        d["adj_data"],
        d["adj_indices"],
        d["adj_indptr"]
    ),
    shape=tuple(d["adj_shape"])
)

labels = d["labels"]

upper = sp.triu(adj, k=1).tocoo()
degree = np.asarray(adj.sum(axis=1)).ravel()

same_edges = np.sum(
    labels[upper.row] == labels[upper.col]
)

h = same_edges / upper.nnz

n_components, component_labels = connected_components(
    adj,
    directed=False,
    return_labels=True
)

component_sizes = np.bincount(component_labels)

original = {
    "family": "Original syn-cora",
    "homophily_mean": h,
    "homophily_sd": np.nan,
    "edges_mean": upper.nnz,
    "edges_sd": np.nan,
    "mean_degree_mean": degree.mean(),
    "mean_degree_sd": np.nan,
    "degree_cv_mean": degree.std() / degree.mean(),
    "degree_cv_sd": np.nan,
    "max_degree_mean": degree.max(),
    "max_degree_sd": np.nan,
    "isolated_mean": np.sum(degree == 0),
    "isolated_sd": np.nan,
    "lcc_pct_mean": (
        100 * component_sizes.max() / len(labels)
    ),
    "lcc_pct_sd": np.nan
}


# ------------------------------------------------------------
# Load constructed summaries
# ------------------------------------------------------------
def load_h010(path):

    rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if abs(float(row["target_h"]) - 0.10) < 1e-9:
                rows.append(row)

    if len(rows) != 3:
        raise ValueError(
            f"Expected 3 h=0.10 replicates in {path}, "
            f"found {len(rows)}"
        )

    return rows


def aggregate(name, rows):

    mapping = {
        "actual_h": "homophily",
        "edges": "edges",
        "mean_degree": "mean_degree",
        "degree_cv": "degree_cv",
        "max_degree": "max_degree",
        "isolated_nodes": "isolated",
        "largest_component_pct": "lcc_pct"
    }

    result = {"family": name}

    for source, target in mapping.items():

        x = np.array(
            [float(r[source]) for r in rows],
            dtype=float
        )

        result[f"{target}_mean"] = x.mean()
        result[f"{target}_sd"] = x.std(ddof=1)

    return result


sbm = aggregate(
    "SBM",
    load_h010(SBM_CSV)
)

dcsbm = aggregate(
    "DC-SBM",
    load_h010(DCSBM_CSV)
)

rows = [original, sbm, dcsbm]


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
fields = [
    "family",
    "homophily_mean",
    "homophily_sd",
    "edges_mean",
    "edges_sd",
    "mean_degree_mean",
    "mean_degree_sd",
    "degree_cv_mean",
    "degree_cv_sd",
    "max_degree_mean",
    "max_degree_sd",
    "isolated_mean",
    "isolated_sd",
    "lcc_pct_mean",
    "lcc_pct_sd"
]

with open(OUT, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(rows)


def fmt(mean, sd, decimals=4):

    if np.isnan(sd):
        return f"{mean:.{decimals}f}"

    return (
        f"{mean:.{decimals}f}"
        f" ± {sd:.{decimals}f}"
    )


print()
print("=" * 100)
print("STRUCTURAL COMPARISON AT h ≈ 0.10")
print("=" * 100)

for r in rows:

    print()
    print(r["family"])

    print(
        "  Homophily:",
        fmt(
            r["homophily_mean"],
            r["homophily_sd"]
        )
    )

    print(
        "  Edges:",
        fmt(
            r["edges_mean"],
            r["edges_sd"],
            1
        )
    )

    print(
        "  Mean degree:",
        fmt(
            r["mean_degree_mean"],
            r["mean_degree_sd"]
        )
    )

    print(
        "  Degree CV:",
        fmt(
            r["degree_cv_mean"],
            r["degree_cv_sd"]
        )
    )

    print(
        "  Max degree:",
        fmt(
            r["max_degree_mean"],
            r["max_degree_sd"],
            2
        )
    )

    print(
        "  Isolated nodes:",
        fmt(
            r["isolated_mean"],
            r["isolated_sd"],
            2
        )
    )

    print(
        "  Largest component %:",
        fmt(
            r["lcc_pct_mean"],
            r["lcc_pct_sd"],
            2
        )
    )

print()
print("Saved:", OUT)
