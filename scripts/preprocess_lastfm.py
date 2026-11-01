import os
import csv
import json
import numpy as np
import scipy.sparse as sp

RAW_DIR = "existing_data/lastfm_asia/lasftm_asia"
OUT_DIR = "existing_data/lastfm_asia/processed"

EDGE_FILE = os.path.join(RAW_DIR, "lastfm_asia_edges.csv")
TARGET_FILE = os.path.join(RAW_DIR, "lastfm_asia_target.csv")
FEATURE_FILE = os.path.join(RAW_DIR, "lastfm_asia_features.json")

SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load labels
# ------------------------------------------------------------
node_ids = []
targets = []

with open(TARGET_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        node_ids.append(int(row["id"]))
        targets.append(int(row["target"]))

node_ids = np.array(node_ids, dtype=np.int64)
labels = np.array(targets, dtype=np.int32)

n = len(labels)

if not np.array_equal(node_ids, np.arange(n)):
    raise ValueError("Expected node IDs to run consecutively from 0.")

classes, class_counts = np.unique(labels, return_counts=True)

print("=== LABELS ===")
print("Nodes:", n)
print("Classes:", classes)
print("Number of classes:", len(classes))
print("Class counts:", class_counts)
print()

# ------------------------------------------------------------
# 2. Load raw edges and create undirected adjacency
# ------------------------------------------------------------
edge_rows = []
edge_cols = []

with open(EDGE_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        u = int(row["node_1"])
        v = int(row["node_2"])

        if u == v:
            continue

        edge_rows.append(u)
        edge_cols.append(v)

edge_rows = np.array(edge_rows, dtype=np.int64)
edge_cols = np.array(edge_cols, dtype=np.int64)

# Add reverse direction because the network is undirected
rows = np.concatenate([edge_rows, edge_cols])
cols = np.concatenate([edge_cols, edge_rows])

data = np.ones(len(rows), dtype=np.float32)

adj = sp.csr_matrix(
    (data, (rows, cols)),
    shape=(n, n),
    dtype=np.float32
)

# Collapse duplicate edges to binary values
adj.data[:] = 1.0
adj.eliminate_zeros()

# Remove any accidental diagonal entries
adj.setdiag(0)
adj.eliminate_zeros()

# ------------------------------------------------------------
# 3. Load artist features and create sparse binary matrix
# ------------------------------------------------------------
with open(FEATURE_FILE) as f:
    feature_dict = json.load(f)

feature_rows = []
feature_cols = []

max_feature_id = -1

for node_str, feature_ids in feature_dict.items():
    node = int(node_str)

    # Remove duplicates within one user's feature list
    unique_features = set(feature_ids)

    for feature_id in unique_features:
        feature_rows.append(node)
        feature_cols.append(feature_id)

        if feature_id > max_feature_id:
            max_feature_id = feature_id

feature_dim = max_feature_id + 1

feature_values = np.ones(
    len(feature_rows),
    dtype=np.float32
)

features = sp.csr_matrix(
    (
        feature_values,
        (
            np.array(feature_rows),
            np.array(feature_cols)
        )
    ),
    shape=(n, feature_dim),
    dtype=np.float32
)

# ------------------------------------------------------------
# 4. Graph statistics
# ------------------------------------------------------------
upper = sp.triu(adj, k=1).tocoo()

num_edges = upper.nnz
avg_degree = (2.0 * num_edges) / n

same_class_edges = np.sum(
    labels[upper.row] == labels[upper.col]
)

edge_homophily = same_class_edges / num_edges

degrees = np.asarray(adj.sum(axis=1)).ravel()
isolated_nodes = int(np.sum(degrees == 0))

symmetry_error = (adj != adj.T).nnz

print("=== GRAPH ===")
print("Undirected edges:", num_edges)
print("Average degree:", round(avg_degree, 4))
print("Same-class edges:", same_class_edges)
print("Edge homophily:", round(edge_homophily, 4))
print("Isolated nodes:", isolated_nodes)
print("Symmetry errors:", symmetry_error)
print()

print("=== FEATURES ===")
print("Feature matrix shape:", features.shape)
print("Feature dimension:", feature_dim)
print("Non-zero feature values:", features.nnz)
print(
    "Average active features per node:",
    round(features.nnz / n, 2)
)
print()

# ------------------------------------------------------------
# 5. Create deterministic class-stratified 25/25/50 split
# ------------------------------------------------------------
rng = np.random.RandomState(SEED)

idx_train = []
idx_val = []
idx_test = []

for c in classes:
    idx = np.where(labels == c)[0]
    rng.shuffle(idx)

    n_class = len(idx)

    n_train = int(0.25 * n_class)
    n_val = int(0.25 * n_class)

    idx_train.extend(idx[:n_train])
    idx_val.extend(
        idx[n_train:n_train + n_val]
    )
    idx_test.extend(
        idx[n_train + n_val:]
    )

idx_train = np.array(sorted(idx_train), dtype=np.int64)
idx_val = np.array(sorted(idx_val), dtype=np.int64)
idx_test = np.array(sorted(idx_test), dtype=np.int64)

print("=== SPLIT ===")
print("Seed:", SEED)
print("Train:", len(idx_train))
print("Validation:", len(idx_val))
print("Test:", len(idx_test))
print(
    "Total:",
    len(idx_train) + len(idx_val) + len(idx_test)
)

print("\nPer-class split:")
for c in classes:
    print(
        "Class", c,
        "train =", np.sum(labels[idx_train] == c),
        "val =", np.sum(labels[idx_val] == c),
        "test =", np.sum(labels[idx_test] == c)
    )

# ------------------------------------------------------------
# 6. Save H2GCN-compatible NPZ
# ------------------------------------------------------------
graph_path = os.path.join(
    OUT_DIR,
    "lastfm_asia_h2gcn.npz"
)

metadata = (
    "lastfm-asia:"
    f"nodes={n}:"
    f"edges={num_edges}:"
    f"classes={len(classes)}:"
    f"edge_homophily={edge_homophily:.6f}:"
    f"feature_dim={feature_dim}"
)

np.savez(
    graph_path,

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

split_path = os.path.join(
    OUT_DIR,
    "lastfm_asia_split_seed42.npz"
)

np.savez(
    split_path,
    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,
    seed=SEED
)

print()
print("=== SAVED ===")
print("Graph:", graph_path)
print("Split:", split_path)
