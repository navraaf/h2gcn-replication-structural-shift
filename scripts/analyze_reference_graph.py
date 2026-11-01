import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

PATH = "npz-datasets/syn-cora/h0.10-r1.npz"

d = np.load(PATH)

adj = sp.csr_matrix(
    (
        d["adj_data"],
        d["adj_indices"],
        d["adj_indptr"]
    ),
    shape=tuple(d["adj_shape"])
).astype(np.float32)

labels = d["labels"]

# Remove diagonal just in case
adj = adj.copy()
adj.setdiag(0)
adj.eliminate_zeros()

upper = sp.triu(adj, k=1).tocoo()

n = adj.shape[0]
edges = upper.nnz

degree = np.asarray(adj.sum(axis=1)).ravel()

avg_degree = degree.mean()
std_degree = degree.std()
degree_cv = std_degree / avg_degree

isolated = np.sum(degree == 0)

num_components, component_labels = connected_components(
    adj,
    directed=False,
    return_labels=True
)

component_sizes = np.bincount(component_labels)
largest_component = component_sizes.max()

same_class = np.sum(
    labels[upper.row] == labels[upper.col]
)

homophily = same_class / edges

print("=== ORIGINAL SYN-CORA STRUCTURAL PROFILE ===")
print("File:", PATH)
print("Nodes:", n)
print("Edges:", edges)
print("Homophily:", round(homophily, 4))

print()
print("=== DEGREE STATISTICS ===")
print("Mean degree:", round(avg_degree, 4))
print("Degree std:", round(std_degree, 4))
print("Degree CV:", round(degree_cv, 4))
print("Min degree:", int(degree.min()))
print("Max degree:", int(degree.max()))
print("Median degree:", round(float(np.median(degree)), 4))
print("Isolated nodes:", int(isolated))

print()
print("=== CONNECTIVITY ===")
print("Connected components:", num_components)
print("Largest component:", int(largest_component))
print(
    "Largest component %:",
    round(100 * largest_component / n, 2)
)

print()
print("=== CLASS DISTRIBUTION ===")
print("Class counts:", np.bincount(labels))
