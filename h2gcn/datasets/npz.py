from . import *
import numpy as np
import scipy.sparse as sp
from ._dataset import PlanetoidData


class NPZData(PlanetoidData):
    """
    Adapter for the official H2GCN synthetic datasets released in NPZ format.

    The graph, features, and labels come from the official NPZ release.
    Train/validation/test indices are supplied separately so that replacement
    splits can be documented explicitly.
    """

    def __init__(self, dataset_path, split_path):
        self._sparse_data = dict()
        self._dense_data = dict()

        self.dataset_path = dataset_path
        self.split_path = split_path
        self.non_valid_samples = set()

        self.load_npz_data(dataset_path, split_path)

        self._original_data = (
            self._sparse_data.copy(),
            self._dense_data.copy()
        )

    @staticmethod
    def _make_mask(indices, n):
        mask = np.zeros(n, dtype=np.bool_)
        mask[indices] = True
        return mask

    def load_npz_data(self, dataset_path, split_path):
        # ------------------------------------------------------------
        # 1. Load graph structure, node features, and labels
        # ------------------------------------------------------------
        data = np.load(dataset_path)

        adj = sp.csr_matrix(
            (
                data["adj_data"],
                data["adj_indices"],
                data["adj_indptr"]
            ),
            shape=tuple(data["adj_shape"])
        ).astype(np.float32)

        features = sp.csr_matrix(
            (
                data["attr_data"],
                data["attr_indices"],
                data["attr_indptr"]
            ),
            shape=tuple(data["attr_shape"])
        ).astype(np.float32)

        labels_int = data["labels"].astype(np.int32)

        n = adj.shape[0]

        # ------------------------------------------------------------
        # 2. Basic integrity checks
        # ------------------------------------------------------------
        if adj.shape[0] != adj.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        if features.shape[0] != n:
            raise ValueError("Feature matrix and graph have different node counts.")

        if len(labels_int) != n:
            raise ValueError("Label count and graph node count do not match.")

        if (adj != adj.T).nnz != 0:
            raise ValueError("Expected an undirected symmetric graph.")

        # ------------------------------------------------------------
        # 3. Convert integer labels to one-hot labels
        # ------------------------------------------------------------
        classes = np.unique(labels_int)

        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError(
                "This adapter expects class labels numbered consecutively from 0."
            )

        labels = np.eye(
            len(classes),
            dtype=np.float32
        )[labels_int]

        # ------------------------------------------------------------
        # 4. Load our documented replacement split
        # ------------------------------------------------------------
        split = np.load(split_path)

        idx_train = split["idx_train"].astype(np.int64)
        idx_val = split["idx_val"].astype(np.int64)
        idx_test = split["idx_test"].astype(np.int64)

        all_indices = np.concatenate(
            [idx_train, idx_val, idx_test]
        )

        if np.any(all_indices < 0) or np.any(all_indices >= n):
            raise ValueError("Split contains invalid node indices.")

        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError(
                "Train, validation, and test splits overlap."
            )

        # Nodes not assigned to train/val/test become wild nodes.
        train_mask = self._make_mask(idx_train, n)
        val_mask = self._make_mask(idx_val, n)
        test_mask = self._make_mask(idx_test, n)

        wild_mask = np.bitwise_not(
            train_mask | val_mask | test_mask
        )

        # ------------------------------------------------------------
        # 5. Create the label matrices expected by H2GCN
        # ------------------------------------------------------------
        y_train = np.zeros_like(labels)
        y_val = np.zeros_like(labels)
        y_test = np.zeros_like(labels)
        y_wild = np.zeros_like(labels)

        y_train[train_mask] = labels[train_mask]
        y_val[val_mask] = labels[val_mask]
        y_test[test_mask] = labels[test_mask]
        y_wild[wild_mask] = labels[wild_mask]

        # ------------------------------------------------------------
        # 6. Store data using the same interface as PlanetoidData
        # ------------------------------------------------------------
        self._sparse_data["sparse_adj"] = adj
        self._sparse_data["features"] = features

        self._dense_data["y_all"] = labels
        self._dense_data["train_mask"] = train_mask
        self._dense_data["val_mask"] = val_mask
        self._dense_data["test_mask"] = test_mask

        self._dense_data["y_train"] = y_train
        self._dense_data["y_val"] = y_val
        self._dense_data["y_test"] = y_test

        self._dense_data["wild_mask"] = wild_mask
        self._dense_data["y_wild"] = y_wild

        # These are private variables used by PlanetoidData preprocessing.
        self._PlanetoidData__preprocessedAdj = None
        self._PlanetoidData__preprocessedFeature = None

        print("===> NPZ graph loaded")
        print("Nodes:", n)
        print("Edges:", adj.nnz // 2)
        print("Features:", features.shape)
        print("Classes:", len(classes))
        print("Train nodes:", train_mask.sum())
        print("Validation nodes:", val_mask.sum())
        print("Test nodes:", test_mask.sum())
        print("Wild nodes:", wild_mask.sum())


def add_subparser_args(parser):
    subparser = parser.add_argument_group(
        "NPZ Data Arguments (datasets/npz.py)"
    )

    subparser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name used for logging."
    )

    subparser.add_argument(
        "--dataset_path",
        type=str,
        dest="_dataset_path",
        required=True,
        help="Path to official H2GCN NPZ graph."
    )

    subparser.add_argument(
        "--split_path",
        type=str,
        dest="_split_path",
        required=True,
        help="Path to replacement train/validation/test split."
    )

    parser.function_hooks["argparse"].appendleft(argparse_callback)


def argparse_callback(args):
    dataset = NPZData(
        args._dataset_path,
        args._split_path
    )

    args.objects["dataset"] = dataset

    print("===> Dataset loaded:", args.dataset)
