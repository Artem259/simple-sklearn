"""Agglomerative Clustering.

This module provides the `AgglomerativeClustering` class.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data

from . import _tools


class AgglomerativeClustering(ClusterMixin, BaseEstimator):  # type: ignore
    """Perform hierarchical agglomerative clustering.

    Recursively merges the pair of clusters that minimally increases a given
    linkage distance.

    Args:
        n_clusters: The number of clusters to find.
        linkage: Which linkage criterion to use.
            The linkage criterion determines which distance to use between sets of observation.
            Must be one of "ward", "complete", "average", or "single".

    Attributes:
        labels_: Cluster labels for each point.
        children_:
            An array of shape `(n_samples - 1, 2)` representing the children of each non-leaf cluster.
        distances_: Distances between clusters in the corresponding places in `children_`.
    """

    labels_: NDArray[np.int_]
    children_: NDArray[np.int_]
    distances_: NDArray[np.float64]

    def __init__(self, n_clusters: int = 2, linkage: str = "ward"):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X: Any, y: Any = None) -> "AgglomerativeClustering":
        """Fit the agglomerative clustering model.

        Args:
            X: Training instances to cluster. Can be an array-like of shape `(n_samples, n_features)`.
            y: Ignored. Present here for API consistency by convention.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If `n_clusters` or `linkage` parameters are invalid.
        """
        X = validate_data(self, X)
        X = np.array(X)
        self.__validate_params()

        num_samples = X.shape[0]
        labels = np.arange(num_samples)
        linkage_matrix = self._init_linkage_matrix(X)
        linkage_indices = [[i, [i]] for i in range(num_samples)]
        children = []
        distances = []

        for i in range(num_samples - 1):
            child, distance, linkage_matrix = self._merge_clusters_iter(X, labels, linkage_matrix, linkage_indices)
            children.append(child)
            distances.append(distance)

            if i == num_samples - self.n_clusters - 1:
                _, self.labels_ = np.unique(labels, return_inverse=True)

        self.children_ = np.array(children)
        self.distances_ = np.array(distances)
        return self

    def _init_linkage_matrix(self, X: NDArray[Any]) -> NDArray[Any]:
        """Initialize the distance matrix.

        Args:
            X: The original input data.

        Returns:
            The initial distance matrix with diagonals set to infinity.
        """
        linkage_matrix = _tools.calc_distance_matrix(X, X)
        np.fill_diagonal(linkage_matrix, np.inf)
        return linkage_matrix

    def _merge_clusters_iter(
        self, X: NDArray[Any], labels: NDArray[Any], linkage_matrix: NDArray[Any], linkage_indices: list[list[Any]]
    ) -> tuple[list[int], float, NDArray[Any]]:
        """Perform a single iteration of merging the two closest clusters.

        Args:
            X: The original input data.
            labels: The current cluster assignments for each sample.
            linkage_matrix: The current matrix of distances between clusters.
            linkage_indices: A mapping of matrix indices to original sample indices.

        Returns:
            A tuple containing:
                - child: A list of the two indices that were merged.
                - distance: The computed distance between the merged clusters.
                - linkage_matrix: The updated distance matrix.
        """
        unraveled = np.unravel_index(np.argmin(linkage_matrix), linkage_matrix.shape)
        lm_min_index = tuple(int(idx) for idx in sorted(unraveled))
        index1, indices1 = linkage_indices[lm_min_index[0]]
        index2, indices2 = linkage_indices[lm_min_index[1]]
        child = [index1, index2]
        distance = linkage_matrix[lm_min_index]

        size = linkage_matrix.shape[0]
        new_index = int(max(labels)) + 1
        new_indices = indices1 + indices2
        new_lm_array = np.array(
            [
                self._calc_clusters_distance(i, lm_min_index[0], lm_min_index[1], linkage_matrix, linkage_indices, X)
                if i != size
                else np.inf
                for i in range(size + 1)
                if i not in lm_min_index
            ]
        )
        new_linkage_index = [new_index, new_indices]

        labels[np.isin(labels, [index1, index2])] = new_index
        linkage_matrix = np.delete(linkage_matrix, lm_min_index, axis=0)
        linkage_matrix = np.delete(linkage_matrix, lm_min_index, axis=1)
        linkage_matrix = np.pad(linkage_matrix, ((0, 1), (0, 1)))
        linkage_matrix[-1, :] = new_lm_array
        linkage_matrix[:, -1] = new_lm_array
        del linkage_indices[lm_min_index[1]]
        del linkage_indices[lm_min_index[0]]
        linkage_indices.append(new_linkage_index)

        return child, distance, linkage_matrix

    def _calc_clusters_distance(
        self,
        i: int,
        i_merged_1: int,
        i_merged_2: int,
        linkage_matrix: NDArray[Any],
        linkage_indices: list[list[Any]],
        X: NDArray[Any],
    ) -> float:
        """Calculate the distance between a target cluster and two newly merged clusters.

        Dispatches the calculation to the specific function based on `self.linkage`.

        Args:
            i: Index of the target cluster.
            i_merged_1: Index of the first merged cluster.
            i_merged_2: Index of the second merged cluster.
            linkage_matrix: The current distance matrix.
            linkage_indices: The current indices mapping.
            X: The original input data.

        Returns:
            The calculated distance according to the specified linkage method.
        """
        linkage_methods: dict[str, Callable[..., float]] = {
            "single": _single_clusters_distance,
            "complete": _complete_clusters_distance,
            "average": _average_clusters_distance,
            "ward": _ward_clusters_distance,
        }
        return linkage_methods[self.linkage](
            i=i,
            i_merged_1=i_merged_1,
            i_merged_2=i_merged_2,
            linkage_matrix=linkage_matrix,
            linkage_indices=linkage_indices,
            X=X,
        )

    def __validate_params(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If `n_clusters` is not a positive integer or
                if `linkage` is not one of the supported string literals.
        """
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError(
                f"The 'n_clusters' parameter must be an int in the range [1, inf). Got '{self.n_clusters}' instead."
            )
        if self.linkage not in ("single", "complete", "average", "ward"):
            raise ValueError(
                f"The 'linkage' parameter must be a str among "
                f"['single', 'complete', 'average', 'ward']. Got '{self.linkage}' instead."
            )


def _single_clusters_distance(
    i: int, i_merged_1: int, i_merged_2: int, linkage_matrix: NDArray[Any], **kwargs: Any
) -> float:
    distance = min(linkage_matrix[i][i_merged_1], linkage_matrix[i][i_merged_2])
    return float(distance)


def _complete_clusters_distance(
    i: int, i_merged_1: int, i_merged_2: int, linkage_matrix: NDArray[Any], **kwargs: Any
) -> float:
    distance = max(linkage_matrix[i][i_merged_1], linkage_matrix[i][i_merged_2])
    return float(distance)


def _average_clusters_distance(
    i: int,
    i_merged_1: int,
    i_merged_2: int,
    linkage_matrix: NDArray[Any],
    linkage_indices: list[list[Any]],
    **kwargs: Any,
) -> float:
    n1, n2 = len(linkage_indices[i_merged_1][1]), len(linkage_indices[i_merged_2][1])
    distance = (linkage_matrix[i][i_merged_1] * n1 + linkage_matrix[i][i_merged_2] * n2) / (n1 + n2)
    return float(distance)


def _ward_clusters_distance(
    i: int,
    i_merged_1: int,
    i_merged_2: int,
    linkage_matrix: NDArray[Any],
    linkage_indices: list[list[Any]],
    **kwargs: Any,
) -> float:
    n = len(linkage_indices[i][1])
    n1, n2 = len(linkage_indices[i_merged_1][1]), len(linkage_indices[i_merged_2][1])
    d0 = linkage_matrix[i_merged_1][i_merged_2]
    d1, d2 = linkage_matrix[i][i_merged_1], linkage_matrix[i][i_merged_2]

    distance_squared = ((n1 + n) * d1**2 + (n2 + n) * d2**2 - n * d0**2) / (n1 + n2 + n)
    distance = np.sqrt(distance_squared)
    return float(distance)
