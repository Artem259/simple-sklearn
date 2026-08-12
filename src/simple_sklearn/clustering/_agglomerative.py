"""Agglomerative Clustering.

This module provides the `AgglomerativeClustering` class.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data
from typing_extensions import Self

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

    labels_: NDArray[np.int64]
    children_: NDArray[np.int64]
    distances_: NDArray[np.float64]

    def __init__(self, n_clusters: int = 2, linkage: str = "ward") -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fit the agglomerative clustering model.

        Args:
            X: Training instances to cluster. Can be an array-like of shape `(n_samples, n_features)`.
            y: Ignored. Present here for API consistency by convention.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If `n_clusters` or `linkage` parameters are invalid, or
                if `n_clusters` exceeds the number of samples in `X`.
        """
        X = validate_data(self, X)
        X = np.array(X)
        self._validate_self_params(X)

        num_samples = X.shape[0]
        labels = np.arange(num_samples)
        self._init_linkage_matrix(X)
        self._linkage_method = _LINKAGE_METHODS[self.linkage]
        self._cluster_to_index = np.arange(2 * num_samples - 1)
        self._index_to_cluster = np.arange(num_samples)
        self._cluster_size = np.ones(num_samples, dtype=np.int64)
        target_labels = labels.copy()
        children = []
        distances = []

        num_clusters = num_samples
        for _ in range(num_samples - 1):
            child, distance = self._merge_clusters_iter(labels, num_clusters)
            children.append(child)
            distances.append(distance)

            num_clusters -= 1
            if num_clusters == self.n_clusters:
                target_labels = labels.copy()

        self.labels_ = np.unique(target_labels, return_inverse=True)[1]
        self.children_ = np.array(children, dtype=np.int64)
        self.distances_ = np.array(distances, dtype=np.float64)
        return self

    def _init_linkage_matrix(self, X: NDArray[Any]) -> None:
        """Initialize the distance matrix.

        Computes the initial distance matrix with its diagonal set to infinity.
        The result is stored in the `_linkage_matrix` attribute.

        Args:
            X: The original input data.
        """
        linkage_matrix = _tools.calc_distance_matrix(X, X)
        np.fill_diagonal(linkage_matrix, np.inf)
        self._linkage_matrix = linkage_matrix

    def _merge_clusters_iter(self, labels: NDArray[Any], num_clusters: int) -> tuple[list[int], float]:
        """Perform a single iteration of merging the two closest clusters.

        Finds the minimum distance in the active portion of the linkage matrix,
        updates the active matrix in-place with the new distances, and maintains
        the mappings between matrix indices and cluster IDs.

        Args:
            labels: The current cluster assignments for each sample. Note that
                this array is modified in-place to reflect the new cluster assignments.
            num_clusters: The number of currently active clusters before the merge.

        Returns:
            child: A list of the two cluster IDs that were merged.
            distance: The computed distance between the merged clusters.
        """
        active_matrix = self._linkage_matrix[:num_clusters, :num_clusters]
        unraveled = np.unravel_index(np.argmin(active_matrix), active_matrix.shape)
        index1, index2 = merged_indices = [int(unraveled[0]), int(unraveled[1])]
        distance = float(self._linkage_matrix[tuple(merged_indices)])
        child = [int(self._index_to_cluster[index1]), int(self._index_to_cluster[index2])]

        new_cluster = 2 * len(labels) - num_clusters
        new_lm_array = self._linkage_method(
            merged_indices=merged_indices,
            cluster_size=self._cluster_size[:num_clusters],
            linkage_matrix=active_matrix,
        )
        new_lm_array[merged_indices] = np.inf

        labels[(labels == child[0]) | (labels == child[1])] = new_cluster

        last_active_index = num_clusters - 1
        cluster_at_last = int(self._index_to_cluster[last_active_index])
        self._linkage_matrix[index1, :num_clusters] = self._linkage_matrix[:num_clusters, index1] = new_lm_array
        self._linkage_matrix[index2, :] = self._linkage_matrix[last_active_index, :]
        self._linkage_matrix[:, index2] = self._linkage_matrix[:, last_active_index]
        self._linkage_matrix[last_active_index, :] = self._linkage_matrix[:, last_active_index] = np.inf

        self._cluster_to_index[new_cluster] = index1
        self._cluster_to_index[cluster_at_last] = index2
        self._cluster_to_index[child] = -1

        self._index_to_cluster[index1] = new_cluster
        self._index_to_cluster[index2] = cluster_at_last

        self._cluster_size[index1] = self._cluster_size[merged_indices].sum()
        self._cluster_size[index2] = self._cluster_size[last_active_index]
        self._cluster_size[last_active_index] = -1

        return child, distance

    def _validate_self_params(self, X: NDArray[Any]) -> None:
        """Validate the hyperparameters against the training data.

        Args:
            X: Training instances. Array of shape `(n_samples, n_features)`.

        Raises:
            ValueError: If `n_clusters` is not a positive integer,
                if `linkage` is not one of the supported string literals, or
                if `n_clusters` is greater than the number of samples.
        """
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError(
                f"The 'n_clusters' parameter must be an int in the range [1, inf). Got {self.n_clusters} instead."
            )
        if self.linkage not in _LINKAGE_METHODS:
            supported = ", ".join(f"'{k}'" for k in _LINKAGE_METHODS)
            raise ValueError(
                f"The 'linkage' parameter must be a str among {{{supported}}}. Got '{self.linkage}' instead."
            )
        num_samples = X.shape[0]
        if self.n_clusters > num_samples:
            raise ValueError(
                f"Cannot extract more clusters than samples: "
                f"{self.n_clusters} clusters were given for a tree with {num_samples} leaves."
            )


def _single_clusters_distance(merged_indices: list[int], linkage_matrix: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
    distances: NDArray[Any] = linkage_matrix[merged_indices].min(axis=0)
    return distances


def _complete_clusters_distance(merged_indices: list[int], linkage_matrix: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
    distances: NDArray[Any] = linkage_matrix[merged_indices].max(axis=0)
    return distances


def _average_clusters_distance(
    merged_indices: list[int],
    cluster_size: NDArray[Any],
    linkage_matrix: NDArray[Any],
    **kwargs: Any,
) -> NDArray[Any]:
    index1, index2 = merged_indices
    n1, n2 = cluster_size[merged_indices]
    distances: NDArray[Any] = (linkage_matrix[index1] * n1 + linkage_matrix[index2] * n2) / (n1 + n2)
    return distances


def _ward_clusters_distance(
    merged_indices: list[int],
    cluster_size: NDArray[Any],
    linkage_matrix: NDArray[Any],
    **kwargs: Any,
) -> NDArray[Any]:
    n1, n2 = cluster_size[merged_indices]
    d0 = linkage_matrix[tuple(merged_indices)]
    d1, d2 = linkage_matrix[merged_indices]

    distances: NDArray[Any] = np.sqrt(
        ((n1 + cluster_size) * d1**2 + (n2 + cluster_size) * d2**2 - cluster_size * d0**2) / (n1 + n2 + cluster_size)
    )
    return distances


_LINKAGE_METHODS: dict[str, Callable[..., NDArray[Any]]] = {
    "single": _single_clusters_distance,
    "complete": _complete_clusters_distance,
    "average": _average_clusters_distance,
    "ward": _ward_clusters_distance,
}
