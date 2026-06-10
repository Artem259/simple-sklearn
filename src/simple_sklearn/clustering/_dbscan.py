"""DBSCAN Clustering.

This module provides the `DBSCAN` class.
"""

import numbers
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data
from typing_extensions import Self

from . import _tools


class DBSCAN(ClusterMixin, BaseEstimator):  # type: ignore
    """Perform DBSCAN clustering.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds core
    samples of high density and expands clusters from them. It is highly effective
    for data containing clusters of similar density and for filtering out noise.

    Args:
        eps: The maximum distance between two samples for one to be considered
            as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be
            considered as a core point. This includes the point itself.

    Attributes:
        labels_: Cluster labels for each point. Noisy samples are assigned the label `-1`.
        distance_matrix_: The precomputed pairwise distance matrix between all points.
        neighbors_: A list of lists containing the indices of neighbors within `eps` for each sample.
        core_sample_indices_: An array of indices identifying the core samples.
    """

    labels_: NDArray[np.int64]
    distance_matrix_: NDArray[np.float64]
    neighbors_: list[list[int]]
    core_sample_indices_: NDArray[np.int64]
    _core_sample_mask: NDArray[np.bool_]

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fit the DBSCAN clustering model.

        Args:
            X: Training instances to cluster. Can be an array-like of shape `(n_samples, n_features)`.
            y: Ignored. Present here for API consistency by convention.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If `eps` or `min_samples` parameters are invalid.
        """
        X = validate_data(self, X)
        X = np.array(X)
        self._validate_self_params()

        num_samples = X.shape[0]
        self.labels_ = np.full(num_samples, -1)
        self.distance_matrix_ = _tools.calc_distance_matrix(X, X)
        self.neighbors_ = self._init_neighbors()
        self._init_core_samples()

        cluster_id = 0
        for i in range(num_samples):
            if self._is_assigned_sample(i):
                continue
            if self._is_core_sample(i):
                self._expand_cluster(i, cluster_id)
                cluster_id += 1

        return self

    def _init_neighbors(self) -> list[list[int]]:
        """Initialize the neighborhood for each sample.

        Finds all neighbors within the `eps` distance for every sample in the dataset
        based on the precomputed distance matrix.

        Returns:
            A list of lists where each sublist contains the indices of the neighbors for a given sample.
        """
        num_samples = self.distance_matrix_.shape[0]
        return [
            [j for j in range(num_samples) if i != j and self.distance_matrix_[i, j] <= self.eps]
            for i in range(num_samples)
        ]

    def _init_core_samples(self) -> None:
        """Identify and store core sample mask and indices.

        A sample is considered a core sample if it has at least `min_samples` neighbors
        (including itself). Since the neighbor list excludes the point itself, the threshold
        is checked against `min_samples - 1`.

        This method populates an internal boolean mask for optimized lookups during traversal,
        and exposes the indices as the public `core_sample_indices_` attribute.
        """
        num_samples = self.distance_matrix_.shape[0]
        self._core_sample_mask = np.array(
            [len(self.neighbors_[i]) >= self.min_samples - 1 for i in range(num_samples)], dtype=bool
        )
        self.core_sample_indices_ = np.where(self._core_sample_mask)[0]

    def _expand_cluster(self, i: int, cluster_id: int) -> None:
        """Expand a cluster from a core sample.

        Traverses the neighborhood of a core sample and adds reachable points to the
        current cluster. If a reached point is also a core sample, its neighbors are
        added to the stack to continue the expansion.

        Args:
            i: The index of the initial core sample.
            cluster_id: The identifier assigned to the newly formed cluster.
        """
        self.labels_[i] = cluster_id

        i_neighbors = self.neighbors_[i]
        stack = list(i_neighbors)
        while stack:
            j = stack.pop()
            if self._is_assigned_sample(j):
                continue
            self.labels_[j] = cluster_id
            if self._is_core_sample(j):
                stack.extend(self.neighbors_[j])

    def _is_assigned_sample(self, index: int) -> bool:
        """Check if a sample has already been assigned to a cluster.

        Args:
            index: The index of the sample to check.

        Returns:
            True if the sample has been assigned a cluster label, False otherwise.
        """
        return bool(self.labels_[index] != -1)

    def _is_core_sample(self, index: int) -> bool:
        """Check if a sample is a core sample utilizing the precomputed mask.

        Args:
            index: The index of the sample to check.

        Returns:
            True if the sample is a core sample, False otherwise.
        """
        return bool(self._core_sample_mask[index])

    def _validate_self_params(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If `eps` is not a positive real number or
                if `min_samples` is not a positive integer.
        """
        if not isinstance(self.eps, numbers.Real) or self.eps <= 0:
            raise ValueError(f"The 'eps' parameter must be a float in the range (0, inf). Got {self.eps} instead.")
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError(
                f"The 'min_samples' parameter must be an int in the range [1, inf). Got {self.min_samples} instead."
            )
