"""K-Means Clustering.

This module provides the `KMeans` class.
"""

import numbers
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _tools
from ._base_partitional import BasePartitionalClustering


class KMeans(BasePartitionalClustering):
    """Perform K-Means clustering.

    K-Means clustering partitions data into `n_clusters` by minimizing the within-cluster
    sum of squares (inertia). It iteratively updates cluster centers and reassigns
    data points until convergence is reached or the maximum number of iterations is met.

    Args:
        n_clusters: The number of clusters to form as well as the number of
            centroids to generate.
        init: Method for initialization. Can be "random" to choose random
            samples from the dataset for the initial centroids, or an array-like of shape
            `(n_clusters, n_features)` for explicit initialization.
        max_iter: Maximum number of iterations of the k-means algorithm for a single run.
        e: Absolute tolerance in regard to the maximum distance between cluster centers
            of two consecutive iterations to declare convergence.
        random_state: Determines random number generation for centroid initialization.
            Pass an int to make the randomness deterministic.

    Attributes:
        cluster_centers_: An array of shape `(n_clusters, n_features)` representing
            the coordinates of cluster centers.
        labels_: Cluster labels for each point.
        n_iter_: The number of iterations the algorithm ran before convergence or stopping.
        inertia_: Sum of squared distances of samples to their closest cluster center.
        random_state_: The validated `RandomState` instance used for internal operations.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | NDArray[Any] | list[Any] = "random",
        max_iter: int = 300,
        e: float = 1e-4,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.e = e
        self.random_state = random_state

    def _init_fit(self, X: NDArray[Any]) -> None:
        """No preliminary setup required for K-Means."""
        pass

    def _init_cluster_centers(self, X: NDArray[Any]) -> NDArray[np.float64]:
        if isinstance(self.init, str) and self.init == "random":
            random_indices = self.random_state_.choice(X.shape[0], self.n_clusters, replace=False)
            return np.array(X[random_indices])
        return np.array(self.init)

    def _recalc_cluster_centers(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Recalculate cluster centers based on the current labels.

        If a cluster becomes empty, its center remains unchanged from the previous iteration.
        """
        return np.array(
            [
                X[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) else self.cluster_centers_[i]
                for i in range(self.n_clusters)
            ]
        )

    def _recalc_labels(self, X: NDArray[Any]) -> NDArray[np.int_]:
        """Recalculate labels by finding the closest cluster center for each sample."""
        distances = _tools.calc_distance_matrix(X, self.cluster_centers_)
        return np.asarray(np.argmin(distances, axis=1))

    def _check_convergence(self, old_cluster_centers: NDArray[np.float64]) -> bool:
        """Check if the algorithm has converged.

        Convergence is declared if the maximum distance between old and new
        cluster centers is less than or equal to the absolute tolerance `e`.
        """
        max_centers_dist_diff = _tools.calc_max_zip_distance(self.cluster_centers_, old_cluster_centers)
        return max_centers_dist_diff <= self.e

    def _calc_inertia(self, X: NDArray[Any]) -> float:
        """Calculate the inertia of the cluster assignments.

        Inertia for K-Means is the sum of squared distances of each sample
        to its closest cluster center.
        """
        distances = _tools.calc_distance_matrix(X, self.cluster_centers_)
        return float(np.sum(np.min(distances, axis=1) ** 2))

    def _validate_self_params(self, X: NDArray[Any]) -> None:
        """Validate the hyperparameters against the input data.

        Raises:
            ValueError: If `e` is negative.
        """
        if not isinstance(self.e, numbers.Real) or self.e < 0:
            raise ValueError(f"The 'e' parameter must be a float in the range [0, inf). Got '{self.e}'.")
