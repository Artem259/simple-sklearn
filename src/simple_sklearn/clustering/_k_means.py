"""K-Means Clustering.

This module provides the `KMeans` class.
"""

import numbers
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data

from . import _tools


class KMeans(ClusterMixin, BaseEstimator):  # type: ignore
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

    cluster_centers_: NDArray[np.float64]
    labels_: NDArray[np.int_]
    n_iter_: int
    inertia_: float
    random_state_: np.random.RandomState

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | NDArray[Any] | list[Any] = "random",
        max_iter: int = 300,
        e: float = 1e-4,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.e = e
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "KMeans":
        """Fit the K-Means clustering model.

        Args:
            X: Training instances to cluster. Can be an array-like of shape `(n_samples, n_features)`.
            y: Ignored. Present here for API consistency by convention.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If any hyperparameters are invalid.
        """
        self.__validate_params(X)
        self.random_state_ = check_random_state(self.random_state)
        X = validate_data(self, X)
        X = np.array(X)

        self._init_fit(X)
        self.cluster_centers_ = self._init_cluster_centers(X)
        self.labels_ = self._recalc_labels(X)
        self.n_iter_ = 0

        while self.n_iter_ < self.max_iter:
            self.n_iter_ += 1

            old_cluster_centers = self.cluster_centers_
            self.cluster_centers_ = self._recalc_cluster_centers(X)
            self.labels_ = self._recalc_labels(X)

            if self._check_convergence(old_cluster_centers):
                break

        self.inertia_ = self._calc_inertia(X)
        return self

    def _init_fit(self, X: NDArray[Any]) -> None:
        """Hook for subclasses to perform preliminary setups before the core iterative loop.

        This method is a placeholder and does nothing by default.

        Args:
            X: The original input data.
        """
        pass

    def _init_cluster_centers(self, X: NDArray[Any]) -> NDArray[Any]:
        """Initialize the cluster centers based on the `init` strategy.

        Args:
            X: The original input data.

        Returns:
            An array containing the initial cluster centers.
        """
        if isinstance(self.init, str) and self.init == "random":
            random_indices = self.random_state_.choice(X.shape[0], self.n_clusters, replace=False)
            return np.array(X[random_indices])
        return np.array(self.init)

    def _recalc_cluster_centers(self, X: NDArray[Any]) -> NDArray[Any]:
        """Recalculate cluster centers based on the current labels.

        If a cluster becomes empty, its center remains unchanged from the previous iteration.

        Args:
            X: The original input data.

        Returns:
            An array containing the newly computed cluster centers.
        """
        return np.array(
            [
                X[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) else self.cluster_centers_[i]
                for i in range(self.n_clusters)
            ]
        )

    def _recalc_labels(self, X: NDArray[Any]) -> NDArray[Any]:
        """Recalculate labels by finding the closest cluster center for each sample.

        Args:
            X: The original input data.

        Returns:
            An array of new cluster labels.
        """
        distances = _tools.calc_distance_matrix(X, self.cluster_centers_)
        return np.asarray(np.argmin(distances, axis=1))

    def _check_convergence(self, old_cluster_centers: NDArray[Any]) -> bool:
        """Check if the algorithm has converged.

        Convergence is declared if the maximum distance between old and new
        cluster centers is less than or equal to the absolute tolerance `e`.

        Args:
            old_cluster_centers: The cluster centers from the previous iteration.

        Returns:
            True if the model has converged, False otherwise.
        """
        max_centers_dist_diff = _tools.calc_max_zip_distance(self.cluster_centers_, old_cluster_centers)
        return max_centers_dist_diff <= self.e

    def _calc_inertia(self, X: NDArray[Any]) -> float:
        """Calculate the inertia of the cluster assignments.

        Inertia is the sum of squared distances of each sample to its closest cluster center.

        Args:
            X: The original input data.

        Returns:
            The calculated inertia value.
        """
        distances = _tools.calc_distance_matrix(X, self.cluster_centers_)
        return float(np.sum(np.min(distances, axis=1) ** 2))

    def __validate_params(self, X: Any) -> None:
        """Validate the hyperparameters against the input data.

        Args:
            X: The original input data.

        Raises:
            ValueError: If `n_clusters` or `max_iter` are not positive integers,
                if `e` is negative, or if the `init` strategy or shape is invalid.
        """
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError(
                f"The 'n_clusters' parameter must be an int in the range [1, inf). Got '{self.n_clusters}' instead."
            )
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError(
                f"The 'max_iter' parameter must be an int in the range [1, inf). Got '{self.max_iter}' instead."
            )
        if not isinstance(self.e, numbers.Real) or self.e < 0:
            raise ValueError(f"The 'e' parameter must be a float in the range [0, inf). Got '{self.e}' instead.")

        if isinstance(self.init, str):
            if self.init not in ("random",):
                raise ValueError(
                    f"The 'init' parameter must be array-like or a str among ['random']. Got '{self.init}' instead."
                )
        else:
            init_shape = np.array(self.init).shape
            expected_shape = (self.n_clusters, X.shape[1])
            if len(init_shape) != 2 or init_shape != expected_shape:
                raise ValueError(
                    f"The shape of the initial centers {init_shape} "
                    f"does not match the expected shape (n_clusters, n_features): {expected_shape}."
                )
