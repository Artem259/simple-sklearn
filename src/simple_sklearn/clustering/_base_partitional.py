"""Partitional Clustering.

This module provides the `BasePartitionalClustering` abstract class.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data
from typing_extensions import Self


class BasePartitionalClustering(ClusterMixin, BaseEstimator, ABC):  # type: ignore
    """Abstract base class for partitional clustering (e.g., K-Means, K-Medoids).

    This class implements generalized Lloyd's algorithm.
    Subclasses must implement the abstract methods to define specific behaviors
    for initialization, center calculation, and distance metrics.

    Args:
        n_clusters: The number of clusters to form.
        init: Method for initialization. Can be "random" to choose random
            samples from the dataset for the initial cluster centers, or an array-like of shape
            `(n_clusters, n_features)` for explicit initialization.
        max_iter: Maximum number of iterations of the algorithm for a single run.
        random_state: Determines random number generation for initialization.

    Attributes:
        cluster_centers_: An array of shape `(n_clusters, n_features)` representing
            the coordinates of cluster centers.
        labels_: Cluster labels for each point.
        n_iter_: The number of iterations the algorithm ran before convergence or stopping.
        inertia_: Target minimized metric.
        random_state_: The validated `RandomState` instance used for internal operations.
    """

    cluster_centers_: NDArray[np.float64]
    labels_: NDArray[np.int64]
    n_iter_: int
    inertia_: float
    random_state_: np.random.RandomState

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | NDArray[Any] | list[Any] = "random",
        max_iter: int = 300,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fit the clustering model.

        Args:
            X: Training instances to cluster. Can be an array-like of shape `(n_samples, n_features)`.
            y: Ignored. Present here for API consistency by convention.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If any hyperparameters are invalid.
        """
        X = validate_data(self, X)
        X = np.array(X)
        self._validate_base_params(X)
        self._validate_self_params(X)

        self.random_state_ = check_random_state(self.random_state)
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

    @abstractmethod
    def _init_fit(self, X: NDArray[Any]) -> None:
        """Perform preliminary setups before the core iterative loop.

        Args:
            X: The original input data.
        """
        pass

    @abstractmethod
    def _init_cluster_centers(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Initialize the cluster centers based on the `init` strategy.

        Args:
            X: The original input data.

        Returns:
            An array containing the initial cluster centers.
        """
        pass

    @abstractmethod
    def _recalc_cluster_centers(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Recalculate cluster centers.

        Args:
            X: The original input data.

        Returns:
            An array containing the newly computed cluster centers.
        """
        pass

    @abstractmethod
    def _recalc_labels(self, X: NDArray[Any]) -> NDArray[np.int64]:
        """Recalculate labels for each sample.

        Args:
            X: The original input data.

        Returns:
            An array of new cluster labels.
        """
        pass

    @abstractmethod
    def _check_convergence(self, old_cluster_centers: NDArray[np.float64]) -> bool:
        """Check if the algorithm has converged.

        Args:
            old_cluster_centers: The cluster centers from the previous iteration.

        Returns:
            True if the model has converged, False otherwise.
        """
        pass

    @abstractmethod
    def _calc_inertia(self, X: NDArray[Any]) -> float:
        """Calculate the inertia of the cluster assignments.

        Args:
            X: The original input data.

        Returns:
            The calculated inertia value.
        """
        pass

    @abstractmethod
    def _validate_self_params(self, X: NDArray[Any]) -> None:
        """Validate algorithm-specific hyperparameters against the input data.

        Args:
            X: The original input data.
        """
        pass

    def _validate_base_params(self, X: NDArray[Any]) -> None:
        """Validate the common hyperparameters against the input data.

        Args:
            X: The original input data.

        Raises:
            ValueError: If `n_clusters` or `max_iter` is not a positive integer,
                if `n_clusters` is greater than the number of samples, or
                if the `init` strategy or shape is invalid.
        """
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError(
                f"The 'n_clusters' parameter of {type(self).__name__} must be an int in the range [1, inf). "
                f"Got {self.n_clusters} instead."
            )
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError(
                f"The 'max_iter' parameter of {type(self).__name__} must be an int in the range [1, inf). "
                f"Got {self.max_iter} instead."
            )
        num_samples = X.shape[0]
        if self.n_clusters > num_samples:
            raise ValueError(f"n_samples={num_samples} should be >= n_clusters={self.n_clusters}.")

        init_value_error = ValueError(
            f"The 'init' parameter of {type(self).__name__} must be array-like or a str among {{'random'}}. "
            f"Got '{self.init}' instead."
        )
        if isinstance(self.init, str):
            if self.init not in ("random",):
                raise init_value_error
        elif not hasattr(self.init, "__iter__"):
            raise init_value_error
        else:
            init_shape = np.asarray(self.init).shape
            expected_shape = (self.n_clusters, X.shape[1])
            if len(init_shape) != 2 or init_shape != expected_shape:
                raise ValueError(
                    f"The shape of the initial centers {init_shape} "
                    f"does not match the expected shape (n_clusters, n_features): {expected_shape}."
                )
