"""K-Nearest Neighbors Classification.

This module provides the `KNeighborsClassifier` class.
"""

import heapq
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from typing_extensions import Self


class KNeighborsClassifier(ClassifierMixin, BaseEstimator):  # type: ignore
    """Perform K-Nearest Neighbors classification.

    The k-nearest neighbors algorithm classifies a new data point based on the
    majority class among its `n_neighbors` closest neighbors in the training set. It
    supports uniform voting weight as well as weighting by the inverse of the
    distance or squared distance.

    Args:
        n_neighbors: The number of nearest neighbors.
        weights: The weight function used in prediction. Must be one of "uniform",
            "distance", or "distance_squared".
        eps: A small float added to distances to prevent division by zero when
            calculating weights.

    Attributes:
        classes_: The unique class labels observed in the training data.
        fitted_x_: The validated training input data array.
        fitted_y_: The validated and label-encoded target values array.
    """

    classes_: NDArray[Any]
    fitted_x_: NDArray[Any]
    fitted_y_: NDArray[Any]

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", eps: float = 1e-9) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.eps = eps

    def fit(self, X: Any, y: Any) -> Self:
        """Fit the k-nearest neighbors classification model.

        Args:
            X: Training instances to fit the model. Can be an array-like of shape `(n_samples, n_features)`.
            y: Target values (class labels) for the training instances. Array-like of shape `(n_samples,)`.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If `y` is of a continuous type or if any hyperparameters are invalid.
        """
        X, y = validate_data(self, X, y)
        X = np.array(X)
        self._validate_self_params()

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        self.fitted_x_ = X
        self.fitted_y_ = y

        return self

    def predict(self, X: Any) -> NDArray[Any]:
        """Predict class labels for the given input data.

        Args:
            X: Instances to predict. Can be an array-like of shape `(n_samples, n_features)`.

        Returns:
            An array of shape `(n_samples,)` containing the predicted class labels for each sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.array(X)

        decision_scores = self._decision_function(X)
        return np.asarray(self.classes_[np.argmax(decision_scores, axis=1)])

    def kneighbors(self, X: Any) -> tuple[NDArray[Any], NDArray[Any]]:
        """Find the k-nearest neighbors of each sample in the given input data.

        Args:
            X: Instances to evaluate. Can be an array-like of shape `(n_samples, n_features)`.

        Returns:
            distances: An array of shape `(n_samples, n_neighbors)` representing the
                Euclidean distances to the nearest points.
            indices: An array of shape `(n_samples, n_neighbors)` representing the
                indices of the nearest points in the fitted training data.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self._kneighbors(X)

    def _decision_function(self, X: NDArray[Any]) -> NDArray[Any]:
        """Compute the decision scores for each class based on neighbor votes.

        Calculates the accumulated weights or unweighted counts for each class
        based on the `self.n_neighbors` nearest neighbors.

        Args:
            X: Instances to evaluate. An array of shape `(n_samples, n_features)`.

        Returns:
            An array of shape `(n_samples, n_classes)` containing the decision scores
                (accumulated votes) for each sample and class.
        """
        decision_scores = []
        for x in X:
            x_neigh_indices = self._find_kneighbors_indices(x, self.n_neighbors)
            x_neigh_labels = self.fitted_y_[x_neigh_indices]
            x_neigh_distances, x_neigh_distances_squared = self._calc_distances(
                x, X_targets=self.fitted_x_[x_neigh_indices]
            )
            if self.weights == "distance":
                weights = 1 / (x_neigh_distances + self.eps)
                x_decision_scores = np.bincount(x_neigh_labels, minlength=len(self.classes_), weights=weights)
            elif self.weights == "distance_squared":
                weights = 1 / (x_neigh_distances_squared + self.eps)
                x_decision_scores = np.bincount(x_neigh_labels, minlength=len(self.classes_), weights=weights)
            else:  # self.weights == 'uniform'
                x_decision_scores = np.bincount(x_neigh_labels, minlength=len(self.classes_))
            decision_scores.append(x_decision_scores)

        return np.array(decision_scores)

    def _kneighbors(self, X: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
        """Compute distances and indices of k-nearest neighbors of each sample.

        Args:
            X: Instances to evaluate. An array of shape `(n_samples, n_features)`.

        Returns:
            neigh_distances: Array of shape `(n_samples, n_neighbors)` containing Euclidean distances.
            neigh_indices: Array of shape `(n_samples, n_neighbors)` containing indices in `self.fitted_x_`.
        """
        neigh_distances = []
        neigh_indices = []
        for x in X:
            x_neigh_indices = self._find_kneighbors_indices(x, self.n_neighbors)
            x_neigh_distances, _ = self._calc_distances(x, X_targets=self.fitted_x_[x_neigh_indices])
            neigh_distances.append(x_neigh_distances)
            neigh_indices.append(x_neigh_indices)

        return np.array(neigh_distances), np.array(neigh_indices)

    def _find_kneighbors_indices(self, x: NDArray[Any], n_neighbors: int) -> NDArray[Any]:
        """Find the indices of the nearest neighbors for a single sample.

        Uses a heap queue to efficiently find the `n_neighbors` samples from `self.fitted_x_`
        with the smallest squared Euclidean distance.

        Args:
            x: A single instance to evaluate. An array-like of shape `(n_features,)`.
            n_neighbors: The number of nearest neighbors to find.

        Returns:
            An array of shape `(n_neighbors,)` containing the indices of the nearest
            neighbors in the training data.
        """
        indices = list(range(self.fitted_x_.shape[0]))
        _, distances_squared = self._calc_distances(x)
        neigh_indices = heapq.nsmallest(n_neighbors, indices, key=lambda i: distances_squared[i])
        return np.array(neigh_indices)

    def _calc_distances(
        self, x_source: NDArray[Any], X_targets: NDArray[Any] | None = None
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Calculate the Euclidean and squared Euclidean distances from a source point.

        Args:
            x_source: The source instance. An array of shape `(n_features,)`.
            X_targets: The target instances to measure against. If `None`, distances are
                calculated against all fitted training instances. An array of shape
                `(n_targets, n_features)`.

        Returns:
            distances: An array of shape `(n_targets,)` containing Euclidean distances.
            distances_squared: An array of shape `(n_targets,)` containing squared Euclidean distances.
        """
        if X_targets is None:
            X_targets = self.fitted_x_
        distances_squared = np.sum((X_targets - x_source) ** 2, axis=1)
        distances = np.sqrt(distances_squared)
        return distances, distances_squared

    def _validate_self_params(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If `n_neighbors` is not a positive integer, if `weights` is not
                one of the supported string literals, or if `eps` is not a float in the range (0, 1).
        """
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError(
                f"The 'n_neighbors' parameter of KNeighborsClassifier must be an int in the range [1, inf). "
                f"Got '{self.n_neighbors}' instead."
            )
        if self.weights not in ("distance", "distance_squared", "uniform"):
            raise ValueError(
                f"The 'weights' parameter of KNeighborsClassifier must be a str among "
                f"['distance', 'distance_squared', 'uniform']. Got '{self.weights}' instead."
            )
        if not isinstance(self.eps, float) or not 0 < self.eps < 1:
            raise ValueError(
                f"The 'eps' parameter of KNeighborsClassifier must be a float in the range (0, 1). "
                f"Got '{self.eps}' instead."
            )
