import heapq

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class KNeighborsClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, weights="uniform", e=1e-9):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.e = e

    def fit(self, X, y):
        self.__validate_params()
        X, y = validate_data(self, X, y)
        X = np.array(X)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        self.fitted_X_ = X
        self.fitted_y_ = y

        return self

    def predict(self, X):
        self.__validate_params()
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.array(X)

        decision_scores = self._decision_function(X)
        return self.classes_[np.argmax(decision_scores, axis=1)]

    def kneighbors(self, X):
        self.__validate_params()
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return self._kneighbors(X)

    def _decision_function(self, X):
        self.__validate_params()

        decision_scores = []
        for x in X:
            x_neigh_indices = self._find_kneighbors_indices(x, self.n_neighbors)
            x_neigh_labels = self.fitted_y_[x_neigh_indices]
            x_neigh_distances, x_neigh_distances_squared = self._calc_distances(
                x, X_targets=self.fitted_X_[x_neigh_indices]
            )
            if self.weights == "distance":
                weights = 1 / (x_neigh_distances + self.e)
                x_decision_scores = np.bincount(x_neigh_labels, minlength=len(self.classes_), weights=weights)
            elif self.weights == "distance_squared":
                weights = 1 / (x_neigh_distances_squared + self.e)
                x_decision_scores = np.bincount(x_neigh_labels, minlength=len(self.classes_), weights=weights)
            else:  # self.weights == 'uniform'
                x_decision_scores = np.bincount(x_neigh_labels, minlength=len(self.classes_))
            decision_scores.append(x_decision_scores)

        return np.array(decision_scores)

    def _kneighbors(self, X):
        neigh_distances = []
        neigh_indices = []
        for x in X:
            x_neigh_indices = self._find_kneighbors_indices(x, self.n_neighbors)
            x_neigh_distances, _ = self._calc_distances(x, X_targets=self.fitted_X_[x_neigh_indices])
            neigh_distances.append(x_neigh_distances)
            neigh_indices.append(x_neigh_indices)

        return np.array(neigh_distances), np.array(neigh_indices)

    def _find_kneighbors_indices(self, x, n_neighbors: int):
        indices = list(range(self.fitted_X_.shape[0]))
        _, distances_squared = self._calc_distances(x)
        neigh_indices = heapq.nsmallest(n_neighbors, indices, key=lambda i: distances_squared[i])
        return np.array(neigh_indices)

    def _calc_distances(self, x_source, X_targets=None):
        if X_targets is None:
            X_targets = self.fitted_X_
        distances_squared = np.sum((X_targets - x_source) ** 2, axis=1)
        distances = np.sqrt(distances_squared)
        return distances, distances_squared

    def __validate_params(self):
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
        if not isinstance(self.e, float) or not 0 < self.e < 1:
            raise ValueError(
                f"The 'e' parameter of KNeighborsClassifier must be a float in the range (0, 1). "
                f"Got '{self.e}' instead."
            )
