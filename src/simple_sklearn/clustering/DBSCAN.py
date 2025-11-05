import numpy as np
from collections import deque
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data

from . import tools


class DBSCAN(ClusterMixin, BaseEstimator):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        self.__validate_params()
        X = validate_data(self, X)
        X = np.array(X)

        num_samples = X.shape[0]
        self.labels_ = np.full(num_samples, -1)
        self.distance_matrix_ = tools.calc_distance_matrix(X, X)
        self.neighbors_ = self._init_neighbors()
        self.core_sample_indices_ = self._init_core_sample_indices()

        cluster_id = 0
        for i in range(num_samples):
            if self._is_visited_sample(i):
                continue
            if self._is_core_sample(i):
                self._expand_cluster(i, cluster_id)
                cluster_id += 1

        return self

    def _init_neighbors(self):
        num_samples = self.distance_matrix_.shape[0]
        return [
            [
                j for j in range(num_samples)
                if i != j and self.distance_matrix_[i, j] <= self.eps
            ]
            for i in range(num_samples)
        ]

    def _init_core_sample_indices(self):
        num_samples = self.distance_matrix_.shape[0]
        return np.array([
            i for i in range(num_samples)
            if len(self.neighbors_[i]) >= self.min_samples - 1
        ])

    def _expand_cluster(self, i, cluster_id):
        self.labels_[i] = cluster_id

        i_neighbors = self.neighbors_[i]
        queue = deque(i_neighbors)
        while queue:
            j = queue.pop()
            if self._is_visited_sample(j):
                continue
            self.labels_[j] = cluster_id
            if self._is_core_sample(j):
                queue.extend(self.neighbors_[j])

    def _is_visited_sample(self, index):
        return self.labels_[index] != -1

    def _is_core_sample(self, index):
        neighbors = self.neighbors_[index]
        return len(neighbors) >= self.min_samples - 1

    def __validate_params(self):
        if not (isinstance(self.eps, float) or isinstance(self.eps, int)) or self.eps <= 0:
            raise ValueError(
                f"The 'eps' parameter must be a float in the range (0, inf). "
                f"Got '{self.eps}' instead."
            )
        if not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError(
                f"The 'min_samples' parameter must be an int in the range [1, inf). "
                f"Got '{self.min_samples}' instead."
            )
