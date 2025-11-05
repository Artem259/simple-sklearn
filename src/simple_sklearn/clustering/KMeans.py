import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data

from . import tools


class KMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=8, init='random', max_iter=300, e=1e-4, random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.e = e
        self.random_state = random_state

    def fit(self, X, y=None):
        self.__validate_params()
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

    def _init_fit(self, X):
        pass

    def _init_cluster_centers(self, X):
        if isinstance(self.init, str) and self.init == 'random':
            random_indices = self.random_state_.choice(X.shape[0], self.n_clusters, replace=False)
            return np.array(X[random_indices])
        return np.array(self.init)

    def _recalc_cluster_centers(self, X):
        return np.array([
            X[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) else self.cluster_centers_[i]
            for i in range(self.n_clusters)
        ])

    def _recalc_labels(self, X):
        distances = tools.calc_distance_matrix(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def _check_convergence(self, old_cluster_centers):
        max_centers_dist_diff = tools.calc_max_zip_distance(
            self.cluster_centers_,
            old_cluster_centers
        )
        return max_centers_dist_diff <= self.e

    def _calc_inertia(self, X):
        distances = tools.calc_distance_matrix(X, self.cluster_centers_)
        return float(np.sum(np.min(distances, axis=1) ** 2))

    def __validate_params(self):
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError(
                f"The 'n_clusters' parameter must be an int in the range [1, inf). "
                f"Got '{self.n_clusters}' instead."
            )
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError(
                f"The 'max_iter' parameter must be an int in the range [1, inf). "
                f"Got '{self.max_iter}' instead."
            )
        if not (isinstance(self.e, float) or isinstance(self.e, int)) or self.e < 0:
            raise ValueError(
                f"The 'e' parameter must be a float in the range [0, inf). "
                f"Got '{self.e}' instead."
            )

        if isinstance(self.init, str):
            if self.init not in ('random', ):
                raise ValueError(
                    f"The 'init' parameter must be array-like or a str among ['random']. "
                    f"Got '{self.init}' instead."
                )
        else:
            init_shape = np.array(self.init).shape
            if not init_shape or init_shape[0] != self.n_clusters:
                raise ValueError(
                    f"The shape of the initial centers {init_shape} "
                    f"does not match the number of clusters {self.n_clusters}."
                )
