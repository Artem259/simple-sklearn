from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _tools
from .k_means import KMeans


class KMedoids(KMeans):
    def __init__(
        self,
        n_clusters: int = 8,
        init: str | NDArray[Any] | list[Any] = "random",
        max_iter: int = 300,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            e=0,
            random_state=random_state,
        )

    def _init_fit(self, X: NDArray[Any]) -> None:
        self.distance_matrix_ = _tools.calc_distance_matrix(X, X)

    def _init_cluster_centers(self, X: NDArray[Any]) -> NDArray[Any]:
        if isinstance(self.init, str) and self.init == "random":
            indices = self.random_state_.choice(X.shape[0], self.n_clusters, replace=False)
        else:
            kmeans_cluster_centers = super()._init_cluster_centers(X)
            indices = _convert_kmeans_cluster_centers(X, kmeans_cluster_centers)

        self.cluster_center_indices_ = indices
        return np.asarray(X[indices])

    def _recalc_cluster_centers(self, X: NDArray[Any]) -> NDArray[Any]:
        new_indices = []
        new_centers = []

        for i in range(self.n_clusters):
            cluster_indices = np.where(self.labels_ == i)[0]

            if cluster_indices.size == 0:
                old_index = self.cluster_center_indices_[i]
                new_indices.append(old_index)
                new_centers.append(X[old_index])
                continue

            cluster_distances = self.distance_matrix_[np.ix_(cluster_indices, cluster_indices)]
            costs = np.sum(cluster_distances, axis=1)
            best_medoid_global_index = cluster_indices[np.argmin(costs)]

            new_indices.append(best_medoid_global_index)
            new_centers.append(X[best_medoid_global_index])

        self.cluster_center_indices_ = np.array(new_indices)
        return np.array(new_centers)

    def _recalc_labels(self, X: NDArray[Any]) -> NDArray[Any]:
        distances_to_medoids = self.distance_matrix_[:, self.cluster_center_indices_]
        return np.asarray(np.argmin(distances_to_medoids, axis=1))

    def _check_convergence(self, old_cluster_centers: NDArray[Any]) -> bool:
        return np.array_equal(self.cluster_centers_, old_cluster_centers)

    def _calc_inertia(self, X: NDArray[Any]) -> float:
        distances_to_medoids = self.distance_matrix_[:, self.cluster_center_indices_]
        min_distances = np.min(distances_to_medoids, axis=1)
        return float(np.sum(min_distances))


def _convert_kmeans_cluster_centers(X: NDArray[Any], kmeans_cluster_centers: NDArray[Any]) -> NDArray[Any]:
    indices_with_centers = [_tools.find_closest_point(X, center) for center in kmeans_cluster_centers]
    indices, _ = zip(*indices_with_centers, strict=True)
    return np.array(indices)
