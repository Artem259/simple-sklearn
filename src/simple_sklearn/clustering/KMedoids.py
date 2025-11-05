import numpy as np

from . import tools
from .KMeans import KMeans


class KMedoids(KMeans):
    def __init__(self, n_clusters=8, init='random', max_iter=300, random_state=None):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            e=0,
            random_state=random_state,
        )

    def _init_fit(self, X):
        self.distance_matrix_ = tools.calc_distance_matrix(X, X)

    def _init_cluster_centers(self, X):
        kmeans_cluster_centers = super()._init_cluster_centers(X)
        indices, centers = _convert_kmeans_cluster_centers(X, kmeans_cluster_centers)
        self.cluster_center_indices_ = indices
        return centers

    def _recalc_cluster_centers(self, X):
        kmeans_cluster_centers = super()._recalc_cluster_centers(X)
        indices, centers = _convert_kmeans_cluster_centers(X, kmeans_cluster_centers)
        self.cluster_center_indices_ = indices
        return centers

    def _recalc_labels(self, X):
        distances = self.distance_matrix_[:, self.cluster_center_indices_]
        return np.argmin(distances, axis=1)


def _convert_kmeans_cluster_centers(X, kmeans_cluster_centers):
    indices_with_centers = [tools.find_closest_point(X, center) for center in kmeans_cluster_centers]
    indices, centers = zip(*indices_with_centers)
    return np.array(indices), np.array(centers)
