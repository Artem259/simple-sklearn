"""K-Medoids Clustering.

This module provides the `KMedoids` class.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _tools
from ._k_means import KMeans


class KMedoids(KMeans):
    """Perform K-Medoids clustering.

    K-Medoids clustering partitions data into `n_clusters` by minimizing the sum of
    distances between data points and their corresponding cluster center. Unlike K-Means,
    K-Medoids strictly uses actual data points from the dataset as cluster centers (medoids).
    It iteratively updates the medoids and reassigns data points until convergence is
    reached or the maximum number of iterations is met.

    Args:
        n_clusters: The number of clusters to form as well as the number of
            medoids to generate.
        init: Method for initialization. Can be "random" to choose random
            samples from the dataset for the initial medoids, or an array-like of shape
            `(n_clusters, n_features)` for explicit initialization. If array-like, the
            algorithm will map these coordinates to the closest actual data points.
        max_iter: Maximum number of iterations of the k-medoids algorithm for a single run.
        random_state: Determines random number generation for medoid initialization.
            Pass an int to make the randomness deterministic.

    Attributes:
        cluster_centers_: An array of shape `(n_clusters, n_features)` representing
            the coordinates of the cluster medoids.
        labels_: Cluster labels for each point.
        n_iter_: The number of iterations the algorithm ran before convergence or stopping.
        inertia_: Sum of distances of samples to their closest cluster medoid.
        random_state_: The validated `RandomState` instance used for internal operations.
        distance_matrix_: The precomputed pairwise distance matrix between all points.
        cluster_center_indices_: An array of shape `(n_clusters,)` containing the indices of
            the data points chosen as medoids.
    """

    distance_matrix_: NDArray[np.float64]
    cluster_center_indices_: NDArray[np.int_]

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
        """Perform preliminary setups by calculating the pairwise distance matrix.

        Args:
            X: The original input data.
        """
        self.distance_matrix_ = _tools.calc_distance_matrix(X, X)

    def _init_cluster_centers(self, X: NDArray[Any]) -> NDArray[Any]:
        """Initialize the medoids based on the `init` strategy.

        If `init` is random, indices are chosen directly from the data. Otherwise,
        it maps the provided initial centers to the nearest actual data points.

        Args:
            X: The original input data.

        Returns:
            An array containing the initial medoids.
        """
        if isinstance(self.init, str) and self.init == "random":
            indices = self.random_state_.choice(X.shape[0], self.n_clusters, replace=False)
        else:
            kmeans_cluster_centers = super()._init_cluster_centers(X)
            indices = _convert_kmeans_cluster_centers(X, kmeans_cluster_centers)

        self.cluster_center_indices_ = indices
        return np.asarray(X[indices])

    def _recalc_cluster_centers(self, X: NDArray[Any]) -> NDArray[Any]:
        """Recalculate medoids by finding the point minimizing the distance sum in each cluster.

        If a cluster becomes empty, its medoid remains unchanged from the previous iteration.

        Args:
            X: The original input data.

        Returns:
            An array containing the newly computed medoids.
        """
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
        """Recalculate labels by finding the closest medoid for each sample.

        Args:
            X: The original input data.

        Returns:
            An array of new cluster labels.
        """
        distances_to_medoids = self.distance_matrix_[:, self.cluster_center_indices_]
        return np.asarray(np.argmin(distances_to_medoids, axis=1))

    def _check_convergence(self, old_cluster_centers: NDArray[Any]) -> bool:
        """Check if the algorithm has converged.

        Convergence is declared if the medoids strictly stop changing
        between iterations.

        Args:
            old_cluster_centers: The medoids from the previous iteration.

        Returns:
            True if the model has converged, False otherwise.
        """
        return np.array_equal(self.cluster_centers_, old_cluster_centers)

    def _calc_inertia(self, X: NDArray[Any]) -> float:
        """Calculate the inertia of the cluster assignments.

        Inertia for K-Medoids is the sum of absolute distances of each sample
        to its closest medoid.

        Args:
            X: The original input data.

        Returns:
            The calculated inertia value.
        """
        distances_to_medoids = self.distance_matrix_[:, self.cluster_center_indices_]
        min_distances = np.min(distances_to_medoids, axis=1)
        return float(np.sum(min_distances))


def _convert_kmeans_cluster_centers(X: NDArray[Any], kmeans_cluster_centers: NDArray[Any]) -> NDArray[Any]:
    """Map continuous coordinates to the indices of the closest actual data points.

    This ensures that when a continuous initialization strategy (like standard K-Means)
    is used, the resulting centers are forced to be valid medoids (actual samples in X).

    Args:
        X: The original input data.
        kmeans_cluster_centers: The hypothetical cluster centers.

    Returns:
        An array of indices pointing to the samples in `X` closest to the input centers.
    """
    indices_with_centers = [_tools.find_closest_point(X, center) for center in kmeans_cluster_centers]
    indices, _ = zip(*indices_with_centers, strict=True)
    return np.array(indices)
