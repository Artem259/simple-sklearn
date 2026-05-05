"""Unsupervised machine learning algorithms for clustering tasks.

This subpackage contains implementations of various clustering models.
All estimators in this module inherit from
[`sklearn.base.BaseEstimator`][] and [`sklearn.base.ClusterMixin`][].

Available models:
 - [`KMeans`][simple_sklearn.clustering.KMeans]:
    K-Means clustering.
 - [`KMedoids`][simple_sklearn.clustering.KMedoids]:
    K-Medoids clustering.
 - [`DBSCAN`][simple_sklearn.clustering.DBSCAN]:
    DBSCAN clustering.
 - [`AgglomerativeClustering`][simple_sklearn.clustering.AgglomerativeClustering]:
    Hierarchical agglomerative clustering.
"""

from ._agglomerative import AgglomerativeClustering
from ._dbscan import DBSCAN
from ._k_means import KMeans
from ._k_medoids import KMedoids

__all__ = [
    "DBSCAN",
    "AgglomerativeClustering",
    "KMeans",
    "KMedoids",
]
