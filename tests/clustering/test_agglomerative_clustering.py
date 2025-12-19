import numpy as np
import numpy.testing as npt
from sklearn.cluster import AgglomerativeClustering as SKAgg
from sklearn.utils.estimator_checks import estimator_checks_generator

from src.simple_sklearn.clustering.AgglomerativeClustering import AgglomerativeClustering


def _small_dataset():
    rs = np.random.RandomState(0)
    X = np.vstack([
        rs.normal(loc=[0,0], scale=0.05, size=(4,2)),
        rs.normal(loc=[1,1], scale=0.05, size=(4,2)),
    ])
    return X

def test_agglomerative_distances_match_sklearn_for_linkages():
    X = _small_dataset()
    for linkage in ['single', 'complete', 'average', 'ward']:
        custom = AgglomerativeClustering(n_clusters=2, linkage=linkage)
        custom.fit(X)

        sk = SKAgg(n_clusters=2, linkage=linkage, compute_distances=True)
        sk.fit(X)

        npt.assert_allclose(custom.distances_, sk.distances_, rtol=1e-6, atol=1e-8)

        # children_ should be present and have correct shape: (n_samples-1, 2)
        assert custom.children_.shape[0] == X.shape[0] - 1
        assert custom.children_.shape[1] == 2


def test_agglomerative_passes_sklearn_checks():
    clusterer = AgglomerativeClustering()
    for (estimator, check) in estimator_checks_generator(clusterer):
        check(estimator)
