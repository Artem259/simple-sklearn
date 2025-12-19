import numpy as np
import numpy.testing as npt
from sklearn.neighbors import KNeighborsClassifier as SKKNN
from sklearn.utils.estimator_checks import estimator_checks_generator

from src.simple_sklearn.classification.KNeighborsClassifier import KNeighborsClassifier


def test_kneighbors_predict_and_kneighbors_match_sklearn():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [10, 10],
        [10, 11],
    ])
    y = np.array([0, 0, 0, 1, 1])

    X_pred = np.array([[0, 0.5]])

    # custom classifier: uniform weights
    clf = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    clf.fit(X, y)
    y_pred = clf.predict(X_pred)

    # compare to sklearn
    sk = SKKNN(n_neighbors=3, weights='uniform')
    sk.fit(X, y)
    sk_y_pred = sk.predict(X_pred)

    assert (y_pred == sk_y_pred).all()

    # compare kneighbors outputs (distances and indices)
    dist, ind = clf.kneighbors(X_pred)
    sk_dist, sk_ind = sk.kneighbors(X_pred)

    npt.assert_allclose(dist, sk_dist, rtol=1e-6, atol=1e-8)
    assert (ind == sk_ind).all()


def test_kneighbors_passes_sklearn_checks():
    classifier = KNeighborsClassifier()
    for (estimator, check) in estimator_checks_generator(classifier):
        check(estimator)
