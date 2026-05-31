import numpy as np
import numpy.testing as npt
import sklearn

from simple_sklearn.classification import KNeighborsClassifier


def test_kneighbors_matches_sklearn_on_simple_data() -> None:
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [10, 10],
            [10, 11],
        ]
    )
    y = np.array([0, 0, 0, 1, 1])

    X_pred = np.array([[0, 0.5]])

    weights_tuple = (("uniform", True), ("distance", True), ("distance_squared", False))
    for weights, match_sklearn in weights_tuple:
        # custom classifier
        clf = KNeighborsClassifier(n_neighbors=3, weights=weights)
        clf.fit(X, y)
        y_pred = clf.predict(X_pred)

        if match_sklearn:
            # compare to sklearn
            sk = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, weights=weights)
            sk.fit(X, y)
            sk_y_pred = sk.predict(X_pred)

            assert (y_pred == sk_y_pred).all()

            # compare kneighbors outputs (distances and indices)
            dist, ind = clf.kneighbors(X_pred)
            sk_dist, sk_ind = sk.kneighbors(X_pred)

            npt.assert_allclose(dist, sk_dist, rtol=1e-6, atol=1e-8)
            assert (ind == sk_ind).all()
