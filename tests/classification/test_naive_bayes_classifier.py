import numpy as np
import numpy.testing as npt
from sklearn.naive_bayes import CategoricalNB
from sklearn.utils.estimator_checks import estimator_checks_generator

from simple_sklearn.classification.naive_bayes import NaiveBayesClassifier


def test_naive_bayes_matches_sklearn_categorical_nb():
    X = np.array([
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [2, 1],
    ])
    y = np.array([0, 0, 1, 1, 2])

    clf = NaiveBayesClassifier()
    clf.fit(X, y)

    X_pred = np.array([[0, 1], [1, 0]])
    y_pred = clf.predict(X_pred)

    # Check expected learned attributes
    assert hasattr(clf, "feature_log_prob_")
    assert hasattr(clf, "feature_unknown_log_probs_")

    # Check prediction shape and content
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(X_pred)

    # Compare against sklearn's CategoricalNB on the same data
    sk = CategoricalNB()
    sk.fit(X, y)
    sk_pred = sk.predict(np.array([[0, 1], [1, 0]]))
    assert (y_pred == sk_pred).all()
    npt.assert_allclose(clf.class_log_prior_, sk.class_log_prior_, rtol=1e-6, atol=1e-8)

    assert clf.class_log_prior_.shape[0] == len(np.unique(y))


def test_naive_bayes_handles_unknown_values():
    X = np.array([
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 2],
        [2, 1],
    ])
    y = np.array([0, 0, 1, 1, 2])
    min_categories = [5, 5]

    clf = NaiveBayesClassifier(min_categories=min_categories)
    clf.fit(X, y)

    # Test data includes unseen feature values (e.g. 3 and 4)
    X_pred = np.array([
        [3, 1],  # unseen in first feature
        [0, 4],  # unseen in second feature
        [1, 0],  # all seen
    ])
    y_pred = clf.predict(X_pred)

    # Check prediction shape and content
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(X_pred)

    # All predictions should be valid known classes
    for p in y_pred:
        assert p in np.unique(y)
        assert p in clf.classes_

    # Compare against sklearn's CategoricalNB on the same data
    sk = CategoricalNB(min_categories=min_categories)
    sk.fit(X, y)
    sk_pred = sk.predict(X_pred)
    assert (y_pred == sk_pred).all()
    npt.assert_allclose(clf.class_log_prior_, sk.class_log_prior_, rtol=1e-6, atol=1e-8)

    assert clf.class_log_prior_.shape[0] == len(np.unique(y))


def test_naive_bayes_passes_sklearn_checks():
    classifier = NaiveBayesClassifier()
    for (estimator, check) in estimator_checks_generator(classifier):
        check(estimator)
