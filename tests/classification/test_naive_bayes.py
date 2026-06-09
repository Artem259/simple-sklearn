from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import sklearn
from numpy.typing import NDArray

from simple_sklearn.classification import NaiveBayesClassifier
from tests.classification.testing_utils import (
    assert_matches_sklearn_predictions,
    assert_valid_classification_predictions,
)
from tests.testing_utils import (
    assert_attributes_match_types,
    assert_parameter_validation_exceptions,
)


@pytest.mark.parametrize(
    ("min_categories", "match_sklearn"),
    [
        (None, True),
        ([1, 1, 1, 1], True),
        ([7, 8, 9, 10], True),
    ],
)
def test_naive_bayes_matches_sklearn_on_simple_data(
    dataset_4f: tuple[NDArray[np.int_], NDArray[np.str_]], min_categories: Any, match_sklearn: bool
) -> None:
    X, y = dataset_4f
    n_features = X.shape[1]
    unique_classes = set(y)
    n_classes = len(unique_classes)
    encoded_classes = set(range(n_classes))

    assert 3 not in X[:, 0], "Value 3 is present in the first feature of X"
    assert 2 not in X[:, 1], "Value 2 is present in the second feature of X"
    X_pred = np.array(
        [
            [2, 1, 1, 1],
            [0, 1, 0, 0],
            [3, 1, 1, 1],  # unknown value in the first feature
            [0, 2, 1, 1],  # unknown value in the second feature
        ]
    )

    expected_attributes = {
        "classes_": NDArray[np.str_],
        "class_log_prior_": pd.Series,
        "feature_unique_values_": list[set[np.int_]],
        "feature_log_prob_": list[pd.Series],
        "feature_unknown_log_probs_": list[pd.Series],
        "num_features_": int,
    }

    clf = NaiveBayesClassifier(min_categories=min_categories)
    clf.fit(X, y)

    # Validate expected learned attributes
    assert_attributes_match_types(clf, expected_attributes)

    # Validate classes_ attribute
    assert clf.classes_.shape[0] == n_classes
    assert set(clf.classes_) == unique_classes

    # Validate class_log_prior_ attribute
    assert clf.class_log_prior_.shape[0] == n_classes

    # Validate feature_unique_values_ attribute
    assert len(clf.feature_unique_values_) == n_features
    for i, unique_vals in enumerate(clf.feature_unique_values_):
        assert unique_vals == set(X[:, i])

    # Validate feature_log_prob_ attribute
    assert len(clf.feature_log_prob_) == n_features
    # Check that the multi-index aligns with the classes
    for series in clf.feature_log_prob_:
        # Ensure the series index has 'y' level matching encoded class indices
        assert set(series.index.get_level_values("y").unique()) == encoded_classes

    # Validate feature_unknown_log_probs_ attribute
    assert len(clf.feature_unknown_log_probs_) == n_features
    for series in clf.feature_unknown_log_probs_:
        # Check that the fallback probabilities are mapped to each encoded class index
        assert set(series.index) == encoded_classes

    # Validate num_features_ attribute
    assert clf.num_features_ == n_features

    # Check predictions
    y_pred = clf.predict(X_pred)
    assert_valid_classification_predictions(X_pred, y_pred, y, clf.classes_)

    # Compare against sklearn's classifier on the same data
    if match_sklearn:
        sk = sklearn.naive_bayes.CategoricalNB(min_categories=min_categories)
        sk.fit(X, y)

        if min_categories is None or min_categories == [1, 1, 1, 1]:
            # Sklearn's CategoricalNB crashes on unknown categories during prediction unless
            # min_categories forces the category counts to be large enough to include the unknown index.
            with pytest.raises(IndexError):
                sk.predict(X_pred)
        else:
            # Compare predictions
            sk_y_pred = sk.predict(X_pred)
            assert_matches_sklearn_predictions(y_pred, sk_y_pred)

        # Compare class_log_prior_ attributes
        npt.assert_allclose(clf.class_log_prior_, sk.class_log_prior_, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    ("params", "expected_error", "match_text", "match_sklearn"),
    [
        # min_categories checks
        ({"min_categories": [5, 5, 5]}, ValueError, r"must have shape \(n_features,\), got \(3,\)", False),
        ({"min_categories": [1, 1, 1, 2.5]}, ValueError, r"should have integral type. Got float64", True),
        ({"min_categories": [1, 1, 1, "b"]}, ValueError, r"should have integral type. Got <U", True),
    ],
)
def test_naive_bayes_parameter_validation_exceptions(
    dataset_4f: tuple[NDArray[np.int_], NDArray[np.str_]],
    params: dict[str, Any],
    expected_error: type[Exception],
    match_text: str,
    match_sklearn: bool,
) -> None:
    X, y = dataset_4f

    clf = NaiveBayesClassifier(**params)
    sk = sklearn.naive_bayes.CategoricalNB(**params) if match_sklearn else None

    assert_parameter_validation_exceptions(clf, X, y, expected_error, match_text, sk)
