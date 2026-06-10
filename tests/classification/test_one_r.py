import numpy as np
import pandas as pd
from numpy.typing import NDArray

from simple_sklearn.classification import OneRClassifier
from tests.classification.testing_utils import assert_valid_classification_predictions
from tests.testing_utils import assert_attributes_match_types


def test_one_r_fit_predict(dataset_4f: tuple[NDArray[np.int64], NDArray[np.str_]]) -> None:
    X, y = dataset_4f
    n_features = X.shape[1]
    unique_classes = set(y)
    n_classes = len(unique_classes)
    encoded_classes = set(range(n_classes))

    assert 3 not in X[:, 0], "Value 3 is present in the first feature of X"
    assert 3 not in X[:, 3], "Value 3 is present in the fourth feature of X"
    X_pred = np.array(
        [
            [2, 1, 1, 1],
            [0, 1, 0, 0],
            [3, 1, 1, 1],  # unknown value in the first feature
            [3, 1, 1, 3],  # unknown value in the fourth (best) feature
        ]
    )

    expected_attributes = {
        "classes_": NDArray[np.str_],
        "best_feature_index_": int,
        "prediction_rules_": pd.Series,
        "fallback_class_": str,
    }

    clf = OneRClassifier()
    clf.fit(X, y)

    # Validate expected learned attributes
    assert_attributes_match_types(clf, expected_attributes)

    # Validate classes_ attribute
    assert clf.classes_.shape[0] == n_classes
    assert set(clf.classes_) == unique_classes

    # Validate best_feature_index_ attribute
    assert 0 <= clf.best_feature_index_ < n_features
    # Expect best feature to be the fourth one for this specific dataset
    assert clf.best_feature_index_ == 3

    # Validate prediction_rules_ attribute
    # Check that the prediction rules map exclusively to valid encoded class indices
    assert set(clf.prediction_rules_.values) <= encoded_classes

    # Validate fallback_class_ attribute
    # The fallback class should be one of the unencoded class labels
    assert clf.fallback_class_ in unique_classes

    # Check predictions
    y_pred = clf.predict(X_pred)
    assert_valid_classification_predictions(X_pred, y_pred, y, clf.classes_)
    assert list(y_pred) == ["y0", "y1", "y0", "y1"]
