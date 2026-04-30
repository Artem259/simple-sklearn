"""OneR Classification.

This module provides the `OneRClassifier` class.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from typing_extensions import Self


class OneRClassifier(ClassifierMixin, BaseEstimator):  # type: ignore
    """Perform 1R (One Rule) classification.

    The 1R algorithm generates a simple, one-level decision tree. It evaluates
    each feature independently, creating a rule that predicts the majority target
    class for each unique value of that feature. It then selects the single
    feature with the lowest overall error rate to use for final predictions.

    Attributes:
        classes_: The unique class labels observed in the training data.
        best_feature_index_: The index of the feature selected with the lowest error rate.
        prediction_rules_: A pandas.Series mapping from the values of the best feature to the predicted class indices.
        fallback_class_: The majority class used as a fallback for unseen feature values during prediction.
    """

    classes_: NDArray[Any]
    best_feature_index_: int
    prediction_rules_: pd.Series
    fallback_class_: Any

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: Any, y: Any) -> Self:
        """Fit the OneR classification model.

        Args:
            X: Training instances to fit the model. Can be an array-like of shape `(n_samples, n_features)`.
            y: Target values (class labels) for the training instances. Array-like of shape `(n_samples,)`.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If `y` is of a continuous type.
        """
        X, y = validate_data(self, X, y)
        X = np.array(X)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        num_samples = X.shape[0]
        best_error_rate = 1.01
        for feature_index, feature_values in enumerate(X.T):
            df = pd.DataFrame({"feat_v": feature_values, "y": y})

            df_grouped = df.groupby("feat_v")["y"]
            prediction_rules = df_grouped.apply(lambda x: x.value_counts().idxmax())
            accuracy = float(df_grouped.apply(lambda x: x.value_counts().max()).sum()) / num_samples
            error_rate = 1 - accuracy

            if error_rate < best_error_rate:
                best_error_rate = error_rate
                self.best_feature_index_ = feature_index
                self.prediction_rules_ = prediction_rules

        self.fallback_class_ = self.classes_[np.bincount(y).argmax()]

        return self

    def predict(self, X: Any) -> NDArray[Any]:
        """Predict class labels for the given input data.

        If an unseen feature value is encountered, the model predicts the overall
        majority class from the training set.

        Args:
            X: Instances to predict. Can be an array-like of shape `(n_samples, n_features)`.

        Returns:
            An array of shape `(n_samples,)` containing the predicted class labels for each sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.array(X)

        preds = []
        for x in X:
            val = x[self.best_feature_index_]
            if val in self.prediction_rules_.index:
                y = self.prediction_rules_[val]
                preds.append(self.classes_[y])
            else:
                preds.append(self.fallback_class_)

        return np.array(preds)
