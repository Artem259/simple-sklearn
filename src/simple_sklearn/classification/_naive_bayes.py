"""Naive Bayes Classification.

This module provides the `NaiveBayesClassifier` class.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from typing_extensions import Self


class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):  # type: ignore
    """Perform Categorical Naive Bayes classification.

    The categorical Naive Bayes classifier assumes that features are discrete.
    It evaluates the conditional probabilities of feature values given the target class.
    To handle zero-frequency issues during training, it applies Laplace (add-one) smoothing.
    Additionally, it calculates a fallback probability to manage unseen feature values
    encountered during prediction.

    Args:
        min_categories: The minimum number of categories per feature for Laplace smoothing.
            Can be an array-like of shape `(n_features,)`. If `None`, the number of
            categories is inferred from the training data.

    Attributes:
        classes_: The unique class labels observed in the training data.
        class_log_prior_: A pandas.Series containing the smoothed empirical log probability of each class.
        feature_unique_values_: A list of sets containing the unique values encountered for each feature.
        feature_log_prob_: A list of pandas.Series containing the empirical log probabilities
            of feature values given the class.
        feature_unknown_log_probs_: A list of pandas.Series containing the fallback log
            probabilities for unknown feature values given the class.
        num_features_: The number of features seen during fitting.
    """

    classes_: NDArray[Any]
    class_log_prior_: pd.Series
    feature_unique_values_: list[set[Any]]
    feature_log_prob_: list[pd.Series]
    feature_unknown_log_probs_: list[pd.Series]
    num_features_: int

    def __init__(self, min_categories: Any = None) -> None:
        super().__init__()
        self.min_categories = min_categories

    def fit(self, X: Any, y: Any) -> Self:
        """Fit the Naive Bayes classification model.

        Args:
            X: Training instances to fit the model. Can be an array-like of shape `(n_samples, n_features)`.
            y: Target values (class labels) for the training instances. Array-like of shape `(n_samples,)`.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If `y` is of a continuous type or if `min_categories` has an incompatible shape.
        """
        X, y = validate_data(self, X, y)
        X = np.array(X)
        self._validate_self_params(X)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        y_series = pd.Series(y)
        self.class_log_prior_ = cast(pd.Series, np.log(y_series.value_counts(normalize=True).sort_index()))

        min_categories = np.array(self.min_categories) if self.min_categories is not None else None
        self.feature_unique_values_ = []
        self.feature_log_prob_ = []
        self.feature_unknown_log_probs_ = []
        for feature_index, feature_values in enumerate(X.T):
            unique_values = set(np.unique(feature_values))
            feature_min_categories = min_categories[feature_index] if min_categories is not None else 1
            unique_num = max(len(unique_values), feature_min_categories)
            self.feature_unique_values_.append(unique_values)

            df = pd.DataFrame({"y": y, "feat_v": feature_values})
            df_grouped = df.groupby("y")["feat_v"]
            grouped_counts = df_grouped.value_counts()

            def _calc_log_probs(x: pd.Series, u: int = unique_num) -> pd.Series:
                return cast(pd.Series, np.log((x + 1) / (x.sum() + u)))

            log_probs = grouped_counts.groupby(level=0).apply(_calc_log_probs).reset_index(level=1, drop=True)
            self.feature_log_prob_.append(log_probs)

            def _calc_unknown(x: pd.Series, u: int = unique_num) -> float:
                return float(np.log(1 / (len(x) + u)))

            unknown_log_probs = df_grouped.apply(_calc_unknown)
            self.feature_unknown_log_probs_.append(unknown_log_probs)

        self.num_features_ = len(self.feature_unique_values_)
        return self

    def predict(self, X: Any) -> NDArray[Any]:
        """Predict class labels for the given input data.

        Args:
            X: Instances to predict. Can be an array-like of shape `(n_samples, n_features)`.

        Returns:
            An array of shape `(n_samples,)` containing the predicted class labels for each sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.array(X)

        decision_scores = self._decision_function(X)
        return np.asarray(self.classes_[np.argmax(decision_scores, axis=1)])

    def _decision_function(self, X: NDArray[Any]) -> NDArray[Any]:
        """Compute the unnormalized posterior log probability for each class.

        Args:
            X: Instances to evaluate. An array of shape `(n_samples, n_features)`.

        Returns:
            An array of shape `(n_samples, n_classes)` containing the decision scores
            (log probabilities) for each sample and class.
        """
        feature_probs_concat = pd.concat(self.feature_log_prob_, keys=range(self.num_features_))

        decision_scores = []
        for x in X:

            def _apply_insert(group: pd.Series, current_x: NDArray[Any] = x) -> float:
                return self._insert_missing_probs(group, current_x)

            x_feature_log_probs = feature_probs_concat.groupby(level=[0, "y"]).apply(_apply_insert)
            x_decision_scores = x_feature_log_probs.groupby("y").sum() + self.class_log_prior_
            decision_scores.append(x_decision_scores)

        return np.array(decision_scores)

    def _insert_missing_probs(self, group: pd.Series, x: NDArray[Any]) -> float:
        """Retrieve the log probability for a feature value or apply the unknown fallback.

        Checks if the observed feature value is present in the training data for the given
        feature and class group. If known, returns its precomputed log probability. If the
        value was not seen during training, returns the smoothed fallback log probability.

        Args:
            group: A pandas.Series representing the log probabilities for a specific feature and class.
            x: The single instance being evaluated.

        Returns:
            The log probability of the feature value given the class.
        """
        feature_index = group.index.get_level_values(0).tolist()[0]
        y_value = group.index.get_level_values("y").tolist()[0]
        known_values = group.index.get_level_values("feat_v").tolist()

        feature_value = x[feature_index]
        if feature_value in known_values:
            return float(group.loc[feature_index, y_value, feature_value])
        return float(self.feature_unknown_log_probs_[feature_index][y_value])

    def _validate_self_params(self, X: NDArray[Any]) -> None:
        """Validate the hyperparameters against the training data.

        Args:
            X: Training instances. Array of shape `(n_samples, n_features)`.

        Raises:
            ValueError: If `min_categories` is provided but its length does not match `n_features`.
        """
        if self.min_categories is not None:
            min_categories = np.array(self.min_categories)
            if not np.issubdtype(min_categories.dtype, np.signedinteger):
                raise ValueError(f"'min_categories' should have integral type. Got {min_categories.dtype} instead.")
            if min_categories.shape[0] != X.shape[1]:
                raise ValueError(
                    f"The 'min_categories' parameter must have shape (n_features,), got {min_categories.shape}"
                )
