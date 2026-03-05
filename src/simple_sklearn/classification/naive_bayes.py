from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):  # type: ignore
    def __init__(self, min_categories: Any = None) -> None:
        super().__init__()
        self.min_categories = min_categories

    def fit(self, X: Any, y: Any) -> "NaiveBayesClassifier":
        self.__validate_params(X)
        X, y = validate_data(self, X, y)
        X = np.array(X)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        y_series = pd.Series(y)
        self.class_log_prior_ = np.log(y_series.value_counts(normalize=True).sort_index())

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

            def _calc_log_probs(x: pd.Series, u: int = unique_num) -> NDArray[Any]:
                return np.asarray(np.log((x + 1) / (sum(x) + u)))

            log_probs = grouped_counts.groupby(level=0).apply(_calc_log_probs).reset_index(level=1, drop=True)
            self.feature_log_prob_.append(log_probs)

            def _calc_unknown(x: pd.Series, u: int = unique_num) -> float:
                return float(np.log(1 / (len(x) + u)))

            unknown_log_probs = df_grouped.apply(_calc_unknown)
            self.feature_unknown_log_probs_.append(unknown_log_probs)

        self.num_features_ = len(self.feature_unique_values_)
        return self

    def predict(self, X: Any) -> NDArray[Any]:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.array(X)

        decision_scores = self._decision_function(X)
        return np.asarray(self.classes_[np.argmax(decision_scores, axis=1)])

    def _decision_function(self, X: NDArray[Any]) -> NDArray[Any]:
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
        feature_index = group.index.get_level_values(0).tolist()[0]
        y_value = group.index.get_level_values("y").tolist()[0]
        known_values = group.index.get_level_values("feat_v").tolist()

        feature_value = x[feature_index]
        if feature_value in known_values:
            return float(group.loc[feature_index, y_value, feature_value])
        return float(self.feature_unknown_log_probs_[feature_index][y_value])

    def __validate_params(self, X: Any) -> None:
        if self.min_categories is not None:
            min_categories = np.array(self.min_categories)
            if min_categories.shape[0] != X.shape[1]:
                raise ValueError(f"min_categories must have shape (n_features,), got {min_categories.shape}")
