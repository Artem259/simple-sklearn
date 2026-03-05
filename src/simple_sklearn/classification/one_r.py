from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class OneRClassifier(ClassifierMixin, BaseEstimator):  # type: ignore
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: Any, y: Any) -> "OneRClassifier":
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
