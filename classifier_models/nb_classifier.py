import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from .base_classifier import BaseClassifier


class NBClassifier(BaseClassifier):
    """Naive Bayes (Gaussian) binary classifier."""

    def __init__(self):
        super().__init__(name="NaiveBayes")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("nb", GaussianNB())
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
