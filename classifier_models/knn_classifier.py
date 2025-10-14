import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from .base_classifier import BaseClassifier


class KNNClassifier(BaseClassifier):
    """K-Nearest Neighbors binary classifier."""

    DEFAULT_K = 5

    def __init__(self, n_neighbors: int = None):
        super().__init__(name="KNN")
        self.n_neighbors = n_neighbors if n_neighbors is not None else self.DEFAULT_K
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=self.n_neighbors))
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
