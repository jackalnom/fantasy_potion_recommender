import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from .base_recommender import BaseRecommender


class KNNRecommender(BaseRecommender):
    """K-Nearest Neighbors Regressor recommender."""

    DEFAULT_K = 5

    def __init__(self, n_neighbors: int = None):
        super().__init__(name="KNN")
        self.n_neighbors = n_neighbors if n_neighbors is not None else self.DEFAULT_K
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=self.n_neighbors))
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
