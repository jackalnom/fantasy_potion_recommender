import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from .base_recommender import BaseRecommender


class LRRecommender(BaseRecommender):
    """Linear Regression recommender."""

    def __init__(self):
        super().__init__(name="LinearRegression")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
