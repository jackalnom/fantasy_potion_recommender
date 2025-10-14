import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .base_recommender import BaseRecommender


class GBRecommender(BaseRecommender):
    """Gradient Boosting Regressor recommender."""

    DEFAULT_N_ESTIMATORS = 100
    DEFAULT_MAX_DEPTH = 5
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_RANDOM_STATE = 42

    def __init__(self, n_estimators: int = None, max_depth: int = None,
                 learning_rate: float = None, random_state: int = None):
        super().__init__(name="GradientBoosting")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators if n_estimators is not None else self.DEFAULT_N_ESTIMATORS,
            max_depth=max_depth if max_depth is not None else self.DEFAULT_MAX_DEPTH,
            learning_rate=learning_rate if learning_rate is not None else self.DEFAULT_LEARNING_RATE,
            random_state=random_state if random_state is not None else self.DEFAULT_RANDOM_STATE
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
