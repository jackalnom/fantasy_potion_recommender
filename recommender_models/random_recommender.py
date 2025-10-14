import pandas as pd
import numpy as np
from .base_recommender import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Random baseline recommender."""

    DEFAULT_RANDOM_STATE = 165  # Different from other models to ensure true randomness

    def __init__(self, random_state: int = None):
        super().__init__(name="Random")
        self.random_state = random_state if random_state is not None else self.DEFAULT_RANDOM_STATE
        self.rng = np.random.RandomState(self.random_state)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.rng.rand(len(X))
