import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """Base class for all recommender models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the recommender model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict scores for candidate items."""
        pass

    def recommend(self, X_candidates: pd.DataFrame, candidate_ids: list) -> list:
        """
        Generate ranked recommendations.

        Args:
            X_candidates: Feature matrix for candidate items
            candidate_ids: List of candidate item IDs

        Returns:
            List of item IDs ranked by predicted score (highest first)
        """
        if X_candidates is None or len(candidate_ids) == 0:
            return []

        scores = self.predict(X_candidates)
        order = np.argsort(-scores)
        return [candidate_ids[i] for i in order]
