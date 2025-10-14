import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_classifier import BaseClassifier


class RFClassifier(BaseClassifier):
    """Random Forest binary classifier."""

    DEFAULT_N_ESTIMATORS = 200
    DEFAULT_RANDOM_STATE = 42

    def __init__(self, n_estimators: int = None, random_state: int = None, n_jobs: int = -1):
        super().__init__(name="RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators if n_estimators is not None else self.DEFAULT_N_ESTIMATORS,
            random_state=random_state if random_state is not None else self.DEFAULT_RANDOM_STATE,
            n_jobs=n_jobs
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
