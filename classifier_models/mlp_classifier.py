import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
from .base_classifier import BaseClassifier


class MLPClassifier(BaseClassifier):
    """Multi-layer Perceptron (neural network) binary classifier."""

    DEFAULT_HIDDEN_LAYERS = (100, 50)
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_MAX_ITER = 500

    def __init__(self, hidden_layer_sizes: tuple = None, random_state: int = None, max_iter: int = None):
        super().__init__(name="MLP")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", SklearnMLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes if hidden_layer_sizes is not None else self.DEFAULT_HIDDEN_LAYERS,
                random_state=random_state if random_state is not None else self.DEFAULT_RANDOM_STATE,
                max_iter=max_iter if max_iter is not None else self.DEFAULT_MAX_ITER
            ))
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
