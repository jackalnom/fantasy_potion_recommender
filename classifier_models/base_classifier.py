import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """Base class for all binary classifiers."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass
