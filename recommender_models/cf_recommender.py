import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from .base_recommender import BaseRecommender


class CFRecommender(BaseRecommender):
    """Collaborative Filtering recommender using Matrix Factorization (SVD)."""

    DEFAULT_N_COMPONENTS = 15
    DEFAULT_RANDOM_STATE = 42

    def __init__(self, n_components: int = None, random_state: int = None):
        super().__init__(name="CollaborativeFiltering")
        self.n_components = n_components if n_components is not None else self.DEFAULT_N_COMPONENTS
        self.random_state = random_state if random_state is not None else self.DEFAULT_RANDOM_STATE
        self.user_factors = None
        self.item_factors = None
        self.singular_values = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.user_means = {}
        self.global_mean = 0.5

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        adv_ids = X_train['adv_id'].values
        potion_ids = X_train['potion_id'].values
        enjoyments = y_train.values

        unique_users = np.unique(adv_ids)
        unique_items = np.unique(potion_ids)

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        user_indices = np.array([self.user_id_map[uid] for uid in adv_ids])
        item_indices = np.array([self.item_id_map[iid] for iid in potion_ids])

        interaction_matrix = csr_matrix(
            (enjoyments, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )

        self.global_mean = np.mean(enjoyments) if len(enjoyments) > 0 else 0.5

        user_means_array = np.zeros(n_users)
        for user_idx in range(n_users):
            user_row = interaction_matrix.getrow(user_idx)
            user_data = user_row.data
            user_means_array[user_idx] = np.mean(user_data) if len(user_data) > 0 else self.global_mean
            self.user_means[user_idx] = user_means_array[user_idx]

        mean_centered_matrix = interaction_matrix.copy().tolil()
        for user_idx in range(n_users):
            user_row = interaction_matrix.getrow(user_idx)
            if user_row.nnz > 0:
                for col_idx in user_row.indices:
                    mean_centered_matrix[user_idx, col_idx] -= self.user_means[user_idx]
        mean_centered_matrix = mean_centered_matrix.tocsr()

        # Fit SVD model
        k = min(self.n_components, min(n_users, n_items) - 1)

        try:
            U, sigma, Vt = svds(mean_centered_matrix, k=k, random_state=self.random_state)
            self.user_factors = U
            self.singular_values = sigma
            self.item_factors = Vt.T
        except Exception as e:
            print(f"Warning: SVD failed ({e}), CF will use user means fallback")
            self.user_factors = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []

        # Fallback if SVD failed during training
        if self.user_factors is None:
            for _, row in X.iterrows():
                user_id = row['adv_id']
                user_idx = self.user_id_map[user_id]
                predictions.append(self.user_means[user_idx])
            return np.array(predictions)

        # Normal prediction using SVD factors
        for _, row in X.iterrows():
            user_id = row['adv_id']
            item_id = row['potion_id']

            user_idx = self.user_id_map[user_id]
            item_idx = self.item_id_map[item_id]

            # Predict: user_mean + (user_factors × singular_values × item_factors)
            pred_deviation = np.dot(
                self.user_factors[user_idx] * self.singular_values,
                self.item_factors[item_idx]
            )
            pred = self.user_means[user_idx] + pred_deviation

            # Clip to valid range [0, 1]
            pred = np.clip(pred, 0.0, 1.0)
            predictions.append(pred)

        return np.array(predictions)
