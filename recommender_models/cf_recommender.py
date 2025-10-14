import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from .base_recommender import BaseRecommender


class CFRecommender(BaseRecommender):
    """Collaborative Filtering recommender using Matrix Factorization (SVD).

    Algorithm:
    1. Build sparse user-item interaction matrix from ratings
    2. Mean-center each user's ratings (subtract user's average)
    3. Apply truncated SVD to decompose into user/item factors
    4. Predict by reconstructing: user_mean + (user_factors × singular_values × item_factors)

    This captures latent patterns in user-item interactions without using content features.
    """

    DEFAULT_N_COMPONENTS = 3
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

        # Map user/item IDs to matrix indices
        unique_users = np.unique(adv_ids)
        unique_items = np.unique(potion_ids)

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        user_indices = np.array([self.user_id_map[uid] for uid in adv_ids])
        item_indices = np.array([self.item_id_map[iid] for iid in potion_ids])

        # Build sparse interaction matrix (users × items)
        interaction_matrix = csr_matrix(
            (enjoyments, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )

        self.global_mean = np.mean(enjoyments)

        # Calculate per-user means for centering
        for user_idx in range(n_users):
            user_row = interaction_matrix.getrow(user_idx)
            user_data = user_row.data
            self.user_means[user_idx] = np.mean(user_data)

        # Mean-center the matrix (subtract each user's mean from their ratings)
        mean_centered_matrix = interaction_matrix.copy().tolil()
        for user_idx in range(n_users):
            user_row = interaction_matrix.getrow(user_idx)
            if user_row.nnz > 0:
                for col_idx in user_row.indices:
                    mean_centered_matrix[user_idx, col_idx] -= self.user_means[user_idx]
        mean_centered_matrix = mean_centered_matrix.tocsr()

        # Apply truncated SVD to find latent factors
        k = min(self.n_components, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(mean_centered_matrix, k=k, random_state=self.random_state)
        self.user_factors = U
        self.singular_values = sigma
        self.item_factors = Vt.T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        user_ids = X['adv_id'].values
        item_ids = X['potion_id'].values

        user_indices = np.array([self.user_id_map[uid] for uid in user_ids])
        item_indices = np.array([self.item_id_map[iid] for iid in item_ids])
        user_means = np.array([self.user_means[idx] for idx in user_indices])

        # Reconstruct ratings: user_mean + (user_factors × singular_values × item_factors)
        user_factors_scaled = self.user_factors[user_indices] * self.singular_values
        item_factors_selected = self.item_factors[item_indices]
        pred_deviations = np.sum(user_factors_scaled * item_factors_selected, axis=1)

        predictions = user_means + pred_deviations
        predictions = np.clip(predictions, 0.0, 1.0)

        return predictions
