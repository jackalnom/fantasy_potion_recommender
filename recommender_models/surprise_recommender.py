import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from .base_recommender import BaseRecommender


class SurpriseRecommender(BaseRecommender):
    """Collaborative Filtering recommender using Surprise library's SVD.

    Algorithm:
    1. Use Surprise's biased SVD implementation
    2. Learns user/item biases + latent factors via SGD optimization
    3. Incorporates regularization to prevent overfitting
    4. Predict: global_mean + user_bias + item_bias + (user_factors · item_factors)

    Differs from custom CF by:
    - Uses biases instead of mean-centering
    - Optimizes via SGD instead of direct SVD decomposition
    - Has built-in regularization (L2 penalty)

    """

    DEFAULT_N_FACTORS = 50
    DEFAULT_N_EPOCHS = 50
    DEFAULT_LR_ALL = 0.01
    DEFAULT_REG_ALL = 0.005
    DEFAULT_RANDOM_STATE = 42

    def __init__(self, n_factors: int = None, n_epochs: int = None,
                 lr_all: float = None, reg_all: float = None, random_state: int = None):
        super().__init__(name="Surprise")
        self.n_factors = n_factors if n_factors is not None else self.DEFAULT_N_FACTORS
        self.n_epochs = n_epochs if n_epochs is not None else self.DEFAULT_N_EPOCHS
        self.lr_all = lr_all if lr_all is not None else self.DEFAULT_LR_ALL
        self.reg_all = reg_all if reg_all is not None else self.DEFAULT_REG_ALL
        self.random_state = random_state if random_state is not None else self.DEFAULT_RANDOM_STATE
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )
        self.user_id_map = {}
        self.item_id_map = {}
        self.global_mean = 0.5

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        adv_ids = X_train['adv_id'].values
        potion_ids = X_train['potion_id'].values
        enjoyments = y_train.values

        unique_users = np.unique(adv_ids)
        unique_items = np.unique(potion_ids)

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.global_mean = np.mean(enjoyments)

        # Convert to Surprise format
        ratings_df = pd.DataFrame({
            'userID': adv_ids,
            'itemID': potion_ids,
            'rating': enjoyments
        })

        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(ratings_df[['userID', 'itemID', 'rating']], reader)
        trainset = data.build_full_trainset()

        # Train via SGD optimization
        self.model.fit(trainset)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []

        for _, row in X.iterrows():
            user_id = row['adv_id']
            item_id = row['potion_id']
            pred = self.model.predict(user_id, item_id, verbose=False)
            predictions.append(np.clip(pred.est, 0.0, 1.0))

        return np.array(predictions)
