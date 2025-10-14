import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from sklearn.ensemble import GradientBoostingRegressor
from .base_recommender import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining collaborative filtering with content features."""

    DEFAULT_N_FACTORS = 60
    DEFAULT_N_EPOCHS = 70
    DEFAULT_LR_ALL = 0.01
    DEFAULT_REG_ALL = 0.003
    DEFAULT_CF_WEIGHT = 0.65
    DEFAULT_CONTENT_WEIGHT = 0.35
    DEFAULT_RANDOM_STATE = 42

    def __init__(self, n_factors: int = None, n_epochs: int = None,
                 lr_all: float = None, reg_all: float = None,
                 cf_weight: float = None, content_weight: float = None,
                 random_state: int = None):
        super().__init__(name="Hybrid")
        self.n_factors = n_factors if n_factors is not None else self.DEFAULT_N_FACTORS
        self.n_epochs = n_epochs if n_epochs is not None else self.DEFAULT_N_EPOCHS
        self.lr_all = lr_all if lr_all is not None else self.DEFAULT_LR_ALL
        self.reg_all = reg_all if reg_all is not None else self.DEFAULT_REG_ALL
        self.cf_weight = cf_weight if cf_weight is not None else self.DEFAULT_CF_WEIGHT
        self.content_weight = content_weight if content_weight is not None else self.DEFAULT_CONTENT_WEIGHT
        self.random_state = random_state if random_state is not None else self.DEFAULT_RANDOM_STATE

        self.cf_model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )
        self.user_id_map = {}
        self.item_id_map = {}
        self.global_mean = 0.5

        self.user_feature_cols = []
        self.item_feature_cols = []
        self.content_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        adv_ids = X_train['adv_id'].values
        potion_ids = X_train['potion_id'].values
        enjoyments = y_train.values

        unique_users = np.unique(adv_ids)
        unique_items = np.unique(potion_ids)

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.global_mean = np.mean(enjoyments)

        ratings_df = pd.DataFrame({
            'userID': adv_ids,
            'itemID': potion_ids,
            'rating': enjoyments
        })

        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(ratings_df[['userID', 'itemID', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.cf_model.fit(trainset)

        self.user_feature_cols = ['avg_phys', 'avg_magic']
        self.item_feature_cols = ['red', 'green', 'blue']

        content_features = X_train[self.user_feature_cols + self.item_feature_cols].values.astype(float)
        self.content_model.fit(content_features, enjoyments)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []

        content_features = X[self.user_feature_cols + self.item_feature_cols].values.astype(float)
        content_scores = self.content_model.predict(content_features)

        for idx, row in X.iterrows():
            user_id = row['adv_id']
            item_id = row['potion_id']

            content_score = content_scores[len(predictions)]

            if user_id in self.user_id_map and item_id in self.item_id_map:
                pred_cf = self.cf_model.predict(user_id, item_id, verbose=False)
                cf_score = np.clip(pred_cf.est, 0.0, 1.0)
                pred = self.cf_weight * cf_score + self.content_weight * content_score
            else:
                pred = content_score

            pred = np.clip(pred, 0.0, 1.0)
            predictions.append(pred)

        return np.array(predictions)
