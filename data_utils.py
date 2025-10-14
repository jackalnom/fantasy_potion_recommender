import pandas as pd
import numpy as np

# Data configuration
CSV_PATH = "interactions.csv"
FEATURE_COLS_RAW = ["adv_id", "potion_id", "avg_phys", "avg_magic", "red", "green", "blue"]
MODEL_FEATURE_COLS = ["avg_phys", "avg_magic", "red", "green", "blue"]
CF_FEATURE_COLS = ["adv_id", "potion_id"]  # For collaborative filtering models
HYBRID_FEATURE_COLS = ["adv_id", "potion_id", "avg_phys", "avg_magic", "red", "green", "blue"]  # For hybrid models
TARGET_COL = "enjoyment"
LIKE_THRESH = 0.5
HOLDOUT_POS_PER_USER = 3
RANDOM_SEED = 42


class RecommenderDataPrep:
    """Utility class for preparing recommender system data."""

    def __init__(self, csv_path: str = CSV_PATH, feature_cols: list = None,
                 target_col: str = TARGET_COL, like_thresh: float = LIKE_THRESH,
                 holdout_per_user: int = HOLDOUT_POS_PER_USER, random_seed: int = RANDOM_SEED):
        self.csv_path = csv_path
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS_RAW
        self.target_col = target_col
        self.like_thresh = like_thresh
        self.holdout_per_user = holdout_per_user
        self.random_seed = random_seed

        self.df = None
        self.train_df = None
        self.eval_users = None
        self.candidates_by_adv = None
        self.relevant_by_adv = None
        self.adv_features = None
        self.potion_features = None
        self.adv_info = None

    def load_and_prepare(self):
        """Load data and prepare train/test splits."""
        # Load data
        self.df = pd.read_csv(
            self.csv_path,
            usecols=self.feature_cols + [self.target_col, "class", "level"]
        )
        self.df["liked"] = (self.df[self.target_col] > self.like_thresh).astype(int)

        # Per-user holdout of positives
        train_rows = []
        heldout_positives = {}

        for aid, g in self.df.groupby("adv_id"):
            likes = g[g["liked"] == 1]
            if len(likes) >= 1:
                n_hold = min(self.holdout_per_user, len(likes))
                test_likes = likes.sample(n=n_hold, random_state=self.random_seed)
                heldout_positives[aid] = set(test_likes["potion_id"].tolist())
                train_rows.append(g.drop(test_likes.index))
            else:
                train_rows.append(g)

        self.train_df = pd.concat(train_rows, ignore_index=True)

        # Candidate sets
        self.eval_users = sorted(heldout_positives.keys())
        all_potions = sorted(self.df["potion_id"].unique().tolist())
        seen_train_by_adv = {aid: set(g["potion_id"].tolist())
                            for aid, g in self.train_df.groupby("adv_id")}
        self.candidates_by_adv = {
            aid: [pid for pid in all_potions if pid not in seen_train_by_adv.get(aid, set())]
            for aid in self.eval_users
        }

        self.relevant_by_adv = {
            aid: (heldout_positives[aid] & set(self.candidates_by_adv.get(aid, [])))
            for aid in self.eval_users
        }

        # Feature lookups
        self.adv_features = self.train_df.groupby("adv_id")[["avg_phys", "avg_magic"]].first()
        self.potion_features = self.df.groupby("potion_id")[["red", "green", "blue"]].first()
        self.adv_info = self.df.groupby("adv_id")[["class", "level"]].first()

    def get_training_data(self, model_feature_cols: list):
        """Get training features and target."""
        X_train = self.train_df[model_feature_cols].astype(float)
        y_train = self.train_df[self.target_col].astype(float)
        return X_train, y_train

    def _create_cf_interactions(self, aid, candidates):
        """Create feature matrix for collaborative filtering (IDs only)."""
        n = len(candidates)
        X_cand = pd.DataFrame({
            "adv_id": [aid] * n,
            "potion_id": candidates
        })
        return X_cand, candidates

    def _create_content_interactions(self, candidates, adv_phys, adv_magic):
        """Create feature matrix for content-based models (features only)."""
        pot_feats = self.potion_features.loc[candidates][["red", "green", "blue"]].values.astype(float)
        n = len(candidates)

        X_cand = pd.DataFrame(
            np.column_stack([
                np.full(n, float(adv_phys)),
                np.full(n, float(adv_magic)),
                pot_feats
            ]),
            columns=["avg_phys", "avg_magic", "red", "green", "blue"]
        )
        return X_cand, candidates

    def _create_hybrid_interactions(self, aid, candidates, adv_phys, adv_magic):
        """Create feature matrix for hybrid models (IDs + features)."""
        pot_feats = self.potion_features.loc[candidates][["red", "green", "blue"]].values.astype(float)
        n = len(candidates)

        X_cand = pd.DataFrame(
            np.column_stack([
                np.full(n, float(aid)),
                np.array(candidates, dtype=float),
                np.full(n, float(adv_phys)),
                np.full(n, float(adv_magic)),
                pot_feats
            ]),
            columns=["adv_id", "potion_id", "avg_phys", "avg_magic", "red", "green", "blue"]
        )
        return X_cand, candidates

    def create_unseen_interactions(self, aid, model_feature_cols: list):
        """Create feature matrix for unseen interactions for a given adventurer."""
        candidates = self.candidates_by_adv[aid]

        # Check if this is a CF model (only needs adv_id and potion_id)
        if set(model_feature_cols) == {"adv_id", "potion_id"}:
            return self._create_cf_interactions(aid, candidates)

        # For hybrid or content-based models: need features
        adv_phys, adv_magic = self.adv_features.loc[aid][["avg_phys", "avg_magic"]]

        # Check if hybrid model (includes IDs + features)
        if "adv_id" in model_feature_cols and "potion_id" in model_feature_cols:
            return self._create_hybrid_interactions(aid, candidates, adv_phys, adv_magic)
        else:
            # Content-based only
            return self._create_content_interactions(candidates, adv_phys, adv_magic)

    def get_positive_rate(self):
        """Calculate global positive rate."""
        return self.df["liked"].mean()

    def get_user_count(self):
        """Get total number of unique users."""
        return self.df["adv_id"].nunique()
