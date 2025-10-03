import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

CSV_PATH = "interactions.csv"
FEATURE_COLS = ["adv_id", "potion_id", "avg_phys", "avg_magic", "red", "green", "blue"]
TARGET_COL = "enjoyment"
LIKE_THRESH = 0.5
RANDOM_SEED = 42

KNN_K = 5
TOP_K = 3                   # for Precision@K / Recall@K / MAP@K
SAMPLE_N = 5               # number of sample users to print in detail
HOLDOUT_POS_PER_USER = 3   # hold out up to this many positives per user

# -----------------------------
# Helpers: ranking metrics
# -----------------------------
def precision_at_k(recommended, relevant, k):
    if k == 0:
        return 0.0
    hits = sum(1 for pid in recommended[:k] if pid in relevant)
    return hits / k

def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for pid in recommended[:k] if pid in relevant)
    return hits / len(relevant)

def average_precision_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    ap, hits = 0.0, 0
    for i, pid in enumerate(recommended[:k], start=1):
        if pid in relevant:
            hits += 1
            ap += hits / i
    return ap / min(len(relevant), k)

def evaluate_rankings(rec_lists, relevant_sets, k):
    p_list, r_list, ap_list = [], [], []
    for aid, recs in rec_lists.items():
        rel = relevant_sets.get(aid, set())
        p_list.append(precision_at_k(recs, rel, k))
        r_list.append(recall_at_k(recs, rel, k))
        ap_list.append(average_precision_at_k(recs, rel, k))
    return (
        np.mean(p_list) if p_list else 0.0,
        np.mean(r_list) if r_list else 0.0,
        np.mean(ap_list) if ap_list else 0.0,
    )

# -----------------------------
# Data
# -----------------------------
# Read class/level for display; models use FEATURE_COLS only
df = pd.read_csv(CSV_PATH, usecols=FEATURE_COLS + [TARGET_COL, "class", "level"])
df["liked"] = (df[TARGET_COL] > LIKE_THRESH).astype(int)

# Multi-positive holdout split
train_rows, test_rows = [], []
heldout_positives = {}  # adv_id -> set of held-out positive potion_ids

for aid, g in df.groupby("adv_id"):
    likes = g[g["liked"] == 1]
    if len(likes) >= 1:
        n_hold = min(HOLDOUT_POS_PER_USER, len(likes))
        test_likes = likes.sample(n=n_hold, random_state=RANDOM_SEED)
        heldout_positives[aid] = set(test_likes["potion_id"].tolist())
        train_rows.append(g.drop(test_likes.index))
        test_rows.append(test_likes)
    else:
        # no positives -> keep all in train; user not evaluated
        train_rows.append(g)

train_df = pd.concat(train_rows, ignore_index=True)
test_df  = pd.concat(test_rows,  ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)

# Users to evaluate (have >=1 held-out positive)
eval_users = sorted(heldout_positives.keys())

# Universe of potions & per-user candidates (unseen in train)
all_potions = sorted(df["potion_id"].unique().tolist())
seen_train_by_adv = {aid: set(g["potion_id"].tolist()) for aid, g in train_df.groupby("adv_id")}
candidates_by_adv = {
    aid: [pid for pid in all_potions if pid not in seen_train_by_adv.get(aid, set())]
    for aid in eval_users
}

# Relevant items: the held-out positives, intersect with candidates (safety)
relevant_by_adv = {
    aid: (heldout_positives[aid] & set(candidates_by_adv.get(aid, [])))
    for aid in eval_users
}

# -----------------------------
# Models
# -----------------------------
# KNN regressor (scaled)
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=KNN_K))
])

# RandomForest regressor (no scaling needed)
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

X_train = train_df[FEATURE_COLS].astype(float)
y_train = train_df[TARGET_COL].astype(float)

knn_pipe.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Precompute lookups to avoid repeated scans
adv_features = df.groupby("adv_id")[["avg_phys", "avg_magic"]].first()
potion_features = df.groupby("potion_id")[["red", "green", "blue"]].first()
adv_info = df.groupby("adv_id")[["class", "level"]].first()

def build_candidate_matrix(aid, candidates):
    """Construct candidate feature matrix for a single adventurer over given potion IDs."""
    if not candidates or aid not in adv_features.index:
        return None, []
    adv_phys, adv_magic = adv_features.loc[aid][["avg_phys", "avg_magic"]]
    pot_feats = potion_features.loc[candidates].values  # Nx3
    n = len(candidates)
    X_cand = pd.DataFrame(
        np.column_stack([
            np.full(n, float(aid)),                 # adv_id
            np.array(candidates, dtype=float),      # potion_id
            np.full(n, float(adv_phys)),            # avg_phys
            np.full(n, float(adv_magic)),           # avg_magic
            pot_feats.astype(float)                 # red, green, blue
        ]),
        columns=FEATURE_COLS
    )
    return X_cand, candidates

def recommend_with_model(model, aid, candidates):
    X_cand, cand_ids = build_candidate_matrix(aid, candidates)
    if X_cand is None:
        return []
    scores = model.predict(X_cand)
    order = np.argsort(-scores)
    return [cand_ids[i] for i in order]

# Build recommendations for all evaluated users
rec_knn = {aid: recommend_with_model(knn_pipe, aid, candidates_by_adv.get(aid, [])) for aid in eval_users}
rec_rf  = {aid: recommend_with_model(rf_model,  aid, candidates_by_adv.get(aid, [])) for aid in eval_users}

# -----------------------------
# Random baseline recommender
# -----------------------------
def random_recs(cands_by_user, seed=123):
    rng = np.random.RandomState(seed)
    recs = {}
    for u, cands in cands_by_user.items():
        cands = list(cands)
        rng.shuffle(cands)
        recs[u] = cands
    return recs

rec_rand = random_recs(candidates_by_adv, seed=RANDOM_SEED + 123)

# -----------------------------
# Evaluation
# -----------------------------
p_knn, r_knn, map_knn = evaluate_rankings(rec_knn,   relevant_by_adv, TOP_K)
p_rf,  r_rf,  map_rf  = evaluate_rankings(rec_rf,    relevant_by_adv, TOP_K)
p_rnd, r_rnd, map_rnd = evaluate_rankings(rec_rand,  relevant_by_adv, TOP_K)

pos_rate = df["liked"].mean() if len(df) else 0.0
print("\n=== Recommender Comparison (Multi-Positive, TOP_K = {}) ===".format(TOP_K))
print(f"Users total: {df['adv_id'].nunique()} | Evaluated (with ≥1 positive): {len(eval_users)}")
print(f"Global positive rate (liked): {pos_rate:.3f}\n")

print("KNN Regressor recommender:")
print(f"  Precision@K: {p_knn:.4f}")
print(f"  Recall@K:    {r_knn:.4f}")
print(f"  MAP@K:       {map_knn:.4f}")

print("\nRandomForest Regressor recommender:")
print(f"  Precision@K: {p_rf:.4f}")
print(f"  Recall@K:    {r_rf:.4f}")
print(f"  MAP@K:       {map_rf:.4f}")

print("\nRandom baseline recommender:")
print(f"  Precision@K: {p_rnd:.4f}")
print(f"  Recall@K:    {r_rnd:.4f}")
print(f"  MAP@K:       {map_rnd:.4f}")

# -----------------------------
# Sample recommendations (evaluated users only)
# -----------------------------
def fmt_rgb(pid):
    r, g, b = potion_features.loc[pid][["red", "green", "blue"]]
    return f"{int(pid)}(RGB:{int(r)},{int(g)},{int(b)})"

print("\n--- Sample top-{} recommendations (first {} evaluated adventurers) ---".format(TOP_K, SAMPLE_N))
sample_advs = eval_users[:SAMPLE_N]

for aid in sample_advs:
    knn_top = rec_knn.get(aid, [])[:TOP_K]
    rf_top  = rec_rf.get(aid, [])[:TOP_K]
    rnd_top = rec_rand.get(aid, [])[:TOP_K]
    rel = sorted(relevant_by_adv.get(aid, set()))
    adv_class, adv_level = adv_info.loc[aid][["class", "level"]]

    print(f"\nAdventurer {aid} ({adv_class}, Level {adv_level}):")
    # Held-out positives (list all with RGB)
    if rel:
        held_list = [fmt_rgb(pid) for pid in rel]
        print(f"  Held-out positives in test: {held_list}")
    else:
        print("  [WARN] No held-out positives found in candidates (unexpected).")

    # Top-K per model (with RGB) + hits count
    knn_hits = [pid for pid in knn_top if pid in rel]
    rf_hits  = [pid for pid in rf_top  if pid in rel]
    rnd_hits = [pid for pid in rnd_top if pid in rel]
    print(f"  KNN top-{TOP_K}: {[fmt_rgb(pid) for pid in knn_top]}  | hits: {len(knn_hits)}")
    print(f"  RF  top-{TOP_K}: {[fmt_rgb(pid) for pid in rf_top]}   | hits: {len(rf_hits)}")
    print(f"  Rand top-{TOP_K}: {[fmt_rgb(pid) for pid in rnd_top]} | hits: {len(rnd_hits)}")

    # FULL candidate set (everything it was choosing from)
    cand = candidates_by_adv.get(aid, [])
    print(f"  Candidate set size: {len(cand)}")
