import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

CSV_PATH = "interactions.csv"

FEATURE_COLS_RAW = ["adv_id", "potion_id", "avg_phys", "avg_magic", "red", "green", "blue"]
TARGET_COL = "enjoyment"
LIKE_THRESH = 0.5
RANDOM_SEED = 42

KNN_K = 5
TOP_K = 3
SAMPLE_N = 5
HOLDOUT_POS_PER_USER = 3

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

# ---- Load data
df = pd.read_csv(CSV_PATH, usecols=FEATURE_COLS_RAW + [TARGET_COL, "class", "level"])
df["liked"] = (df[TARGET_COL] > LIKE_THRESH).astype(int)

# ---- Per-user holdout of positives
train_rows, test_rows = [], []
heldout_positives = {}

for aid, g in df.groupby("adv_id"):
    likes = g[g["liked"] == 1]
    if len(likes) >= 1:
        n_hold = min(HOLDOUT_POS_PER_USER, len(likes))
        test_likes = likes.sample(n=n_hold, random_state=RANDOM_SEED)
        heldout_positives[aid] = set(test_likes["potion_id"].tolist())
        train_rows.append(g.drop(test_likes.index))
        test_rows.append(test_likes)
    else:
        train_rows.append(g)

train_df = pd.concat(train_rows, ignore_index=True)
test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)

# ---- Candidate sets (exclude anything seen in train for that user)
eval_users = sorted(heldout_positives.keys())
all_potions = sorted(df["potion_id"].unique().tolist())
seen_train_by_adv = {aid: set(g["potion_id"].tolist()) for aid, g in train_df.groupby("adv_id")}
candidates_by_adv = {
    aid: [pid for pid in all_potions if pid not in seen_train_by_adv.get(aid, set())]
    for aid in eval_users
}

relevant_by_adv = {
    aid: (heldout_positives[aid] & set(candidates_by_adv.get(aid, [])))
    for aid in eval_users
}

adv_features = (
    train_df.groupby("adv_id")[["avg_phys", "avg_magic"]]
            .mean()
)

potion_features = df.groupby("potion_id")[["red", "green", "blue"]].first()

adv_info = df.groupby("adv_id")[["class", "level"]].first()

MODEL_FEATURE_COLS = ["avg_phys", "avg_magic", "red", "green", "blue"]

# ---- Models
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=KNN_K))
])
rf_model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED, max_depth=5, learning_rate=0.1)

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

X_train = train_df[MODEL_FEATURE_COLS].astype(float)
y_train = train_df[TARGET_COL].astype(float)
knn_pipe.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lr_pipe.fit(X_train, y_train)

def build_candidate_matrix(aid, candidates):
    if not candidates or aid not in adv_features.index:
        return None, []
    adv_phys, adv_magic = adv_features.loc[aid][["avg_phys", "avg_magic"]]
    
    valid_cands = [pid for pid in candidates if pid in potion_features.index]
    if not valid_cands:
        return None, []
    pot_feats = potion_features.loc[valid_cands][["red", "green", "blue"]].values.astype(float)
    n = len(valid_cands)
    X_cand = pd.DataFrame(
        np.column_stack([
            np.full(n, float(adv_phys)),
            np.full(n, float(adv_magic)),
            pot_feats
        ]),
        columns=MODEL_FEATURE_COLS
    )
    return X_cand, valid_cands

def recommend_with_model(model, aid, candidates):
    X_cand, cand_ids = build_candidate_matrix(aid, candidates)
    if X_cand is None:
        return []
    scores = model.predict(X_cand)
    order = np.argsort(-scores)
    return [cand_ids[i] for i in order]


rec_knn = {aid: recommend_with_model(knn_pipe, aid, candidates_by_adv.get(aid, [])) for aid in eval_users}
rec_rf  = {aid: recommend_with_model(rf_model,  aid, candidates_by_adv.get(aid, [])) for aid in eval_users}
rec_gb  = {aid: recommend_with_model(gb_model,  aid, candidates_by_adv.get(aid, [])) for aid in eval_users}
rec_lr  = {aid: recommend_with_model(lr_pipe,   aid, candidates_by_adv.get(aid, [])) for aid in eval_users}

def random_recs(cands_by_user, seed=123):
    rng = np.random.RandomState(seed)
    recs = {}
    for u, cands in cands_by_user.items():
        cands = list(cands)
        rng.shuffle(cands)
        recs[u] = cands
    return recs

rec_rand = random_recs(candidates_by_adv, seed=RANDOM_SEED + 123)


p_knn, r_knn, map_knn = evaluate_rankings(rec_knn, relevant_by_adv, TOP_K)
p_rf,  r_rf,  map_rf  = evaluate_rankings(rec_rf,  relevant_by_adv, TOP_K)
p_gb,  r_gb,  map_gb  = evaluate_rankings(rec_gb,  relevant_by_adv, TOP_K)
p_lr,  r_lr,  map_lr  = evaluate_rankings(rec_lr,  relevant_by_adv, TOP_K)
p_rnd, r_rnd, map_rnd = evaluate_rankings(rec_rand, relevant_by_adv, TOP_K)

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

print("\nGradientBoosting Regressor recommender:")
print(f"  Precision@K: {p_gb:.4f}")
print(f"  Recall@K:    {r_gb:.4f}")
print(f"  MAP@K:       {map_gb:.4f}")

print("\nLinearRegression recommender:")
print(f"  Precision@K: {p_lr:.4f}")
print(f"  Recall@K:    {r_lr:.4f}")
print(f"  MAP@K:       {map_lr:.4f}")

print("\nRandom baseline recommender:")
print(f"  Precision@K: {p_rnd:.4f}")
print(f"  Recall@K:    {r_rnd:.4f}")
print(f"  MAP@K:       {map_rnd:.4f}")


def fmt_rgb(pid):
    r, g, b = potion_features.loc[pid][["red", "green", "blue"]]
    return f"{int(pid)}(RGB:{int(r)},{int(g)},{int(b)})"

print("\n--- Sample top-{} recommendations (first {} evaluated adventurers) ---".format(TOP_K, SAMPLE_N))
sample_advs = eval_users[:SAMPLE_N]

for aid in sample_advs:
    knn_top = rec_knn.get(aid, [])[:TOP_K]
    rf_top  = rec_rf.get(aid, [])[:TOP_K]
    gb_top  = rec_gb.get(aid, [])[:TOP_K]
    lr_top  = rec_lr.get(aid, [])[:TOP_K]
    rnd_top = rec_rand.get(aid, [])[:TOP_K]
    rel = sorted(relevant_by_adv.get(aid, set()))
    adv_class, adv_level = adv_info.loc[aid][["class", "level"]]

    print(f"\nAdventurer {aid} ({adv_class}, Level {adv_level}):")
    if rel:
        held_list = [fmt_rgb(pid) for pid in rel if pid in potion_features.index]
        print(f"  Held-out positives in test: {held_list}")
    else:
        print("  [WARN] No held-out positives found in candidates (unexpected).")

    knn_hits = [pid for pid in knn_top if pid in rel]
    rf_hits  = [pid for pid in rf_top if pid in rel]
    gb_hits  = [pid for pid in gb_top if pid in rel]
    lr_hits  = [pid for pid in lr_top if pid in rel]
    rnd_hits = [pid for pid in rnd_top if pid in rel]
    print(f"  KNN top-{TOP_K}: {[fmt_rgb(pid) for pid in knn_top if pid in potion_features.index]}  | hits: {len(knn_hits)}")
    print(f"  RF  top-{TOP_K}: {[fmt_rgb(pid) for pid in rf_top  if pid in potion_features.index]}   | hits: {len(rf_hits)}")
    print(f"  GB  top-{TOP_K}: {[fmt_rgb(pid) for pid in gb_top  if pid in potion_features.index]}   | hits: {len(gb_hits)}")
    print(f"  LR  top-{TOP_K}: {[fmt_rgb(pid) for pid in lr_top  if pid in potion_features.index]}   | hits: {len(lr_hits)}")
    print(f"  Rand top-{TOP_K}: {[fmt_rgb(pid) for pid in rnd_top if pid in potion_features.index]} | hits: {len(rnd_hits)}")

    cand = candidates_by_adv.get(aid, [])
    print(f"  Candidate set size: {len(cand)}")
