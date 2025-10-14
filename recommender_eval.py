import numpy as np
from recommender_models import (
    KNNRecommender,
    RFRecommender,
    GBRecommender,
    LRRecommender,
    RandomRecommender,
    CFRecommender,
    HybridRecommender,
    SurpriseRecommender,
    SURPRISE_AVAILABLE
)
from data_utils import RecommenderDataPrep, MODEL_FEATURE_COLS, CF_FEATURE_COLS, HYBRID_FEATURE_COLS

# Evaluation configuration
TOP_K = 3
SAMPLE_N = 5


def precision_at_k(recommended, relevant, k):
    """Calculate precision at k."""
    hits = sum(1 for pid in recommended[:k] if pid in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Calculate recall at k."""
    hits = sum(1 for pid in recommended[:k] if pid in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    """Calculate average precision at k."""
    ap, hits = 0.0, 0
    for i, pid in enumerate(recommended[:k], start=1):
        if pid in relevant:
            hits += 1
            ap += hits / i
    return ap / min(len(relevant), k)


def evaluate_rankings(rec_lists, relevant_sets, k):
    """Evaluate recommendation rankings across all users."""
    p_list, r_list, ap_list = [], [], []
    for aid, recs in rec_lists.items():
        rel = relevant_sets[aid]
        p_list.append(precision_at_k(recs, rel, k))
        r_list.append(recall_at_k(recs, rel, k))
        ap_list.append(average_precision_at_k(recs, rel, k))
    return (
        np.mean(p_list),
        np.mean(r_list),
        np.mean(ap_list),
    )


def print_results(results, data_prep, eval_users):
    """Print evaluation results in a formatted table."""
    pos_rate = data_prep.get_positive_rate()
    user_count = data_prep.get_user_count()

    print("\n=== Recommender Comparison (TOP_K = {}) ===".format(TOP_K))
    print(f"Users total: {user_count} | Evaluated (with ≥1 positive): {len(eval_users)}")
    print(f"Global positive rate (liked): {pos_rate:.3f}\n")

    for name, metrics in results.items():
        print(f"{name} recommender:")
        print(f"  Precision@K: {metrics['precision']:.4f}")
        print(f"  Recall@K:    {metrics['recall']:.4f}")
        print(f"  MAP@K:       {metrics['map']:.4f}")
        print()


def print_sample_recommendations(recommendations, data_prep, eval_users):
    """Print sample recommendations for first N users."""
    def fmt_rgb(pid):
        r, g, b = data_prep.potion_features.loc[pid][["red", "green", "blue"]]
        return f"{int(pid)}(RGB:{int(r)},{int(g)},{int(b)})"

    print("\n--- Sample top-{} recommendations (first {} evaluated adventurers) ---".format(TOP_K, SAMPLE_N))
    sample_advs = eval_users[:SAMPLE_N]

    for aid in sample_advs:
        rel = sorted(data_prep.relevant_by_adv.get(aid, set()))
        adv_class, adv_level = data_prep.adv_info.loc[aid][["class", "level"]]

        print(f"\nAdventurer {aid} ({adv_class}, Level {adv_level}):")
        if rel:
            held_list = [fmt_rgb(pid) for pid in rel if pid in data_prep.potion_features.index]
            print(f"  Held-out positives in test: {held_list}")
        else:
            print("  [WARN] No held-out positives found in candidates (unexpected).")

        for name in recommendations.keys():
            top_recs = recommendations[name].get(aid, [])[:TOP_K]
            hits = [pid for pid in top_recs if pid in rel]
            formatted_recs = [fmt_rgb(pid) for pid in top_recs if pid in data_prep.potion_features.index]
            print(f"  {name:20s} top-{TOP_K}: {formatted_recs} | hits: {len(hits)}")

        cand = data_prep.candidates_by_adv.get(aid, [])
        print(f"  Candidate set size: {len(cand)}")


def main():
    # Prepare data
    print("Loading and preparing data...")
    data_prep = RecommenderDataPrep()
    data_prep.load_and_prepare()

    # Initialize models with default hyperparameters
    # Content-based models (use features)
    content_models = {
        "KNN": (KNNRecommender(), MODEL_FEATURE_COLS),
        "RandomForest": (RFRecommender(), MODEL_FEATURE_COLS),
        "GradientBoosting": (GBRecommender(), MODEL_FEATURE_COLS),
        "LinearRegression": (LRRecommender(), MODEL_FEATURE_COLS),
        "Random": (RandomRecommender(), MODEL_FEATURE_COLS)
    }

    # Collaborative filtering models (only use IDs)
    cf_models = {
        "CollaborativeFiltering": (CFRecommender(), CF_FEATURE_COLS)
    }

    if SURPRISE_AVAILABLE:
        cf_models["Surprise"] = (SurpriseRecommender(), CF_FEATURE_COLS)

    # Hybrid model (uses both CF and content features)
    hybrid_models = {
        "Hybrid": (HybridRecommender(), HYBRID_FEATURE_COLS)
    }

    # Combine all models
    all_models = {**content_models, **cf_models, **hybrid_models}

    # Train all models
    for name, (model, feature_cols) in all_models.items():
        print(f"Training {name}...")
        X_train, y_train = data_prep.get_training_data(feature_cols)
        model.fit(X_train, y_train)

    # Generate recommendations
    recommendations = {}
    for name, (model, feature_cols) in all_models.items():
        print(f"Generating recommendations for {name}...")
        recommendations[name] = {}
        for aid in data_prep.eval_users:
            X_cand, cand_ids = data_prep.create_unseen_interactions(aid, feature_cols)
            recommendations[name][aid] = model.recommend(X_cand, cand_ids)

    # Evaluate all models
    results = {}
    for name in all_models.keys():
        p, r, map_score = evaluate_rankings(recommendations[name], data_prep.relevant_by_adv, TOP_K)
        results[name] = {"precision": p, "recall": r, "map": map_score}

    # Print results
    print_results(results, data_prep, data_prep.eval_users)
    print_sample_recommendations(recommendations, data_prep, data_prep.eval_users)


if __name__ == "__main__":
    main()
