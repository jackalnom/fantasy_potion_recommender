#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import plotly.graph_objects as go

from classifier_models import (
    KNNClassifier,
    RFClassifier,
    SVMClassifier,
    MLPClassifier,
    NBClassifier
)

CSV_PATH = "interactions.csv"
FEATURE_COLS = ["avg_phys", "avg_magic", "red", "green", "blue",
                "random_0", "random_1", "random_2", "random_3", "random_4",
                "random_5", "random_6", "random_7", "random_8", "random_9",
                "phys_var1", "phys_var2"]
LIKE_THRESHOLD = 0.5
TEST_SIZE = 0.2
RANDOM_STATE = 42


def plot_roc_curves(y_true, scores_dict, out_html="roc_curve.html"):
    fig = go.Figure()
    for name, y_score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Chance", line=dict(dash="dash")))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend_title_text=None,
        width=900, height=550
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def plot_pr_curves(y_true, scores_dict, out_html="pr_curve.html"):
    fig = go.Figure()
    pos_rate = (sum(y_true) / len(y_true)) if len(y_true) else 0.0
    for name, y_score in scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                 name=f"{name} (AP={ap:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[pos_rate, pos_rate], mode="lines",
                             name=f"Baseline (pos rate={pos_rate:.3f})",
                             line=dict(dash="dash")))
    fig.update_layout(
        title="Precision–Recall Curve Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        legend_title_text=None,
        width=900, height=550
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def print_report(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH, usecols=FEATURE_COLS + ["enjoyment"])

    X = df[FEATURE_COLS]
    y = (df["enjoyment"] > LIKE_THRESHOLD).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "KNN": KNNClassifier(),
        "RandomForest": RFClassifier(),
        "SVM": SVMClassifier(),
        "MLP": MLPClassifier(),
        "NaiveBayes": NBClassifier()
    }

    predictions = {}
    probabilities = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)

    print(f"\n=== Binary Classifier Comparison (enjoyment > {LIKE_THRESHOLD}) ===")
    for name in models.keys():
        print_report(name, y_test, predictions[name], probabilities[name])

    plot_roc_curves(y_test, probabilities, out_html="roc_curve.html")
    plot_pr_curves(y_test, probabilities, out_html="pr_curve.html")

    print("\nSaved interactive plots:")
    print(" - roc_curve.html")
    print(" - pr_curve.html")


if __name__ == "__main__":
    main()
