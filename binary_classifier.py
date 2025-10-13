#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
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

FEATURE_COLS = ["avg_phys_squared", "avg_magic", "red", "green", "blue", 
                "random_0", "random_1", "random_2", "random_3", "random_4",
                "random_5", "random_6", "random_7", "random_8", "random_9",
                "avg_phys_squared", "phys_var1", "phys_var2"]

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

def main():
    df = pd.read_csv("interactions.csv", usecols=FEATURE_COLS + ["enjoyment"])

    X = df[FEATURE_COLS]
    y = (df["enjoyment"] > 0.5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize classifiers
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, random_state=42))
    ])

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500))
    ])

    nb = Pipeline([
        ("scaler", StandardScaler()),
        ("nb", GaussianNB())
    ])

    # Train all classifiers
    classifiers = {
        "KNN": knn,
        "RandomForest": rf,
        "SVM": svm,
        "MLP": mlp,
        "NaiveBayes": nb
    }

    predictions = {}
    probabilities = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        predictions[name] = clf.predict(X_test)
        probabilities[name] = clf.predict_proba(X_test)[:, 1]

    def report(name, y_true, y_pred, y_proba):
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

    print("\n=== Enjoyment > 0.5 Classifier Comparison ===")
    for name in classifiers.keys():
        report(name, y_test, predictions[name], probabilities[name])

    plot_roc_curves(y_test, probabilities, out_html="roc_curve.html")
    plot_pr_curves(y_test, probabilities, out_html="pr_curve.html")

    print("\nSaved interactive plots:")
    print(" - roc_curve.html")
    print(" - pr_curve.html")

if __name__ == "__main__":
    main()
