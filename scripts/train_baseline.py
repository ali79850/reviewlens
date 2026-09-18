"""
Phase 4: TF-IDF classical sentiment baseline.

Trains TF-IDF + LogisticRegression and TF-IDF + LinearSVC on the same split,
evaluates both on the held-out test set, and saves whichever wins on macro F1
(the fairer metric here given the Neutral class is ~7% of the data).

Run from project root: python scripts/train_baseline.py
"""
import json
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

sys.path.insert(0, "src")
from reviewlens.preprocessing.clean import clean_text
from reviewlens.sentiment.tfidf import TfidfSentimentModel
from reviewlens.utils.paths import load_config, resolve

LABELS = ["Negative", "Neutral", "Positive"]


def evaluate(y_true, y_pred, model_name: str) -> dict:
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def save_confusion_matrix(y_true, y_pred, model_name: str, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    processed_dir = resolve(cfg["paths"]["data_processed"])
    models_dir = resolve(cfg["paths"]["models"])
    metrics_dir = resolve(cfg["paths"]["results"] + "/metrics")
    figures_dir = resolve(cfg["paths"]["results"] + "/figures")

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    X_train_text = train_df["text"].apply(clean_text)
    X_test_text = test_df["text"].apply(clean_text)
    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    candidates = {
        "tfidf-logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=cfg["random_seed"]
        ),
        "tfidf-svm": LinearSVC(
            class_weight="balanced", random_state=cfg["random_seed"]
        ),
    }

    all_metrics = []
    fitted = {}
    predictions = {}

    for name, clf in candidates.items():
        print(f"\nFitting {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fitted[name] = clf
        predictions[name] = y_pred

        metrics = evaluate(y_test, y_pred, name)
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))

        report_text = classification_report(y_test, y_pred, labels=LABELS, zero_division=0)
        print(report_text)
        with open(metrics_dir / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        save_confusion_matrix(y_test, y_pred, name, figures_dir / f"{name}_confusion_matrix.png")

    # Pick the winner on macro F1 — the fairer metric given class imbalance
    winner_metrics = max(all_metrics, key=lambda m: m["f1_macro"])
    winner_name = winner_metrics["model"]
    print(f"\nWinner (by macro F1): {winner_name}")

    winner_model = TfidfSentimentModel(vectorizer, fitted[winner_name], winner_name)
    winner_model.save(models_dir / "baseline_sentiment.joblib")

    # A few concrete failure examples for the eventual error-analysis phase —
    # sampled now while we have predictions in hand, not invented later.
    test_df = test_df.reset_index(drop=True)
    y_pred_winner = predictions[winner_name]
    wrong_mask = pd.Series(y_pred_winner) != y_test.reset_index(drop=True)
    sample_errors = (
        test_df[wrong_mask.values]
        .assign(predicted=pd.Series(y_pred_winner)[wrong_mask.values].values)
        .rename(columns={"sentiment": "true_label"})[["text", "true_label", "predicted", "rating"]]
        .head(15)
    )
    sample_errors.to_csv(metrics_dir / "baseline_sample_errors.csv", index=False)

    comparison_path = metrics_dir / "baseline_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_metrics, "winner": winner_name}, f, indent=2)

    print(f"\nSaved winning model to {models_dir / 'baseline_sentiment.joblib'}")
    print(f"Saved comparison table to {comparison_path}")
    print(f"Saved {len(sample_errors)} sample errors to {metrics_dir / 'baseline_sample_errors.csv'}")


if __name__ == "__main__":
    main()