"""
Phase 5 (cont.): evaluate the fine-tuned transformer on the FULL local test
set (not just Colab's 5k subset), for a fair apples-to-apples comparison with
the TF-IDF baseline, which was evaluated on all 9,821 test rows.

Run from project root: python scripts/evaluate_transformer.py
"""
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, "src")
from reviewlens.sentiment.transformer import TransformerSentimentModel
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


def main():
    cfg = load_config()
    processed_dir = resolve(cfg["paths"]["data_processed"])
    models_dir = resolve(cfg["paths"]["models"])
    metrics_dir = resolve(cfg["paths"]["results"] + "/metrics")
    figures_dir = resolve(cfg["paths"]["results"] + "/figures")

    test_df = pd.read_parquet(processed_dir / "test.parquet").reset_index(drop=True)
    print(f"Evaluating on full test set: {len(test_df)} rows (this may take a few minutes on CPU)")

    model = TransformerSentimentModel(models_dir / "transformer_sentiment", batch_size=8)
    predictions = model.predict_batch(test_df["text"].tolist())
    y_pred = [p.label for p in predictions]
    y_true = test_df["sentiment"].tolist()

    metrics = evaluate(y_true, y_pred, "distilbert-finetuned")
    print(json.dumps(metrics, indent=2))

    report_text = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    print(report_text)
    with open(metrics_dir / "transformer_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - distilbert-finetuned")
    fig.tight_layout()
    fig.savefig(figures_dir / "transformer_confusion_matrix.png", dpi=150)
    plt.close(fig)

    wrong_mask = pd.Series(y_pred) != pd.Series(y_true)
    sample_errors = (
        test_df[wrong_mask.values]
        .assign(predicted=pd.Series(y_pred)[wrong_mask.values].values)
        .rename(columns={"sentiment": "true_label"})[["text", "true_label", "predicted", "rating"]]
        .head(15)
    )
    sample_errors.to_csv(metrics_dir / "transformer_sample_errors.csv", index=False)

    # Merge with the baseline comparison for one final side-by-side table
    with open(metrics_dir / "baseline_comparison.json", "r", encoding="utf-8") as f:
        baseline_results = json.load(f)["results"]

    all_results = baseline_results + [metrics]
    winner = max(all_results, key=lambda m: m["f1_macro"])

    final_comparison = {
        "results": all_results,
        "winner_by_macro_f1": winner["model"],
        "note": (
            "Baseline models evaluated on the full 9,821-row test set. "
            "Transformer fine-tuned on a 20k-row Colab subsample (T4 GPU) "
            "and evaluated here on the same full 9,821-row test set for a "
            "fair comparison."
        ),
    }
    with open(metrics_dir / "sentiment_model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(final_comparison, f, indent=2)

    print(f"\nFinal comparison (winner by macro F1: {winner['model']}):")
    print(json.dumps(final_comparison, indent=2))
    print(f"\nSaved to {metrics_dir / 'sentiment_model_comparison.json'}")


if __name__ == "__main__":
    main()