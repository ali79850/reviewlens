"""
Preprocessing: star->sentiment label mapping, outlier handling, stratified split.

Run from project root: python scripts/prepare_labels.py
"""
import json
import sys

import pandas as pd

sys.path.insert(0, "src")
from reviewlens.utils.paths import load_config, resolve


def rating_to_sentiment(rating: float) -> str:
    """1-2 stars -> Negative, 3 -> Neutral, 4-5 -> Positive.

    This is a documented proxy label, not a native sentiment annotation --
    see README limitations. In particular 3-star reviews are a noisy stand-in
    for "neutral" and this is expected to be the hardest class to classify.
    """
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def main():
    cfg = load_config()
    raw_dir = resolve(cfg["paths"]["data_raw"])
    processed_dir = resolve(cfg["paths"]["data_processed"])
    metrics_dir = resolve(cfg["paths"]["results"] + "/metrics")

    df = pd.read_parquet(raw_dir / "amazon_electronics_raw.parquet")
    n_before = len(df)

    # Drop exact-duplicate review text (403 found in Phase 2 stats)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    n_after_dedup = len(df)

    # Cap extreme-length outliers at the 99th percentile rather than an
    # arbitrary fixed number, so the threshold is derived from this dataset's
    # actual distribution and stays reproducible if the corpus changes.
    length_cap = int(df["text"].str.len().quantile(0.99))
    df = df[df["text"].str.len() <= length_cap].reset_index(drop=True)
    n_after_cap = len(df)

    # Label mapping
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)

    # Stratified train/test split (80/20), seeded for reproducibility
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["sentiment"],
        random_state=cfg["random_seed"],
    )

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    report = {
        "n_rows_raw": n_before,
        "n_rows_after_dedup": n_after_dedup,
        "duplicates_dropped": n_before - n_after_dedup,
        "length_cap_chars_p99": length_cap,
        "n_rows_after_length_cap": n_after_cap,
        "outliers_dropped": n_after_dedup - n_after_cap,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "sentiment_distribution_full": df["sentiment"].value_counts().to_dict(),
        "sentiment_distribution_train": train_df["sentiment"].value_counts().to_dict(),
        "sentiment_distribution_test": test_df["sentiment"].value_counts().to_dict(),
    }
    with open(metrics_dir / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved train ({len(train_df)} rows) and test ({len(test_df)} rows) to {processed_dir}")


if __name__ == "__main__":
    main()