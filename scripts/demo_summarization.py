"""
Phase 9 demo: qualitative check of extractive summarization.

No reference summaries exist for this corpus, so ROUGE isn't possible --
this is judged by reading original vs. summary side by side, same honest
pattern as aspects/emotions/topics.

Run from project root: python scripts/demo_summarization.py
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
from reviewlens.summarization.summarizer import ExtractiveSummarizer
from reviewlens.utils.paths import load_config, resolve


def main():
    cfg = load_config()
    processed_dir = resolve(cfg["paths"]["data_processed"])

    print("Loading summarizer...")
    summarizer = ExtractiveSummarizer()

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    test_df["word_count"] = test_df["text"].str.split().str.len()

    # Sample from longer reviews specifically, since short ones trivially
    # skip summarization -- we want to see the actual selection behavior
    long_reviews = test_df[test_df["word_count"] >= 60].sample(n=6, random_state=cfg["random_seed"])

    for text in long_reviews["text"]:
        result = summarizer.summarize(text)
        print(f"\n{'='*80}")
        print(f"ORIGINAL ({len(text.split())} words):\n{text}")
        print(f"\nSUMMARY ({result.method}, {result.n_sentences_selected}/{result.n_sentences_total} sentences):")
        print(f"  {result.summary}")


if __name__ == "__main__":
    main()
