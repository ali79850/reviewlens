"""
Phase 6 demo: qualitative check of aspect extraction + aspect sentiment.

This is unsupervised and has no gold labels on this corpus, so quality is
assessed by reading real output, not a metric. Run this and eyeball whether
the extracted aspects and their sentiment look reasonable.

Run from project root: python scripts/demo_aspects.py
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
from reviewlens.aspects.aspect_sentiment import get_aspect_sentiments
from reviewlens.aspects.extractor import AspectExtractor
from reviewlens.sentiment.transformer import TransformerSentimentModel
from reviewlens.utils.paths import load_config, resolve

EXAMPLE_REVIEW = (
    "The laptop is fast and the display is excellent, but the battery dies "
    "within four hours and customer support never responded."
)


def print_result(text: str, extractor: AspectExtractor, sentiment_model):
    print(f"\nREVIEW: {text}")
    mentions = extractor.extract(text)
    if not mentions:
        print("  (no aspects extracted)")
        return
    results = get_aspect_sentiments(mentions, sentiment_model)
    for r in results:
        tag = "catalogued" if r.is_catalogued else "novel"
        print(f"  - {r.aspect:25s} {r.sentiment:10s} (conf={r.confidence:.2f}, {tag})")
        print(f"      from: \"{r.source_sentence}\"")


def main():
    cfg = load_config()
    models_dir = resolve(cfg["paths"]["models"])
    processed_dir = resolve(cfg["paths"]["data_processed"])

    print("Loading models (this takes a few seconds)...")
    extractor = AspectExtractor(models_dir / "aspect_vocabulary.json")
    sentiment_model = TransformerSentimentModel(models_dir / "transformer_sentiment")

    print_result(EXAMPLE_REVIEW, extractor, sentiment_model)

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    sample = test_df.sample(n=5, random_state=cfg["random_seed"])
    for text in sample["text"]:
        print_result(text, extractor, sentiment_model)


if __name__ == "__main__":
    main()
