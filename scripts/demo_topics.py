"""
Phase 8 demo: show topic assignment for sample reviews.

Run from project root: python scripts/demo_topics.py
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
from reviewlens.topics.topic_model import TopicModel
from reviewlens.utils.paths import load_config, resolve

EXAMPLE_REVIEW = (
    "The laptop is fast and the display is excellent, but the battery dies "
    "within four hours and customer support never responded."
)


def main():
    cfg = load_config()
    models_dir = resolve(cfg["paths"]["models"])
    processed_dir = resolve(cfg["paths"]["data_processed"])

    print("Loading topic model...")
    model = TopicModel(models_dir / "topic_model.joblib")

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    sample_texts = [EXAMPLE_REVIEW] + test_df.sample(
        n=6, random_state=cfg["random_seed"]
    )["text"].tolist()

    for text in sample_texts:
        pred = model.predict(text)
        print(f"\nREVIEW: {text[:150]}{'...' if len(text) > 150 else ''}")
        print(f"  Topic {pred.topic_id} (confidence={pred.confidence}): {', '.join(pred.keywords)}")


if __name__ == "__main__":
    main()
