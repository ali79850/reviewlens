"""
Phase 7 demo: qualitative check of emotion detection.

No emotion-labeled data exists for this corpus, so this is judged by reading
real output, not a metric -- consistent with the aspect extraction phase.

Run from project root: python scripts/demo_emotions.py
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
from reviewlens.emotions.classifier import EmotionClassifier
from reviewlens.utils.paths import load_config, resolve

EXAMPLE_REVIEW = (
    "The laptop is fast and the display is excellent, but the battery dies "
    "within four hours and customer support never responded."
)


def main():
    cfg = load_config()
    processed_dir = resolve(cfg["paths"]["data_processed"])

    print("Loading emotion classifier...")
    classifier = EmotionClassifier()

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    sample_texts = [EXAMPLE_REVIEW] + test_df.sample(
        n=6, random_state=cfg["random_seed"]
    )["text"].tolist()

    for text in sample_texts:
        pred = classifier.predict(text)
        print(f"\nREVIEW: {text[:150]}{'...' if len(text) > 150 else ''}")
        print(f"  Primary emotion: {pred.primary_emotion} (score={pred.primary_score})")
        sorted_dist = sorted(pred.distribution.items(), key=lambda x: -x[1])
        for emotion, score in sorted_dist:
            bar = "█" * int(score * 20)
            print(f"    {emotion:15s} {score:.3f} {bar}")


if __name__ == "__main__":
    main()
