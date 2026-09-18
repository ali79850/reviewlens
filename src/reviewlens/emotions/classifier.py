"""Pretrained emotion classifier (GoEmotions), aggregated into a 7-class
taxonomy. See taxonomy.py for the documented, lossy label mapping.

IMPORTANT: this model was trained on Reddit comments, not product reviews.
Its accuracy on this domain is unvalidated -- there are no emotion labels
for this corpus to evaluate against. This is disclosed to the end user in
the app, not presented as a measured capability.
"""
from dataclasses import dataclass

from transformers import pipeline

from reviewlens.emotions.taxonomy import GOEMOTIONS_TO_TAXONOMY, TAXONOMY
from reviewlens.preprocessing.clean import clean_text

MODEL_NAME = "SamLowe/roberta-base-go_emotions"


@dataclass
class EmotionPrediction:
    primary_emotion: str
    primary_score: float
    distribution: dict  # {taxonomy_label: aggregated_score}, for a bar-chart UI
    method: str = "pretrained-go_emotions (unvalidated on product reviews)"


class EmotionClassifier:
    def __init__(self, batch_size: int = 16):
        self._pipe = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            top_k=None,  # return scores for all 28 labels, not just the top one
        )
        self.batch_size = batch_size

    def _aggregate(self, raw_scores: list[dict]) -> EmotionPrediction:
        """raw_scores: list of {'label': ..., 'score': ...} for all 28 GoEmotions labels."""
        bucket_scores = {t: 0.0 for t in TAXONOMY}
        for item in raw_scores:
            bucket = GOEMOTIONS_TO_TAXONOMY.get(item["label"], "Neutral")
            bucket_scores[bucket] += item["score"]

        primary = max(bucket_scores, key=bucket_scores.get)
        return EmotionPrediction(
            primary_emotion=primary,
            primary_score=round(bucket_scores[primary], 4),
            distribution={k: round(v, 4) for k, v in bucket_scores.items()},
        )

    def predict(self, text: str) -> EmotionPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[EmotionPrediction]:
        cleaned = [clean_text(t) for t in texts]
        results = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start:start + self.batch_size]
            raw_batch = self._pipe(batch, truncation=True, max_length=256)
            for raw_scores in raw_batch:
                results.append(self._aggregate(raw_scores))
        return results
