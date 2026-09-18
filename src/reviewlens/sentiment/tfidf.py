"""TF-IDF + linear classifier sentiment model (the classical baseline)."""
from pathlib import Path

import joblib
import numpy as np

from reviewlens.preprocessing.clean import clean_text
from reviewlens.sentiment.base import SentimentModel, SentimentPrediction


class TfidfSentimentModel(SentimentModel):
    """Wraps a fitted (vectorizer, classifier) pair. Works with either
    LogisticRegression (has predict_proba) or LinearSVC (uses decision
    function distance as a confidence proxy instead)."""

    def __init__(self, vectorizer, classifier, method_name: str):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.method_name = method_name
        self._has_proba = hasattr(classifier, "predict_proba")

    def _confidence_for(self, X_row) -> tuple[str, float]:
        if self._has_proba:
            proba = self.classifier.predict_proba(X_row)[0]
            idx = int(np.argmax(proba))
            label = self.classifier.classes_[idx]
            confidence = float(proba[idx])
        else:
            # LinearSVC: no probabilities. Use the margin (decision function)
            # and squash it into (0, 1) as a rough, clearly-labelled proxy —
            # this is NOT a calibrated probability, and the app must say so.
            scores = self.classifier.decision_function(X_row)[0]
            scores = np.atleast_1d(scores)
            idx = int(np.argmax(scores))
            label = self.classifier.classes_[idx]
            margin = float(scores[idx])
            confidence = float(1 / (1 + np.exp(-margin)))
        return label, confidence

    def predict(self, text: str) -> SentimentPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        results = []
        for i in range(X.shape[0]):
            label, confidence = self._confidence_for(X[i])
            results.append(SentimentPrediction(label=label, confidence=confidence, method=self.method_name))
        return results

    def save(self, path: Path) -> None:
        joblib.dump(
            {"vectorizer": self.vectorizer, "classifier": self.classifier, "method_name": self.method_name},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TfidfSentimentModel":
        obj = joblib.load(path)
        return cls(obj["vectorizer"], obj["classifier"], obj["method_name"])