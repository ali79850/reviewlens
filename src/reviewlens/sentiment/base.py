"""Common interface every sentiment model implementation must satisfy.

This is what makes the baseline-vs-transformer comparison a config choice
rather than a code fork: the Streamlit app and evaluation scripts call
`.predict()` without knowing which implementation backs it.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SentimentPrediction:
    label: str          # "Positive" | "Negative" | "Neutral"
    confidence: float   # 0-1, model's own probability/decision-based estimate
    method: str          # e.g. "tfidf-logreg", "tfidf-svm", "distilbert-finetuned"


class SentimentModel(ABC):
    @abstractmethod
    def predict(self, text: str) -> SentimentPrediction:
        ...

    @abstractmethod
    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        ...