"""Aspect-level sentiment via a documented heuristic.

There is no trained aspect-sentiment classifier in this project's scope
(see README limitations). Instead, we run the whole-review sentiment model
on just the clause/sentence containing each aspect, which is a real model
applied at finer granularity -- not a fabricated or hardcoded output -- but
is explicitly weaker than a purpose-trained ABSA model would be, since a
clause can still itself contain more than one opinion.
"""
from dataclasses import dataclass

from reviewlens.aspects.extractor import AspectMention
from reviewlens.sentiment.base import SentimentModel


@dataclass
class AspectSentimentResult:
    aspect: str
    sentiment: str
    confidence: float
    is_catalogued: bool
    source_sentence: str
    method: str


def get_aspect_sentiments(
    mentions: list[AspectMention], sentiment_model: SentimentModel
) -> list[AspectSentimentResult]:
    if not mentions:
        return []

    sentences = [m.source_sentence for m in mentions]
    predictions = sentiment_model.predict_batch(sentences)

    results = []
    for mention, pred in zip(mentions, predictions):
        results.append(AspectSentimentResult(
            aspect=mention.aspect,
            sentiment=pred.label,
            confidence=pred.confidence,
            is_catalogued=mention.is_catalogued,
            source_sentence=mention.source_sentence,
            method=f"clause-heuristic+{pred.method}",
        ))
    return results
