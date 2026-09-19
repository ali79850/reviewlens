"""Typed result returned by ReviewAnalyzer.analyze().

Using a dataclass (not a raw dict) means the UI can't silently break if a
component's output shape changes, and every field carries its own method
tag so the app can honestly label what's measured-and-validated vs.
pretrained-and-unvalidated vs. a documented heuristic.
"""
from dataclasses import dataclass, field


@dataclass
class AspectResult:
    aspect: str
    sentiment: str
    confidence: float
    is_catalogued: bool
    source_sentence: str
    method: str


@dataclass
class AnalysisResult:
    text: str
    sentiment: str
    sentiment_confidence: float
    sentiment_method: str

    aspects: list[AspectResult] = field(default_factory=list)

    emotion_primary: str = "Neutral"
    emotion_distribution: dict = field(default_factory=dict)
    emotion_method: str = ""

    topic_id: int | None = None
    topic_keywords: list[str] = field(default_factory=list)
    topic_confidence: float = 0.0

    summary: str = ""
    summary_method: str = ""
