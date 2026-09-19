"""ReviewAnalyzer: single facade combining every component behind one
.analyze() call. Models are loaded once (lazily, on first use) and cached
for the analyzer's lifetime -- given each component's real measured size
(transformer ~268MB, MiniLM ~90MB, GoEmotions ~499MB, spaCy ~12MB), all of
them fit comfortably in RAM together on this project's target hardware
(~8GB), so no aggressive load/evict cycling is needed. This revises an
earlier, more conservative assumption from initial planning, now that
actual model sizes are known rather than estimated.
"""
from pathlib import Path

from reviewlens.aspects.aspect_sentiment import get_aspect_sentiments
from reviewlens.aspects.extractor import AspectExtractor
from reviewlens.emotions.classifier import EmotionClassifier
from reviewlens.pipeline.schemas import AnalysisResult, AspectResult
from reviewlens.sentiment.tfidf import TfidfSentimentModel
from reviewlens.sentiment.transformer import TransformerSentimentModel
from reviewlens.summarization.summarizer import ExtractiveSummarizer
from reviewlens.topics.topic_model import TopicModel


class ReviewAnalyzer:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._baseline_model = None
        self._transformer_model = None
        self._aspect_extractor = None
        self._emotion_classifier = None
        self._topic_model = None
        self._summarizer = None

    @property
    def baseline_model(self):
        if self._baseline_model is None:
            self._baseline_model = TfidfSentimentModel.load(self.models_dir / "baseline_sentiment.joblib")
        return self._baseline_model

    @property
    def transformer_model(self):
        if self._transformer_model is None:
            self._transformer_model = TransformerSentimentModel(self.models_dir / "transformer_sentiment")
        return self._transformer_model

    @property
    def aspect_extractor(self):
        if self._aspect_extractor is None:
            self._aspect_extractor = AspectExtractor(self.models_dir / "aspect_vocabulary.json")
        return self._aspect_extractor

    @property
    def emotion_classifier(self):
        if self._emotion_classifier is None:
            self._emotion_classifier = EmotionClassifier()
        return self._emotion_classifier

    @property
    def topic_model(self):
        if self._topic_model is None:
            self._topic_model = TopicModel(self.models_dir / "topic_model.joblib")
        return self._topic_model

    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = ExtractiveSummarizer()
        return self._summarizer

    def analyze(self, text: str, model_choice: str = "transformer") -> AnalysisResult:
        """Run the full pipeline on a single review.

        model_choice: "baseline" or "transformer" -- selects which sentiment
        model is used both for the headline sentiment and for aspect-level
        sentiment (which reuses this same model on each aspect's clause).
        """
        sentiment_model = self.transformer_model if model_choice == "transformer" else self.baseline_model

        overall = sentiment_model.predict(text)

        mentions = self.aspect_extractor.extract(text)
        aspect_sentiments = get_aspect_sentiments(mentions, sentiment_model)
        aspects = [
            AspectResult(
                aspect=a.aspect,
                sentiment=a.sentiment,
                confidence=a.confidence,
                is_catalogued=a.is_catalogued,
                source_sentence=a.source_sentence,
                method=a.method,
            )
            for a in aspect_sentiments
        ]

        # "Mixed" overrides the whole-review model output when aspects
        # genuinely disagree -- derived from real aspect outputs, not a
        # separate trained class (no labels for "Mixed" exist in this corpus).
        aspect_labels = {a.sentiment for a in aspects}
        if "Positive" in aspect_labels and "Negative" in aspect_labels:
            final_sentiment = "Mixed"
        else:
            final_sentiment = overall.label

        emotion_pred = self.emotion_classifier.predict(text)
        topic_pred = self.topic_model.predict(text)
        summary_result = self.summarizer.summarize(text)

        return AnalysisResult(
            text=text,
            sentiment=final_sentiment,
            sentiment_confidence=overall.confidence,
            sentiment_method=overall.method,
            aspects=aspects,
            emotion_primary=emotion_pred.primary_emotion,
            emotion_distribution=emotion_pred.distribution,
            emotion_method=emotion_pred.method,
            topic_id=topic_pred.topic_id,
            topic_keywords=topic_pred.keywords,
            topic_confidence=topic_pred.confidence,
            summary=summary_result.summary,
            summary_method=summary_result.method,
        )
