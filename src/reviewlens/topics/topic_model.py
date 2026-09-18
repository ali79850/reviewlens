"""Topic model wrapper: assigns a review to its nearest topic cluster."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from reviewlens.preprocessing.clean import clean_text


@dataclass
class TopicPrediction:
    topic_id: int
    keywords: list[str]
    confidence: float  # cosine similarity to the nearest cluster centroid
    method: str = "kmeans-c-tfidf"


class TopicModel:
    def __init__(self, model_path: Path):
        bundle = joblib.load(model_path)
        self.kmeans = bundle["kmeans"]
        self.cluster_keywords = bundle["cluster_keywords"]
        self.embedder = SentenceTransformer(bundle["embedder_name"])
        # Normalize centroids once for fast cosine-similarity confidence scoring
        centroids = self.kmeans.cluster_centers_
        self._centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    def predict(self, text: str) -> TopicPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[TopicPrediction]:
        cleaned = [clean_text(t) for t in texts]
        embeddings = self.embedder.encode(cleaned, normalize_embeddings=True)

        results = []
        for emb in embeddings:
            sims = self._centroids_norm @ emb
            topic_id = int(np.argmax(sims))
            confidence = float(sims[topic_id])
            results.append(TopicPrediction(
                topic_id=topic_id,
                keywords=self.cluster_keywords[topic_id],
                confidence=round(confidence, 4),
            ))
        return results
