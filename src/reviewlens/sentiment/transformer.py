"""Fine-tuned DistilBERT sentiment model (CPU inference wrapper)."""
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reviewlens.preprocessing.clean import clean_text
from reviewlens.sentiment.base import SentimentModel, SentimentPrediction


class TransformerSentimentModel(SentimentModel):
    """Loads a fine-tuned DistilBERT sequence classifier for CPU inference.
    id2label mapping comes from the model's own config (set during training),
    so label ordering is never assumed here."""

    def __init__(self, model_dir: Path, method_name: str = "distilbert-finetuned", batch_size: int = 16):
        self.model_dir = str(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.method_name = method_name
        self.batch_size = batch_size
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> SentimentPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[SentimentPrediction]:
        from tqdm import tqdm

        results = []
        cleaned = [clean_text(t) for t in texts]

        for start in tqdm(range(0, len(cleaned), self.batch_size), desc="Transformer inference"):
            batch = cleaned[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch, truncation=True, max_length=256, padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=-1)

            for row in probs:
                idx = int(torch.argmax(row).item())
                label = self.id2label[idx]
                confidence = float(row[idx].item())
                results.append(SentimentPrediction(label=label, confidence=confidence, method=self.method_name))

        return results