"""Extractive summarization via embedding-centroid sentence selection.

Per project scope decision: extractive only, no abstractive model. This
corpus has no reference summaries, so ROUGE evaluation is impossible;
extractive selection also cannot hallucinate content that wasn't in the
original review, which matters more here than fluency.

Method: embed each sentence, compute the review's overall meaning as the
centroid of its sentence embeddings, then select whichever sentence(s) sit
closest to that centroid. This is a standard, well-established technique
(centroid-based extractive summarization), not a shortcut invention.
"""
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from reviewlens.aspects.candidates import load_spacy_model
from reviewlens.preprocessing.clean import clean_text

MIN_WORDS_TO_SUMMARIZE = 40  # below this, the review IS already a summary
MAX_SUMMARY_SENTENCES = 3


@dataclass
class SummaryResult:
    summary: str
    method: str
    n_sentences_selected: int
    n_sentences_total: int


class ExtractiveSummarizer:
    def __init__(self):
        self.nlp = load_spacy_model()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def summarize(self, text: str) -> SummaryResult:
        cleaned = clean_text(text)
        word_count = len(cleaned.split())

        if word_count < MIN_WORDS_TO_SUMMARIZE:
            return SummaryResult(
                summary=cleaned,
                method="original (too short to summarize meaningfully)",
                n_sentences_selected=0,
                n_sentences_total=0,
            )

        doc = self.nlp(cleaned)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if len(sentences) <= 1:
            return SummaryResult(
                summary=cleaned,
                method="original (single sentence)",
                n_sentences_selected=0,
                n_sentences_total=len(sentences),
            )

        embeddings = self.embedder.encode(sentences, normalize_embeddings=True)
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        similarities = embeddings @ centroid
        n_select = min(MAX_SUMMARY_SENTENCES, max(1, round(len(sentences) * 0.3)))
        top_indices = sorted(similarities.argsort()[::-1][:n_select])  # keep original order

        summary_text = " ".join(sentences[i] for i in top_indices)
        return SummaryResult(
            summary=summary_text,
            method="extractive-centroid (MiniLM)",
            n_sentences_selected=n_select,
            n_sentences_total=len(sentences),
        )
