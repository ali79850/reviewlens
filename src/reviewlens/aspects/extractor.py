"""Online aspect extraction: dependency-parse candidates, snap to the
learned vocabulary when similar enough, otherwise keep as a novel aspect.

This is Approach A from the design phase (unsupervised), chosen for this
project given time/compute constraints. It has no gold-label evaluation
possible on this corpus -- quality is assessed qualitatively (see
scripts/demo_aspects.py) and documented as such, not claimed as validated.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from reviewlens.aspects.candidates import extract_candidate_phrases, load_spacy_model
from reviewlens.preprocessing.clean import clean_text

DEFAULT_SIMILARITY_THRESHOLD = 0.55


@dataclass
class AspectMention:
    aspect: str            # canonical name if matched, else the raw phrase (title-cased)
    source_sentence: str   # the clause/sentence this aspect was found in
    is_catalogued: bool    # True if matched to the learned vocabulary


class AspectExtractor:
    def __init__(self, vocab_path: Path, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.nlp = load_spacy_model()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_threshold = similarity_threshold

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.canonical_names = [v["canonical"] for v in vocab_data["vocabulary"]]
        self.canonical_embeddings = (
            self.embedder.encode(self.canonical_names, normalize_embeddings=True)
            if self.canonical_names else np.zeros((0, 384))
        )

    def extract(self, text: str) -> list[AspectMention]:
        doc = self.nlp(clean_text(text))
        candidates = extract_candidate_phrases(doc)
        if not candidates:
            return []

        # Dedup candidate phrases within this review, keeping first source sentence
        seen: dict[str, str] = {}
        for phrase, sentence in candidates:
            seen.setdefault(phrase, sentence)

        phrases = list(seen.keys())
        embeddings = self.embedder.encode(phrases, normalize_embeddings=True)

        results: dict[str, AspectMention] = {}
        for phrase, emb in zip(phrases, embeddings):
            if len(self.canonical_names) > 0:
                sims = self.canonical_embeddings @ emb  # cosine sim (both normalized)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
            else:
                best_sim = 0.0

            if best_sim >= self.similarity_threshold:
                aspect_name = self.canonical_names[best_idx]
                catalogued = True
            else:
                aspect_name = phrase.title()
                catalogued = False

            # If two raw phrases map to the same canonical aspect, keep the first
            if aspect_name not in results:
                results[aspect_name] = AspectMention(
                    aspect=aspect_name,
                    source_sentence=seen[phrase],
                    is_catalogued=catalogued,
                )

        return list(results.values())
