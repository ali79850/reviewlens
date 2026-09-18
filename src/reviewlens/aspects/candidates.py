"""Candidate aspect phrase extraction via spaCy dependency parsing.

Shared between the offline vocabulary-building script and the online
extractor so both use identical logic — otherwise the vocabulary built
offline could silently drift from what inference actually produces.
"""
import re

_LEADING_DET_RE = re.compile(
    r"^(the|a|an|this|that|these|those|my|your|his|her|its|our|their)\s+", re.I
)
_STOP_LEMMAS = {"i", "it", "this", "that", "thing", "one", "something", "anything", "everything"}


def load_spacy_model():
    import spacy
    return spacy.load("en_core_web_sm")


_CONTRAST_MARKERS = {"but", "however", "although", "though", "yet", "whereas"}


def _split_into_clauses(sent):
    """Split a spaCy sentence into clauses on contrastive conjunctions.

    spaCy's sentence boundaries don't split compound sentences joined by
    commas + 'but'/'however'/etc — e.g. "X is great, but Y is terrible" stays
    one sentence. Without this, every aspect in a mixed-sentiment sentence
    gets the *same* source text, and a whole-sentence sentiment model then
    assigns them all the same (wrong) label. This is a simple, documented
    heuristic, not a claim of full clause-level parsing.
    """
    tokens = list(sent)
    split_positions = [i for i, t in enumerate(tokens) if t.lower_ in _CONTRAST_MARKERS]
    if not split_positions:
        return [(sent.text.strip(), sent.start, sent.end)]

    boundaries = [0] + split_positions + [len(tokens)]
    clauses = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if start == end:
            continue
        span_tokens = tokens[start:end]
        text = "".join(t.text_with_ws for t in span_tokens).strip()
        if text:
            clauses.append((text, sent.start + start, sent.start + end))
    return clauses


def extract_candidate_phrases(doc) -> list[tuple[str, str]]:
    """Return (normalized_phrase, source_clause_text) pairs for noun
    chunks headed by a noun/proper noun, excluding pronoun-like chunks.
    Deliberately generic (no fixed aspect list) so it generalizes to
    aspects not seen during vocabulary building. Each candidate is matched
    to its containing clause (see _split_into_clauses), not the whole
    sentence, so aspect-level sentiment isn't diluted by unrelated clauses."""
    results = []
    for sent in doc.sents:
        clauses = _split_into_clauses(sent)
        for chunk in sent.noun_chunks:
            if chunk.root.pos_ not in ("NOUN", "PROPN"):
                continue
            if chunk.root.is_stop or chunk.root.lemma_.lower() in _STOP_LEMMAS:
                continue

            phrase = _LEADING_DET_RE.sub("", chunk.text.strip().lower())
            if not phrase or phrase.isdigit() or len(phrase.split()) > 4:
                continue

            # Find which clause this chunk falls in
            clause_text = sent.text.strip()  # fallback
            for text, start, end in clauses:
                if start <= chunk.start < end:
                    clause_text = text
                    break

            results.append((phrase, clause_text))
    return results
