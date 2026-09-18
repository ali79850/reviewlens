"""Text cleaning shared across the classical and transformer pipelines.

Kept deliberately light: transformers generally perform worse with heavy
cleaning (they rely on natural casing/punctuation for context), so this
function only strips things that are unambiguously noise, and the TF-IDF
vectorizer's own `lowercase=True` handles case-folding for the baseline.
"""
import html
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip URLs, HTML tags, and decode HTML entities (e.g. &#34; -> "),
    then collapse whitespace. Preserves case and punctuation, since both
    matter for sentiment and for transformer models."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text
