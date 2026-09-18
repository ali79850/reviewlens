"""Mapping from GoEmotions' 28 labels to this project's 7-emotion taxonomy.

This mapping is inherently lossy and is documented as such (see README
limitations): GoEmotions was trained on Reddit comments, not product
reviews, and has no native "frustration" class -- the nearest neighbor
(annoyance) is used instead. Labels with no confident match to the
requested taxonomy default to Neutral rather than being force-fit, since a
wrong forced mapping would misrepresent the model's actual output.
"""

TAXONOMY = ["Joy", "Satisfaction", "Anger", "Frustration", "Sadness", "Disappointment", "Neutral"]

GOEMOTIONS_TO_TAXONOMY = {
    # Joy: high-arousal positive affect
    "joy": "Joy",
    "amusement": "Joy",
    "excitement": "Joy",
    # Satisfaction: contentment / approval-type positive affect
    "approval": "Satisfaction",
    "gratitude": "Satisfaction",
    "relief": "Satisfaction",
    "admiration": "Satisfaction",
    "pride": "Satisfaction",
    "optimism": "Satisfaction",
    "caring": "Satisfaction",
    "love": "Satisfaction",
    # Anger: hostile negative affect
    "anger": "Anger",
    "disgust": "Anger",
    "disapproval": "Anger",
    # Frustration: GoEmotions has no direct equivalent -- "annoyance" is the
    # nearest neighbor, used here as a documented, imperfect substitute.
    "annoyance": "Frustration",
    # Sadness
    "sadness": "Sadness",
    "grief": "Sadness",
    "remorse": "Sadness",
    # Disappointment
    "disappointment": "Disappointment",
    # Neutral (native)
    "neutral": "Neutral",
    # No confident mapping to the requested taxonomy -- default to Neutral
    # rather than force a misleading fit.
    "confusion": "Neutral",
    "curiosity": "Neutral",
    "desire": "Neutral",
    "fear": "Neutral",
    "nervousness": "Neutral",
    "realization": "Neutral",
    "surprise": "Neutral",
    "embarrassment": "Neutral",
}
