import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewlens.pipeline.analyzer import ReviewAnalyzer
from reviewlens.utils.paths import load_config, resolve

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")

cfg = load_config()
models_dir = resolve(cfg["paths"]["models"])


@st.cache_resource(show_spinner=False)
def get_analyzer():
    return ReviewAnalyzer(models_dir)


analyzer = get_analyzer()

# ---------------------------------------------------------------------------
# Styling — see design plan: cool paper canvas + dark control-panel sidebar,
# teal/coral/amber mapped consistently to positive/negative/neutral, thin
# bordered panels with a colored left edge instead of generic card shadows.
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink: #14171C;
  --paper: #F5F5F2;
  --slate: #5B6572;
  --teal: #157A6E;
  --coral: #D64545;
  --amber: #B8790E;
  --line: #DEDDD7;
}

html { color-scheme: light; }

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

div[data-testid="stTextArea"] textarea {
  background-color: #FFFFFF !important;
  color: #14171C !important;
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
}

section[data-testid="stSidebar"] { background-color: var(--ink); }
section[data-testid="stSidebar"] * { color: #E7E7E4 !important; }
section[data-testid="stSidebar"] hr { border-color: #33383F; }

.rl-hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 2.6rem;
  color: var(--ink);
  margin-bottom: 0.15rem;
  letter-spacing: -0.01em;
}
.rl-hero-subtitle { color: var(--slate); font-size: 1.05rem; margin-bottom: 0.4rem; }
.rl-hairline { border: none; border-top: 1px solid var(--line); margin: 1.4rem 0 1.6rem 0; }

.rl-card {
  background: #FFFFFF;
  border: 1px solid var(--line);
  border-left: 3px solid var(--ink);
  border-radius: 4px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 1rem;
}
.rl-card.positive { border-left-color: var(--teal); }
.rl-card.negative { border-left-color: var(--coral); }
.rl-card.neutral, .rl-card.mixed { border-left-color: var(--amber); }

.rl-card-label { font-size: 0.8rem; color: var(--slate); margin-bottom: 0.3rem; }

.rl-sentiment-word {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.9rem;
}
.rl-sentiment-word.positive { color: var(--teal); }
.rl-sentiment-word.negative { color: var(--coral); }
.rl-sentiment-word.neutral, .rl-sentiment-word.mixed { color: var(--amber); }

.rl-mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.83rem; color: var(--slate); }

.rl-aspect-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.5rem 0; border-bottom: 1px solid var(--line);
}
.rl-aspect-row:last-child { border-bottom: none; }
.rl-aspect-left { display: flex; align-items: center; }
.rl-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.6rem; flex-shrink: 0; }
.rl-dot.positive { background: var(--teal); }
.rl-dot.negative { background: var(--coral); }
.rl-dot.neutral { background: var(--amber); }
.rl-novel-tag {
  font-size: 0.72rem; color: var(--slate); border: 1px solid var(--line);
  border-radius: 3px; padding: 0 5px; margin-left: 0.5rem;
}

.rl-emotion-row { margin-bottom: 0.6rem; }
.rl-emotion-label { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.25rem; }
.rl-bar-track { background: var(--line); border-radius: 3px; height: 8px; overflow: hidden; }
.rl-bar-fill { background: var(--ink); height: 100%; border-radius: 3px; }

div.stButton > button {
  font-family: 'IBM Plex Sans', sans-serif;
  font-weight: 600;
  border-radius: 4px;
}

.rl-caveat { font-size: 0.82rem; color: var(--slate); margin-top: 0.6rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def sentiment_class(label: str) -> str:
    return label.lower() if label.lower() in ("positive", "negative", "neutral", "mixed") else "neutral"


# --- Sidebar ---
st.sidebar.markdown("### Settings")
model_choice_label = st.sidebar.radio(
    "Sentiment model",
    ["Transformer (DistilBERT, fine-tuned)", "Baseline (TF-IDF + Logistic Regression)"],
    index=0,
)
model_choice = "transformer" if model_choice_label.startswith("Transformer") else "baseline"

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p class="rl-caveat" style="color:#B7BBC2 !important;">'
    "<strong style='color:#E7E7E4 !important;'>Aspect extraction</strong> — unsupervised "
    "(dependency parsing + embedding clustering). No gold labels exist for this "
    "corpus, so quality is qualitative, not a measured accuracy.</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<p class="rl-caveat" style="color:#B7BBC2 !important;">'
    "<strong style='color:#E7E7E4 !important;'>Emotion detection</strong> — pretrained on "
    "Reddit comments (GoEmotions), not product reviews. Accuracy on this domain "
    "is unvalidated.</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<p class="rl-caveat" style="color:#B7BBC2 !important;">'
    "<strong style='color:#E7E7E4 !important;'>Topics</strong> — derived from clustering a "
    "15k-review sample into 6 broad themes.</p>",
    unsafe_allow_html=True,
)

# --- Hero ---
st.markdown('<div class="rl-hero-title">Customer Review Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="rl-hero-subtitle">Transform customer feedback into actionable insights.</div>',
    unsafe_allow_html=True,
)

review_text = st.text_area(
    "Paste a customer review",
    height=150,
    placeholder=(
        "The laptop is fast and the display is excellent, but the battery "
        "dies within four hours and customer support never responded."
    ),
    label_visibility="collapsed",
)

analyze_clicked = st.button("Analyze Review", type="primary")
st.markdown('<hr class="rl-hairline">', unsafe_allow_html=True)

if analyze_clicked:
    if not review_text or not review_text.strip():
        st.warning("Please enter a review to analyze.")
    elif len(review_text) > 20000:
        st.warning("This review is very long — results may be slow. Consider a shorter excerpt.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = analyzer.analyze(review_text, model_choice=model_choice)
            except Exception as e:
                st.error(f"Something went wrong during analysis: {e}")
                result = None

        if result:
            sc = sentiment_class(result.sentiment)

            col1, col2 = st.columns([2, 1])
            with col1:
                svm_note = (
                    '<div class="rl-mono">Note: SVM confidence is a margin-based proxy, not a calibrated probability.</div>'
                    if "svm" in result.sentiment_method else ""
                )
                st.markdown(
                    f"""
                    <div class="rl-card {sc}">
                      <div class="rl-card-label">Overall sentiment</div>
                      <div class="rl-sentiment-word {sc}">{result.sentiment}</div>
                      <div class="rl-bar-track" style="margin-top:0.6rem;">
                        <div class="rl-bar-fill" style="width:{min(result.sentiment_confidence,1.0)*100:.0f}%;"></div>
                      </div>
                      <div class="rl-mono" style="margin-top:0.4rem;">
                        confidence {result.sentiment_confidence:.0%} · method: {result.sentiment_method}
                      </div>
                      {svm_note}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                if result.topic_keywords:
                    st.markdown(
                        f"""
                        <div class="rl-card">
                          <div class="rl-card-label">Detected topic</div>
                          <div style="font-weight:600; margin-bottom:0.3rem;">Topic {result.topic_id}</div>
                          <div class="rl-mono">{', '.join(result.topic_keywords[:6])}</div>
                          <div class="rl-mono" style="margin-top:0.3rem;">confidence {result.topic_confidence:.0%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div class="rl-card">No topic model loaded.</div>', unsafe_allow_html=True)

            # Aspects
            if result.aspects:
                rows = ""
                for a in result.aspects:
                    ac = sentiment_class(a.sentiment)
                    novel = '<span class="rl-novel-tag">novel</span>' if not a.is_catalogued else ""
                    rows += f"""
                    <div class="rl-aspect-row">
                      <div class="rl-aspect-left"><span class="rl-dot {ac}"></span>{a.aspect}{novel}</div>
                      <div class="rl-mono">{a.sentiment} · {a.confidence:.0%}</div>
                    </div>
                    """
                st.markdown(
                    f"""
                    <div class="rl-card">
                      <div class="rl-card-label">Aspect analysis</div>
                      {rows}
                      <div class="rl-caveat">Aspect sentiment applies the sentiment model to each aspect's own clause, not a purpose-trained aspect-sentiment classifier.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="rl-card"><div class="rl-card-label">Aspect analysis</div>No aspects detected in this review.</div>',
                    unsafe_allow_html=True,
                )

            # Emotions
            sorted_emotions = sorted(result.emotion_distribution.items(), key=lambda x: -x[1])
            max_score = max((s for _, s in sorted_emotions), default=1.0) or 1.0
            emotion_rows = ""
            for emotion, score in sorted_emotions:
                width = min(score / max_score, 1.0) * 100
                emotion_rows += f"""
                <div class="rl-emotion-row">
                  <div class="rl-emotion-label"><span>{emotion}</span><span class="rl-mono">{score:.2f}</span></div>
                  <div class="rl-bar-track"><div class="rl-bar-fill" style="width:{width:.0f}%;"></div></div>
                </div>
                """
            st.markdown(
                f"""
                <div class="rl-card">
                  <div class="rl-card-label">Detected emotions</div>
                  {emotion_rows}
                  <div class="rl-caveat">Pretrained on Reddit comments, not product reviews — treat as indicative, not validated.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Summary
            st.markdown(
                f"""
                <div class="rl-card">
                  <div class="rl-card-label">Summary</div>
                  <div>{result.summary}</div>
                  <div class="rl-mono" style="margin-top:0.5rem;">method: {result.summary_method}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
