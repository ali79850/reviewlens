"""
Phase 8: unsupervised topic discovery.

Design note: BERTopic's default backend depends on `hdbscan`, which has a
long history of failing to build on Windows (it compiles C extensions and
needs Visual Studio Build Tools) -- the same category of problem this
project already hit with `pandas` in Phase 2. Rather than risk that again,
this implements the same core idea BERTopic is built on -- embed documents,
cluster them, then extract each cluster's distinguishing vocabulary via
class-based TF-IDF (c-TF-IDF: treat each cluster as one "document" and run
TF-IDF across clusters, not across individual reviews) -- using only
scikit-learn and sentence-transformers, both already installed. This is the
"embeddings + clustering" approach the project brief explicitly allows as an
alternative to BERTopic.

K (number of topics) is chosen via silhouette score over a candidate range
on a subsample, rather than picked arbitrarily.

Run from project root: python scripts/build_topic_model.py
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.metrics import silhouette_score

sys.path.insert(0, "src")
from reviewlens.preprocessing.clean import clean_text
from reviewlens.utils.paths import load_config, resolve

TOPIC_SAMPLE_SIZE = 15_000       # capped per project scope decision (RAM/time)
K_CANDIDATES = [6, 8, 10, 12, 15, 18]
SILHOUETTE_SUBSAMPLE = 3000
TOP_KEYWORDS_PER_TOPIC = 10

# Standard English stopwords don't cover generic review-praise vocabulary
# ("great", "works", "product") since these are normal content words in
# general English -- just domain-generic here. With ~75% of this corpus
# being 5-star reviews using near-identical enthusiastic language, leaving
# these in causes c-TF-IDF to cluster partly on writing tone rather than
# product topic. This is standard domain-specific stopword augmentation,
# not a way of engineering a nicer-looking result.
GENERIC_REVIEW_STOPWORDS = {
    "great", "good", "like", "use", "used", "uses", "using", "works", "work",
    "working", "product", "products", "really", "nice", "love", "loved",
    "loves", "just", "also", "would", "get", "got", "one", "much", "many",
    "well", "recommend", "recommended", "buy", "bought", "item", "items",
    "amazon",
}
CUSTOM_STOPWORDS = list(sk_text.ENGLISH_STOP_WORDS.union(GENERIC_REVIEW_STOPWORDS))


def choose_best_k(embeddings: np.ndarray, seed: int) -> tuple[int, dict]:
    """Pick K by silhouette score on a subsample, so the topic count is
    derived from the data rather than an arbitrary choice."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embeddings), size=min(SILHOUETTE_SUBSAMPLE, len(embeddings)), replace=False)
    sub = embeddings[idx]

    scores = {}
    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        labels = km.fit_predict(sub)
        score = silhouette_score(sub, labels, metric="cosine")
        scores[k] = round(float(score), 4)
        print(f"  K={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    return best_k, scores


def main():
    cfg = load_config()
    processed_dir = resolve(cfg["paths"]["data_processed"])
    models_dir = resolve(cfg["paths"]["models"])
    metrics_dir = resolve(cfg["paths"]["results"] + "/metrics")

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    sample = train_df.sample(
        n=min(TOPIC_SAMPLE_SIZE, len(train_df)), random_state=cfg["random_seed"]
    ).reset_index(drop=True)
    texts = [clean_text(t) for t in sample["text"]]

    cache_path = models_dir / "topic_embeddings_cache.joblib"
    if cache_path.exists():
        cached = joblib.load(cache_path)
        if cached.get("n_texts") == len(texts):
            print("Using cached embeddings (delete topic_embeddings_cache.joblib to force recompute).")
            embeddings = cached["embeddings"]
        else:
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print(f"Embedding {len(texts)} reviews with MiniLM...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        joblib.dump({"n_texts": len(texts), "embeddings": embeddings}, cache_path)

    print("Selecting K via silhouette score...")
    best_k, silhouette_scores = choose_best_k(embeddings, cfg["random_seed"])
    print(f"Chosen K={best_k}")

    print(f"Fitting final KMeans with K={best_k} on the full sample...")
    kmeans = KMeans(n_clusters=best_k, random_state=cfg["random_seed"], n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # c-TF-IDF: treat each cluster as one combined document, run TF-IDF
    # across clusters (not across individual reviews) to find each
    # cluster's distinguishing vocabulary.
    cluster_docs = []
    cluster_sizes = {}
    for c in range(best_k):
        members = [texts[i] for i in range(len(texts)) if cluster_labels[i] == c]
        cluster_docs.append(" ".join(members))
        cluster_sizes[c] = len(members)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words=CUSTOM_STOPWORDS, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(cluster_docs)
    feature_names = vectorizer.get_feature_names_out()

    cluster_keywords = {}
    for c in range(best_k):
        row = tfidf_matrix[c].toarray().flatten()
        top_idx = row.argsort()[::-1][:TOP_KEYWORDS_PER_TOPIC]
        cluster_keywords[c] = [feature_names[i] for i in top_idx]

    # Save model bundle
    bundle = {
        "kmeans": kmeans,
        "vectorizer": vectorizer,
        "cluster_keywords": cluster_keywords,
        "embedder_name": "all-MiniLM-L6-v2",
        "k": best_k,
    }
    joblib.dump(bundle, models_dir / "topic_model.joblib")

    summary = {
        "method": "MiniLM embeddings + KMeans + c-TF-IDF (BERTopic-style, hdbscan-free)",
        "reason_for_not_using_bertopic": (
            "BERTopic's default backend (hdbscan) has a well-documented history "
            "of failing to build on Windows with recent Python versions, "
            "requiring Visual Studio Build Tools -- the same category of issue "
            "already encountered with pandas earlier in this project. This "
            "implementation reproduces BERTopic's core idea (cluster + "
            "class-based TF-IDF) using only scikit-learn and "
            "sentence-transformers, both already required elsewhere in the "
            "project."
        ),
        "n_reviews_used": len(texts),
        "k_selection_method": "silhouette score (cosine) over candidates on a 3000-review subsample",
        "silhouette_scores_by_k": silhouette_scores,
        "chosen_k": best_k,
        "cluster_sizes": cluster_sizes,
        "cluster_keywords": {str(k): v for k, v in cluster_keywords.items()},
    }
    with open(metrics_dir / "topic_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model to {models_dir / 'topic_model.joblib'}")
    print(f"Saved summary to {metrics_dir / 'topic_model_summary.json'}")
    print("\nTopics found (sorted by size):")
    for c, size in sorted(cluster_sizes.items(), key=lambda x: -x[1]):
        print(f"  Topic {c} (n={size}): {', '.join(cluster_keywords[c])}")


if __name__ == "__main__":
    main()
