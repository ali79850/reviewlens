"""
Phase 6: build an unsupervised aspect vocabulary from the training corpus.

This does NOT hardcode a list of aspects. It extracts frequent noun-phrase
candidates via dependency parsing, then clusters semantically similar
phrases (e.g. "battery", "battery life", "the battery pack") into canonical
aspects using sentence embeddings. The result is derived entirely from data
and is used only as a *prior* at inference time — novel/unseen phrases still
pass through uncatalogued rather than being dropped.

Run from project root: python scripts/build_aspect_vocabulary.py
"""
import json
import sys
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

sys.path.insert(0, "src")
from reviewlens.aspects.candidates import extract_candidate_phrases, load_spacy_model
from reviewlens.preprocessing.clean import clean_text
from reviewlens.utils.paths import load_config, resolve

VOCAB_SAMPLE_SIZE = 8000       # reviews scanned to build the vocabulary
MIN_PHRASE_FREQUENCY = 5       # a candidate must appear at least this often
MAX_VOCAB_CANDIDATES = 250     # top-N most frequent candidates get clustered
CLUSTER_DISTANCE_THRESHOLD = 0.35  # cosine distance; lower = stricter merging


def main():
    import pandas as pd

    cfg = load_config()
    processed_dir = resolve(cfg["paths"]["data_processed"])
    models_dir = resolve(cfg["paths"]["models"])

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    sample = train_df.sample(
        n=min(VOCAB_SAMPLE_SIZE, len(train_df)), random_state=cfg["random_seed"]
    )["text"]

    print(f"Scanning {len(sample)} reviews for candidate aspect phrases...")
    nlp = load_spacy_model()
    counter = Counter()

    for doc in tqdm(nlp.pipe((clean_text(t) for t in sample), batch_size=64), total=len(sample)):
        phrases = {p for p, _ in extract_candidate_phrases(doc)}  # dedup within a review
        counter.update(phrases)

    frequent = [(p, c) for p, c in counter.most_common() if c >= MIN_PHRASE_FREQUENCY]
    frequent = frequent[:MAX_VOCAB_CANDIDATES]
    print(f"Found {len(counter)} distinct candidates; "
          f"{len(frequent)} pass the frequency threshold and will be clustered.")

    phrases = [p for p, _ in frequent]
    freqs = {p: c for p, c in frequent}

    print("Embedding candidates with MiniLM...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(phrases, normalize_embeddings=True)

    print("Clustering semantically similar phrases...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    cluster_labels = clustering.fit_predict(embeddings)

    clusters: dict[int, list[str]] = {}
    for phrase, label in zip(phrases, cluster_labels):
        clusters.setdefault(label, []).append(phrase)

    vocabulary = []
    for members in clusters.values():
        canonical = max(members, key=lambda p: freqs[p])  # most frequent member names the cluster
        total_freq = sum(freqs[p] for p in members)
        vocabulary.append({
            "canonical": canonical,
            "members": sorted(members),
            "frequency": total_freq,
        })

    vocabulary.sort(key=lambda v: v["frequency"], reverse=True)

    out_path = models_dir / "aspect_vocabulary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": "unsupervised clustering over training corpus noun-phrase candidates",
            "n_reviews_scanned": len(sample),
            "n_raw_candidates": len(counter),
            "n_canonical_aspects": len(vocabulary),
            "vocabulary": vocabulary,
        }, f, indent=2)

    print(f"\nBuilt {len(vocabulary)} canonical aspects from {len(frequent)} candidates.")
    print(f"Saved to {out_path}")
    print("\nTop 15 canonical aspects by frequency:")
    for v in vocabulary[:15]:
        print(f"  {v['canonical']!r:30s} freq={v['frequency']:4d}  members={v['members']}")


if __name__ == "__main__":
    main()
