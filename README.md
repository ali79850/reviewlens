# ReviewLens

An end-to-end NLP system for customer review intelligence: sentiment, aspect-level sentiment, emotion, topic discovery, and summarization — with an explicit, disclosed line between what's supervised and evaluated, what's a pretrained model used as-is, and what's a documented heuristic.

This README is written after the fact, from real measured results. Nothing below is estimated or aspirational; where something wasn't validated, that's stated directly.

## Hardware this was built and run on

- AMD Ryzen 5 7430U (6c/12t), 8 GB RAM (~7.4 GB usable), integrated Radeon graphics — no CUDA/ROCm training path
- Windows 11, Python 3.13, VS Code, cmd/PowerShell
- Model **training** done on Google Colab (free T4 GPU tier); all **inference** runs locally on CPU

This constrained model sizes, corpus sizes, and the local/Colab split throughout — see the design decisions below.

## Setup

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python -m spacy download en_core_web_sm
copy .env.example .env
```

Note on `requirements.txt`: versions are specified as lower bounds (`>=`), not exact pins, because this project was built on Python 3.13, which is new enough that several packages (`pandas`, `huggingface_hub`) required versions substantially newer than their "stable" releases to have prebuilt Windows wheels available at all. If reproducing this on an older Python, you may be able to pin tighter; on 3.13, loosen further if a specific version fails to find a wheel.

## Reproducing the full pipeline, in order

```cmd
python scripts\download_data.py                    # ~30-60 min (one-time, cached after)
python scripts\prepare_labels.py
python scripts\train_baseline.py
# then: fine-tune the transformer on Colab (notebooks\colab_train_transformer.ipynb), download the model into models\transformer_sentiment\
python scripts\evaluate_transformer.py
python scripts\build_aspect_vocabulary.py
python scripts\build_topic_model.py
python scripts\generate_error_analysis_report.py
streamlit run app\app.py
```

Demo/qualitative-check scripts (`demo_aspects.py`, `demo_emotions.py`, `demo_topics.py`, `demo_summarization.py`) can be run any time after their corresponding build step, and are how several of the findings below were actually discovered.

## Dataset

**Source**: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (UCSD), Electronics category.

**Access note**: the Hugging Face repo's official loading script no longer runs under `datasets>=4.0` (script-based dataset loading was removed), and only the `raw_meta_*` configs were ever migrated to Parquet on the Hub — review data was not. This project reads the review data directly from the dataset authors' own hosted JSONL.gz file, streamed and reservoir-sampled (uniform, unbiased by file order) rather than downloaded in full, since the source file is ~6.5 GB compressed and substantially larger uncompressed.

**Sample collected**: 50,000 reviews, reservoir-sampled from ~43.9 million.

**Real rating distribution** (before any filtering):

| Rating | Count |
|---|---|
| 1★ | 6,399 |
| 2★ | 2,758 |
| 3★ | 3,457 |
| 4★ | 6,492 |
| 5★ | 30,894 |

Text length: mean 264 chars, median 137, min 15, max 26,723 (heavy right skew — the max was treated as an outlier, see preprocessing).

**Preprocessing**: 403 exact-duplicate reviews dropped (49,597 remain); reviews longer than the 99th percentile (2,007 chars) dropped as length outliers (49,101 remain); stratified 80/20 train/test split (39,280 / 9,821), seeded for reproducibility.

**Sentiment label mapping** (a documented proxy, not a native annotation): 1–2★ → Negative, 3★ → Neutral, 4–5★ → Positive. Resulting distribution: Positive 36,635 (74.6%), Negative 9,069 (18.5%), Neutral 3,397 (6.9%). The Neutral class's small size and its role as a noisy proxy for what is often genuinely mixed sentiment is the single biggest driver of model error across every sentiment model in this project (see Error Analysis below).

**License**: subject to the Amazon Reviews 2023 dataset's own terms as published on its Hugging Face dataset card — this project uses it for non-commercial, educational/portfolio purposes only.

## Architecture

```
app/                      Streamlit UI (presentation only — no model logic)
  app.py                  Single-review analysis page
src/reviewlens/
  utils/                  Config + path resolution
  preprocessing/          Shared text cleaning
  sentiment/              TF-IDF baseline + fine-tuned transformer, behind one interface
  aspects/                Unsupervised extraction + clause-level aspect sentiment
  emotions/               Pretrained GoEmotions classifier + taxonomy mapping
  topics/                 Embedding + clustering topic discovery
  summarization/          Extractive (centroid-based) summarization
  pipeline/               ReviewAnalyzer facade + typed AnalysisResult schema
scripts/                  One script per pipeline stage (see "Reproducing" above)
notebooks/                Colab transformer training notebook
models/                   Trained artifacts (gitignored)
data/                     Raw + processed data (gitignored)
results/metrics/          Real metrics, classification reports, error analysis (committed)
results/figures/          Confusion matrices (committed)
```

`ReviewAnalyzer.analyze(text)` is the single entry point the UI calls; it returns a typed `AnalysisResult` where every field carries its own `method` string, so the UI (and this README) can be honest about which numbers are measured, which are pretrained-and-unvalidated, and which are a disclosed heuristic.

## Sentiment: baseline vs. transformer

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 82.3% | 63.9% | 69.7% | 65.9% |
| TF-IDF + Linear SVM | 85.9% | 65.9% | 65.0% | 65.3% |
| **DistilBERT (fine-tuned)** | **88.0%** | **68.8%** | **68.4%** | **68.5%** |

All three evaluated on the same full 9,821-row held-out test set. The transformer was fine-tuned on Colab (T4 GPU, 3 epochs, 20k-row training subsample for tractability) and re-evaluated locally on the full test set for a fair comparison — its Colab-reported metrics (on a 5k random eval subsample) were slightly higher (F1-macro 70.2%) and are not used as the headline number, since they aren't measured on the same set as the baselines.

**Why macro F1, not accuracy, decides the winner**: TF-IDF + LogReg has the *best* Neutral-class recall (44%) of all three models, despite losing on every other metric — a fact accuracy alone would completely hide. Given the class imbalance (Neutral is 6.9% of the data), macro F1 is the metric that actually reflects balanced performance across all three classes.

**Neutral class breakdown** (the consistent weak point across every model):

| Model | Neutral precision / recall / F1 |
|---|---|
| TF-IDF + LogReg | 0.26 / 0.44 / 0.33 |
| TF-IDF + SVM | 0.31 / 0.24 / 0.27 |
| DistilBERT | 0.33 / 0.28 / 0.30 |

## Aspect extraction & aspect-level sentiment

**Method**: unsupervised. spaCy dependency parsing extracts candidate noun-phrase aspects (no hardcoded aspect list); a vocabulary of 183 canonical aspects was built by clustering 250 frequent candidates (from 24,967 raw candidates found scanning 8,000 training reviews) via MiniLM embeddings + agglomerative clustering, so e.g. "battery"/"battery life"/"the battery pack" collapse into one aspect. Novel aspects not matching the learned vocabulary still surface (title-cased), rather than being dropped.

**Aspect-level sentiment** is a disclosed heuristic: the sentiment model is applied to the clause containing each aspect, not a purpose-trained aspect-sentiment classifier (no gold aspect-sentiment labels exist for this corpus).

**No quantitative evaluation is possible or claimed** — there are no gold aspect labels for this corpus. Quality was assessed qualitatively via `demo_aspects.py`.

**Two real, distinct issues found and handled differently:**
- *Whole-sentence dilution (fixed)*: the flagship example — "The laptop is fast and the display is excellent, but the battery dies within four hours..." — is grammatically one spaCy sentence. Before a fix, every aspect received the same source text, so all five aspects (including the clearly-positive laptop/display) came back Negative. Fixed by splitting on contrastive conjunctions (but/however/although/etc.) before assigning each aspect its clause.
- *Semantic over-clustering (documented, not fixed)*: "retailer" clustered into the same canonical aspect as "amazon" — an inherent trade-off of embedding-based clustering with no gold labels to correct against.

## Emotion detection

**Method**: pretrained `SamLowe/roberta-base-go_emotions` (trained on GoEmotions — Reddit comments, 28 emotion labels). Mapped into this project's 7-class taxonomy (Joy, Satisfaction, Anger, Frustration, Sadness, Disappointment, Neutral); labels with no confident match default to Neutral rather than being force-fit. GoEmotions has no native "frustration" class — "annoyance" is used as the nearest available substitute, disclosed as such.

**Unvalidated on this domain** — no emotion-labeled review data exists to evaluate against. A concrete, observed failure: a clearly negative review (defective product, fake branding sticker, broken buttons) scored **Satisfaction (0.65)** as its top emotion, with Anger/Frustration both under 0.03 — a real example of the Reddit→review domain gap, not just a generic disclaimer.

## Topic discovery

**Method**: MiniLM embeddings + KMeans + class-based TF-IDF (c-TF-IDF), over a 15,000-review sample. **Not BERTopic** — its default backend (`hdbscan`) has a well-documented history of failing to build on Windows with recent Python versions (compiles C extensions, needs Visual Studio Build Tools), the same category of problem this project hit with `pandas` in an earlier phase. This implementation reproduces BERTopic's actual core idea (cluster, then extract each cluster's distinguishing vocabulary via TF-IDF across clusters) using only packages already required elsewhere in the project.

K=6 was chosen via silhouette score over K∈{6,8,10,12,15,18} on a subsample, rather than picked arbitrarily (scores: 0.068, 0.061, 0.064, 0.060, 0.066, 0.068 respectively — K=6 highest).

Resulting topics, by keyword: audio/headphones (sound, ear, speaker, bass, bluetooth), cameras (camera, lens, video, picture), tablet/case accessories (case, ipad, cover, keyboard), cables/connectivity (cable, usb, power, charge), computers/laptops (drive, laptop, screen, computer), and one broader general-positive/price-quality cluster.

**Important distinction**: silhouette scores stayed low (~0.06–0.07) even after tuning — this measures *embedding-space separation*, and real product reviews genuinely blend topics more than, say, news categories would. Keyword *coherence* (a separate claim) is strong once domain-specific praise vocabulary ("great," "works," "good") was excluded from the c-TF-IDF step, since ~75% of this corpus is 5-star reviews using near-identical enthusiastic language regardless of product category.

## Summarization

**Method**: extractive only (centroid-based) — embed each sentence, compute the review's overall meaning as the mean of its sentence embeddings, select whichever sentence(s) sit closest to that centroid. Chosen over an abstractive model because (a) this corpus has no reference summaries, so ROUGE evaluation is impossible either way, and (b) extractive selection cannot hallucinate content that wasn't in the original review. Reviews under 40 words are returned unchanged (they're already a summary).

**A real, disclosed weakness**: in a 71-word review about a laptop keyboard, centroid selection picked "I seem to wear out the space bars on my keyboards" — the least informative sentence — over more specific content about disassembly and hardware. With few candidate sentences, centrality can be dominated by generic shared vocabulary rather than actual informativeness. Across 6 sampled long reviews this was the only clear miss (5/6 were representative).

## Known, disclosed limitations (full list)

- Sentiment labels are derived from star ratings, a noisy proxy — 3-star reviews in particular often reflect genuinely mixed sentiment, not neutral sentiment. At least one real, observed example in the test set has text that reads unambiguously positive with a 1-star rating, suggesting label noise in the source data itself, independent of any model.
- "Mixed" sentiment is a derived label (aspect-level disagreement), not a directly trained model output.
- Aspect extraction is unsupervised; no gold aspect labels exist for this corpus, so it's evaluated qualitatively only. Semantic over-clustering (e.g. "amazon"/"retailer") is a known, undisclosed-severity limitation of the embedding-clustering approach.
- Aspect-level sentiment reuses the sentence-level model on each aspect's clause — a documented heuristic, not a trained aspect-sentiment classifier.
- Emotion detection is trained on Reddit comments, not product reviews; accuracy on this domain is unvalidated, with at least one concrete observed failure case.
- Topic model uses KMeans + c-TF-IDF rather than BERTopic, for Windows-compatibility reasons explained above. Embedding-space cluster separation is modest; keyword coherence is the stronger claim.
- Summarization is extractive; no reference summaries exist for this corpus, so no ROUGE score is reported. Centroid selection can pick a low-information sentence in reviews with few, short sentences.
- Model sizes and corpus sizes throughout (20k transformer training subsample, 15k topic-modeling sample, 8k aspect-vocabulary sample) were capped for tractability on 8 GB RAM / no-GPU local hardware, documented at each script rather than silently assumed.

## What was intentionally out of scope, given project time constraints

- A supervised, purpose-trained aspect-based sentiment classifier (e.g. fine-tuned on SemEval-2014 ABSA data) — would require in-domain gold labels this project doesn't have time to annotate.
- A Streamlit "Dataset Analysis" page showing aggregate distributions across many reviews — the single-review analysis page and the metrics already in `results/metrics/` cover the same ground for this portfolio scope.
- Full ROUGE-evaluated abstractive summarization.
