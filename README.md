# ReviewLens

Status: **in progress** — this README grows alongside each phase; nothing below is claimed until it's been run and measured.

## What this is

An end-to-end NLP system that analyzes customer reviews for sentiment, aspects,
aspect-level sentiment, emotion, recurring topics, and summaries — with an
explicit, documented separation between what's supervised-and-evaluated,
what's a transfer/pretrained model, and what's a disclosed heuristic.

## Hardware this was built and run on

- AMD Ryzen 5 7430U (6c/12t), 8 GB RAM (~7.4 GB usable), integrated Radeon
  graphics (no CUDA/ROCm training path)
- Windows 11, VS Code, PowerShell
- Training done on Google Colab (free tier); inference/dev locally on CPU

This constrains model sizes, corpus sizes, and the local/Colab split
documented throughout — see each phase's notes.

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

## Phase log

- [x] Phase 1 — Architecture & tech stack decisions
- [ ] Phase 2 — Dataset download & real stats (`scripts/download_data.py`)
- [ ] Phase 3 — Preprocessing
- [ ] Phase 4 — Baseline sentiment (TF-IDF + Logistic Regression / Linear SVM)
- [ ] Phase 5 — Transformer sentiment (DistilBERT, Colab-trained)
- [ ] Phase 6 — Aspect extraction (unsupervised) + heuristic aspect sentiment
- [ ] Phase 7 — Emotion detection (pretrained, flagged unvalidated in-domain)
- [ ] Phase 8 — Topic discovery (BERTopic)
- [ ] Phase 9 — Extractive summarization
- [ ] Phase 10 — Error analysis
- [ ] Phase 11 — Streamlit app
- [ ] Phase 12 — Cleanup & final docs

## Dataset

Source, license, sample counts and label distribution are filled in at the
end of Phase 2, from the actual output of `results/metrics/dataset_stats.json`
— not estimated here.

## Known, disclosed limitations

- Sentiment labels are derived from star ratings, not native sentiment
  annotations — 3-star reviews in particular are a noisy proxy for "neutral."
- "Mixed" sentiment is a derived label based on aspect-level disagreement,
  not a direct model output.
- Aspect extraction is unsupervised (dependency parsing + embedding
  clustering); no gold aspect labels exist for this corpus, so it is
  evaluated qualitatively only.
- Aspect-level sentiment reuses the sentence-level sentiment model applied to
  the clause containing the aspect — a documented heuristic, not a trained
  aspect-sentiment classifier.
- Emotion detection uses a model trained on a different domain (Reddit
  comments); its accuracy on product reviews is unvalidated.
- Summarization is extractive; no reference summaries exist for this corpus,
  so no ROUGE score is reported.
